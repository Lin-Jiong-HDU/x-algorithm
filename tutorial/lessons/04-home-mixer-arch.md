# 第 4 节：Home Mixer 整体架构

> **学习目标**：理解 Home Mixer 如何组装完整的推荐管道，掌握 gRPC 服务设计

---

## 1. 概念讲解

### 1.1 Home Mixer 的定位

Home Mixer 是 X For You 推荐系统的**编排协调层**（Orchestration Layer）。它的职责是：

1. **接收请求**：暴露 gRPC 接口，接收客户端的推荐请求
2. **组装管道**：将各种组件（Source、Filter、Scorer 等）组装成完整的推荐管道
3. **执行管道**：调用 `CandidatePipeline::execute()` 完成推荐流程
4. **返回结果**：将处理后的候选转换为响应格式返回

```
┌─────────────────────────────────────────────────────────────────────┐
│                           客户端请求                                 │
│                    "给我推荐 20 条帖子"                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Home Mixer                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    PhoenixCandidatePipeline                  │    │
│  │                                                              │    │
│  │   QueryHydrators → Sources → Hydrators → Filters            │    │
│  │        → Scorers → Selector → PostSelection                 │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                    │                                 │
│                                    ▼                                 │
│                          转换为 ScoredPost                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          返回排序后的 Feed
```

### 1.2 目录结构

```
home-mixer/
├── main.rs                    # 服务入口，启动 gRPC 服务器
├── server.rs                  # gRPC 服务实现
├── lib.rs                     # 模块导出
├── candidate_pipeline/        # 管道定义
│   ├── mod.rs
│   ├── query.rs               # ScoredPostsQuery 定义
│   ├── candidate.rs           # PostCandidate 定义
│   ├── query_features.rs      # 用户特征
│   ├── candidate_features.rs  # 候选特征
│   └── phoenix_candidate_pipeline.rs  # 管道组装
├── query_hydrators/           # 查询增强器
├── sources/                   # 候选来源
├── candidate_hydrators/       # 候选增强器
├── filters/                   # 过滤器
├── scorers/                   # 评分器
├── selectors/                 # 选择器
├── side_effects/              # 副作用
└── clients/                   # 外部服务客户端
```

---

## 2. 代码分析

### 2.1 服务入口：main.rs

```rust
#[derive(Parser, Debug)]
#[command(about = "HomeMixer gRPC Server")]
struct Args {
    #[arg(long)]
    grpc_port: u16,
    #[arg(long)]
    metrics_port: u16,
    #[arg(long)]
    reload_interval_minutes: u64,
    #[arg(long)]
    chunk_size: usize,
}

#[xai_stats_macro::main(name = "home-mixer")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    xai_init_utils::init().log();
    xai_init_utils::init().rustls();

    // 创建服务实例
    let service = HomeMixerServer::new().await;

    // 配置 gRPC 反射服务（用于调试）
    let reflection_service = Builder::configure()
        .register_encoded_file_descriptor_set(pb::FILE_DESCRIPTOR_SET)
        .build_v1()?;

    // 配置 gRPC 路由
    let mut grpc_routes = RoutesBuilder::default();
    grpc_routes.add_service(
        pb::scored_posts_service_server::ScoredPostsServiceServer::new(service)
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
    );

    // 启动服务器
    let mut server = HttpServer::new(
        args.metrics_port,
        http_router,
        Some(grpc_config),
        CancellationToken::new(),
        Duration::from_secs(20),
    )
    .await?;

    server.set_readiness(true);
    server.wait_for_termination().await;
    Ok(())
}
```

**关键配置**：

- **消息大小限制**：`max_decoding_message_size` / `max_encoding_message_size`
- **压缩支持**：Gzip 和 Zstd
- **优雅关闭**：`CancellationToken` + 超时

### 2.2 gRPC 服务实现：server.rs

```rust
pub struct HomeMixerServer {
    phx_candidate_pipeline: Arc<PhoenixCandidatePipeline>,
}

impl HomeMixerServer {
    pub async fn new() -> Self {
        HomeMixerServer {
            phx_candidate_pipeline: Arc::new(PhoenixCandidatePipeline::prod().await),
        }
    }
}

#[tonic::async_trait]
impl pb::scored_posts_service_server::ScoredPostsService for HomeMixerServer {
    async fn get_scored_posts(
        &self,
        request: Request<pb::ScoredPostsQuery>,
    ) -> Result<Response<ScoredPostsResponse>, Status> {
        let proto_query = request.into_inner();

        // 1. 参数验证
        if proto_query.viewer_id == 0 {
            return Err(Status::invalid_argument("viewer_id must be specified"));
        }

        let start = Instant::now();

        // 2. 转换为内部 Query 类型
        let query = ScoredPostsQuery::new(
            proto_query.viewer_id,
            proto_query.client_app_id,
            proto_query.country_code,
            proto_query.language_code,
            proto_query.seen_ids,
            proto_query.served_ids,
            proto_query.in_network_only,
            proto_query.is_bottom_request,
            proto_query.bloom_filter_entries,
        );

        // 3. 执行推荐管道
        let pipeline_result = self.phx_candidate_pipeline.execute(query).await;

        // 4. 转换为响应格式
        let scored_posts: Vec<ScoredPost> = pipeline_result
            .selected_candidates
            .into_iter()
            .map(|candidate| ScoredPost {
                tweet_id: candidate.tweet_id as u64,
                author_id: candidate.author_id,
                score: candidate.score.unwrap_or(0.0) as f32,
                // ... 其他字段
            })
            .collect();

        info!(
            "Scored Posts response - {} posts ({} ms)",
            scored_posts.len(),
            start.elapsed().as_millis()
        );

        Ok(Response::new(ScoredPostsResponse { scored_posts }))
    }
}
```

**处理流程**：

1. **参数验证**：检查必要字段
2. **类型转换**：Proto → 内部类型
3. **执行管道**：调用 `execute()`
4. **类型转换**：内部类型 → Proto
5. **返回响应**

### 2.3 管道组装：phoenix_candidate_pipeline.rs

这是 Home Mixer 最核心的文件，展示了如何组装完整的推荐管道：

```rust
pub struct PhoenixCandidatePipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>>,
    hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>>,
    selector: TopKScoreSelector,
    post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>>,
}
```

#### Query Hydrators（2 个）

```rust
let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
    Box::new(UserActionSeqQueryHydrator::new(uas_fetcher)),  // 用户行为序列
    Box::new(UserFeaturesQueryHydrator { strato_client }),   // 用户特征（关注列表等）
];
```

#### Sources（2 个，并行）

```rust
let sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>> = vec![
    Box::new(PhoenixSource { phoenix_retrieval_client }),  // Out-of-Network
    Box::new(ThunderSource { thunder_client }),            // In-Network
];
```

#### Hydrators（5 个，并行）

```rust
let hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
    Box::new(InNetworkCandidateHydrator),                          // 标记来源
    Box::new(CoreDataCandidateHydrator::new(tes_client).await),    // 核心数据（文本等）
    Box::new(VideoDurationCandidateHydrator::new(tes_client).await), // 视频时长
    Box::new(SubscriptionHydrator::new(tes_client).await),         // 订阅信息
    Box::new(GizmoduckCandidateHydrator::new(gizmoduck_client).await), // 作者信息
];
```

#### Filters（10 个，顺序）

```rust
let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
    Box::new(DropDuplicatesFilter),           // 1. 去重
    Box::new(CoreDataHydrationFilter),        // 2. 过滤未获取到核心数据的
    Box::new(AgeFilter::new(Duration::from_secs(params::MAX_POST_AGE))),  // 3. 时效性
    Box::new(SelfTweetFilter),                // 4. 过滤自己的帖子
    Box::new(RetweetDeduplicationFilter),     // 5. 转发去重
    Box::new(IneligibleSubscriptionFilter),   // 6. 订阅内容过滤
    Box::new(PreviouslySeenPostsFilter),      // 7. 已看过的
    Box::new(PreviouslyServedPostsFilter),    // 8. 本次会话已推荐的
    Box::new(MutedKeywordFilter::new()),      // 9. 关键词屏蔽
    Box::new(AuthorSocialgraphFilter),        // 10. 屏蔽/静音用户
];
```

**为什么是这个顺序？**

- **去重最先**：避免后续处理重复数据
- **核心数据检查第二**：确保后续 Filter 有数据可用
- **时效性第三**：尽早过滤旧内容，减少后续处理量
- **用户偏好相关最后**：需要完整数据才能判断

#### Scorers（4 个，顺序）

```rust
let scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>> = vec![
    Box::new(PhoenixScorer { phoenix_client }),      // 1. ML 预测
    Box::new(WeightedScorer),                         // 2. 加权组合
    Box::new(AuthorDiversityScorer::default()),       // 3. 作者多样性
    Box::new(OONScorer),                              // 4. Out-of-Network 调整
];
```

**顺序的原因**：

1. `PhoenixScorer` 输出各种行为概率
2. `WeightedScorer` 基于概率计算加权分数
3. `AuthorDiversityScorer` 基于分数调整重复作者
4. `OONScorer` 最后调整网络外内容

#### Post-Selection

```rust
// Hydrators
let post_selection_hydrators = vec![
    Box::new(VFCandidateHydrator::new(vf_client).await),  // 可见性信息
];

// Filters
let post_selection_filters = vec![
    Box::new(VFFilter),              // 可见性过滤（删除/违规等）
    Box::new(DedupConversationFilter), // 对话去重
];
```

#### Side Effects

```rust
let side_effects = Arc::new(vec![
    Box::new(CacheRequestInfoSideEffect { strato_client }),  // 缓存请求信息
]);
```

### 2.4 数据类型定义

#### Query 类型

```rust
#[derive(Clone, Default, Debug)]
pub struct ScoredPostsQuery {
    pub user_id: i64,
    pub client_app_id: i32,
    pub country_code: String,
    pub language_code: String,
    pub seen_ids: Vec<i64>,           // 已看过的帖子 ID
    pub served_ids: Vec<i64>,         // 本次会话已推荐的
    pub in_network_only: bool,
    pub is_bottom_request: bool,
    pub bloom_filter_entries: Vec<ImpressionBloomFilterEntry>,
    pub user_action_sequence: Option<UserActionSequence>,  // QueryHydrator 填充
    pub user_features: UserFeatures,                       // QueryHydrator 填充
    pub request_id: String,
}
```

#### Candidate 类型

```rust
#[derive(Clone, Debug, Default)]
pub struct PostCandidate {
    pub tweet_id: i64,
    pub author_id: u64,
    pub tweet_text: String,
    pub in_reply_to_tweet_id: Option<u64>,
    pub retweeted_tweet_id: Option<u64>,
    pub retweeted_user_id: Option<u64>,

    // ML 分数（PhoenixScorer 填充）
    pub phoenix_scores: PhoenixScores,
    pub prediction_request_id: Option<u64>,
    pub last_scored_at_ms: Option<u64>,

    // 最终分数（WeightedScorer 填充）
    pub weighted_score: Option<f64>,
    pub score: Option<f64>,

    // 元数据（各 Hydrator 填充）
    pub in_network: Option<bool>,
    pub video_duration_ms: Option<i32>,
    pub author_screen_name: Option<String>,
    pub visibility_reason: Option<FilteredReason>,
    // ...
}

#[derive(Clone, Debug, Default)]
pub struct PhoenixScores {
    pub favorite_score: Option<f64>,
    pub reply_score: Option<f64>,
    pub retweet_score: Option<f64>,
    pub click_score: Option<f64>,
    // ... 更多行为分数
}
```

---

## 3. 实践练习

### 动手任务

1. **绘制完整的组件依赖图**
   - 标注每个组件的输入来源和输出目标
   - 用箭头表示数据流向

2. **分析为什么 Filter 的顺序是这样安排的**
   - 如果交换 `AgeFilter` 和 `MutedKeywordFilter` 的顺序，会有什么影响？

### 思考题

1. **为什么 `HomeMixerServer` 只持有一个 `PhoenixCandidatePipeline`，而不是每次请求创建一个新的？**

   <details>
   <summary>提示</summary>

   考虑：Pipeline 内部的组件是否是有状态的？创建成本如何？

   </details>

2. **如果需要添加一个新的 Source（如"热门帖子"），需要修改哪些文件？**

3. **Post-Selection 阶段的 Filter 为什么放在选择之后而不是之前？**

---

## 小结

本节我们理解了 Home Mixer 的整体架构：

1. **服务入口**：`main.rs` 启动 gRPC 服务器
2. **服务实现**：`server.rs` 处理请求，调用管道
3. **管道组装**：`phoenix_candidate_pipeline.rs` 组装所有组件
4. **数据类型**：`ScoredPostsQuery` 和 `PostCandidate`

**关键洞察**：Home Mixer 本身不包含推荐逻辑，它只是一个**组装者和协调者**。所有推荐逻辑都封装在各个 Trait 实现中，这实现了关注点的完美分离。

下一节，我们将深入 Query Hydrators，看系统如何获取用户上下文。
