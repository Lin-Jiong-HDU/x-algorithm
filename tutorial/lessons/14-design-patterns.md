# 第 14 节：系统设计亮点总结

> **学习目标**：提炼可复用的系统设计模式，建立对工业级推荐系统的完整认知

---

## 1. 架构模式总结

### 1.1 管道模式（Pipeline Pattern）

整个推荐系统采用**管道模式**，将复杂任务分解为一系列独立阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline Pattern                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐            │
│   │ Query   │──▶│ Source  │──▶│Hydrator │──▶│ Filter  │──▶ ...     │
│   │Hydrate │   │         │   │         │   │         │            │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘            │
│                                                                     │
│   优势：                                                            │
│   - 每个阶段职责单一，易于理解和测试                                │
│   - 可以灵活组合和替换                                              │
│   - 易于添加新功能                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Rust 实现**：通过 `CandidatePipeline` Trait 定义管道接口，各阶段通过独立 Trait 实现。

### 1.2 策略模式（Strategy Pattern）

每个管道阶段都采用**策略模式**：

```rust
// 策略接口
trait Filter<Q, C> {
    async fn filter(&self, query: &Q, candidates: Vec<C>) -> Result<FilterResult<C>, String>;
}

// 具体策略
struct AgeFilter { max_age: Duration }
struct SelfTweetFilter;
struct MutedKeywordFilter;

// 使用
fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
```

**优势**：
- 新增过滤逻辑只需实现 Trait
- 每个策略可独立测试
- 运行时可动态启用/禁用

### 1.3 组合模式（Composite Pattern）

通过组合多个组件构建复杂逻辑：

```rust
// 多个 Source 并行
let sources: Vec<Box<dyn Source<Q, C>>> = vec![
    Box::new(ThunderSource { ... }),
    Box::new(PhoenixSource { ... }),
];

// 多个 Filter 串行
let filters: Vec<Box<dyn Filter<Q, C>>> = vec![
    Box::new(DropDuplicatesFilter),
    Box::new(AgeFilter::new(...)),
    Box::new(SelfTweetFilter),
    // ...
];
```

---

## 2. 性能优化策略

### 2.1 并行执行

**原则**：独立操作并行执行，减少延迟

```
并行阶段：
├── Query Hydrators (join_all)
├── Sources (join_all)
├── Hydrators (join_all)
└── Side Effects (tokio::spawn + join_all)

串行阶段（有依赖）：
├── Filters (顺序执行，一个的输出是下一个的输入)
├── Scorers (顺序执行，分数逐步调整)
└── Selector (单次执行)
```

### 2.2 异步副作用

```rust
fn run_side_effects(&self, input: Arc<SideEffectInput<Q, C>>) {
    let side_effects = self.side_effects();
    tokio::spawn(async move {
        // 在后台执行，不阻塞响应
        let _ = join_all(side_effects.iter().map(|se| se.run(input))).await;
    });
}
```

**效果**：用户立即获得响应，副作用（缓存、日志）在后台完成。

### 2.3 内存缓存

Thunder 将最近帖子存储在内存中：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Thunder 内存缓存                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Kafka 帖子事件 ──▶ 内存 PostStore ──▶ 亚毫秒级查询               │
│                                                                     │
│   数据结构：                                                        │
│   HashMap<user_id, Vec<Post>>                                      │
│                                                                     │
│   优势：                                                            │
│   - 查询延迟 < 1ms（vs 数据库 10-100ms）                           │
│   - 支持高并发                                                      │
│   - 数据实时更新                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 并发控制

```rust
// Semaphore 限制最大并发
let request_semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

// 请求处理时
let _permit = match self.request_semaphore.try_acquire() {
    Ok(permit) => permit,
    Err(_) => return Err(Status::resource_exhausted("Server at capacity")),
};
// _permit 离开作用域时自动释放
```

---

## 3. 可靠性设计

### 3.1 优雅降级

```rust
// 单个组件失败不影响整体
for hydrator in hydrators {
    match hydrator.hydrate(&query).await {
        Ok(result) => hydrator.update(&mut query, result),
        Err(err) => {
            error!("hydrator {} failed: {}", hydrator.name(), err);
            // 继续执行，不中断
        }
    }
}
```

**原则**：推荐系统宁可给出不完美的结果，也不要完全失败。

### 3.2 详细日志

```rust
info!(
    "request_id={} stage={:?} component={} kept {} removed {}",
    request_id, stage, filter.name(), kept.len(), removed.len()
);
```

每个阶段都有详细日志，便于：
- 性能分析
- 问题排查
- A/B 测试对比

### 3.3 指标上报

```rust
GET_IN_NETWORK_POSTS_DURATION.observe(duration);
GET_IN_NETWORK_POSTS_COUNT.observe(posts.len());
```

---

## 4. ML 设计亮点

### 4.1 端到端学习

**传统推荐系统**：
```
手工特征工程 → 模型训练 → 部署
```

**X 的方案**：
```
用户行为序列 → Transformer → 行为概率
```

**优势**：
- 不需要领域专家设计特征
- 模型自动发现模式
- 减少工程复杂度

### 4.2 多任务学习

```python
# 同时预测 19 种行为
logits: [B, num_candidates, 19]

# 行为包括：
- 正向：like, reply, repost, click, share, dwell, follow_author
- 负向：not_interested, block_author, mute_author, report
```

**优势**：
- 共享底层表示，提高效率
- 负向行为帮助过滤不良内容

### 4.3 Candidate Isolation

```
传统：候选可以相互 attend
X：候选只能 attend User + History + 自己

好处：
- 分数一致性
- 可缓存性
- 批处理一致性
```

---

## 5. 可复用设计原则

### 5.1 关注点分离

| 模块 | 职责 |
|------|------|
| `candidate-pipeline` | 定义管道框架 |
| `home-mixer` | 组装业务管道 |
| `thunder` | 实时数据存储 |
| `phoenix` | ML 模型 |

### 5.2 接口与实现分离

```rust
// 接口（Trait）
trait Source<Q, C> {
    async fn get_candidates(&self, query: &Q) -> Result<Vec<C>, String>;
}

// 实现
struct ThunderSource { ... }
struct PhoenixSource { ... }
```

### 5.3 配置驱动

```rust
// 通过配置控制行为
let filters: Vec<Box<dyn Filter<_, _>>> = vec![
    Box::new(AgeFilter::new(Duration::from_secs(params::MAX_POST_AGE))),
    // params::MAX_POST_AGE 可通过配置文件修改
];
```

---

## 6. 可扩展性

### 6.1 添加新的候选来源

```rust
// 1. 实现 Source Trait
struct TrendingSource {
    trending_client: Arc<TrendingClient>,
}

impl Source<ScoredPostsQuery, PostCandidate> for TrendingSource {
    async fn get_candidates(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        self.trending_client.get_trending_posts().await
    }
}

// 2. 添加到管道
sources.push(Box::new(TrendingSource { trending_client }));
```

### 6.2 添加新的过滤逻辑

```rust
// 1. 实现 Filter Trait
struct LanguageFilter {
    preferred_languages: Vec<String>,
}

impl Filter<ScoredPostsQuery, PostCandidate> for LanguageFilter {
    async fn filter(&self, query: &ScoredPostsQuery, candidates: Vec<PostCandidate>)
        -> Result<FilterResult<PostCandidate>, String>
    {
        // 过滤逻辑
    }
}

// 2. 添加到管道（注意顺序）
filters.insert(5, Box::new(LanguageFilter::new()));
```

### 6.3 添加新的行为预测

```python
# 1. 修改 PhoenixScores
pub struct PhoenixScores {
    // 现有字段...
    pub bookmark_score: Option<f64>,  // 新增
}

# 2. 修改 num_actions 配置
num_actions: 20  // 19 + 1

# 3. 添加权重
BOOKMARK_WEIGHT: f64 = 0.5
```

---

## 7. 思考题

1. **这个架构可以应用到哪些其他场景？**（如电商推荐、内容分发）

2. **如果要支持实时个性化（用户行为立即影响推荐），需要修改哪些部分？**

3. **Thunder 的内存存储有什么局限性？如何解决？**

4. **如何评估推荐系统的效果？需要哪些指标？**

---

## 小结

本节我们总结了 X For You 推荐系统的设计亮点：

| 类别 | 设计点 |
|------|--------|
| **架构模式** | 管道模式、策略模式、组合模式 |
| **性能优化** | 并行执行、异步副作用、内存缓存、并发控制 |
| **可靠性** | 优雅降级、详细日志、指标上报 |
| **ML 设计** | 端到端学习、多任务学习、Candidate Isolation |
| **可扩展性** | Trait 抽象、配置驱动、关注点分离 |

这些设计原则可以应用到任何大规模推荐系统或其他数据处理管道中。

下一节，我们将通过实战项目巩固所学知识。
