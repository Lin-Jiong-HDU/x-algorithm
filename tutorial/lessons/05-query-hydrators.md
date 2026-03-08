# 第 5 节：Query Hydrators - 用户上下文获取

> **学习目标**：理解 Query Hydrator 如何获取用户上下文，为后续推荐提供关键信息

---

## 1. 概念讲解

### 1.1 为什么需要 Query Hydrator？

推荐系统的核心是个性化——根据用户特征推荐不同内容。但在收到推荐请求时，我们只有基本的用户 ID，需要**额外获取**用户的上下文信息：

```
请求输入                      需要获取
─────────────────────────────────────────────
user_id: 12345         →     用户行为序列（最近点赞了什么）
client_app_id: 1       →     用户特征（关注列表、偏好）
country_code: "US"     →
language_code: "en"    →
```

这些信息对于：
- **候选获取**：知道用户关注了谁，才能获取 In-Network 候选
- **ML 评分**：用户行为序列是 Transformer 模型的关键输入
- **过滤**：根据用户偏好过滤不感兴趣的内容

### 1.2 两个 Query Hydrator

Home Mixer 使用两个 Query Hydrator：

| Hydrator | 职责 | 获取的信息 |
|----------|------|-----------|
| `UserActionSeqQueryHydrator` | 用户行为序列 | 最近点赞、回复、转发的帖子 |
| `UserFeaturesQueryHydrator` | 用户特征 | 关注列表、用户偏好设置 |

---

## 2. 代码分析

### 2.1 UserActionSeqQueryHydrator

**文件**：`home-mixer/query_hydrators/user_action_seq_query_hydrator.rs`

这个 Hydrator 获取用户的**历史行为序列**，用于 ML 模型的输入。

#### 处理流程

```rust
async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
    // 1. 从 UAS 服务获取原始行为序列
    let uas_thrift = self.uas_fetcher.get_by_user_id(query.user_id).await?;

    // 2. 聚合处理
    let aggregated = self.aggregate_user_action_sequence(query.user_id, uas_thrift)?;

    // 3. 返回增强后的 Query
    Ok(ScoredPostsQuery {
        user_action_sequence: Some(aggregated),
        ..Default::default()
    })
}
```

#### 聚合处理详解

```rust
fn aggregate_user_action_sequence(&self, user_id: i64, uas_thrift: ThriftUserActionSequence)
    -> Result<UserActionSequence, String>
{
    // 1. 提取原始行为
    let actions = uas_thrift.user_actions.unwrap_or_default();

    // 2. 预过滤：移除无效行为
    let filtered = self.global_filter.run(actions);

    // 3. 聚合：按时间窗口合并连续行为
    let mut aggregated = self.aggregator.run(&filtered, params::UAS_WINDOW_TIME_MS, 0);

    // 4. 后过滤
    for filter in &self.post_filters {
        aggregated = filter.run(aggregated);
    }

    // 5. 截断到最大长度
    if aggregated.len() > params::UAS_MAX_SEQUENCE_LENGTH {
        aggregated.drain(0..aggregated.len() - params::UAS_MAX_SEQUENCE_LENGTH);
    }

    // 6. 转换为 Proto 格式
    convert_to_proto_sequence(user_id, metadata, aggregated, self.aggregator.name())
}
```

**为什么需要聚合？**
- **减少数据量**：合并相似行为
- **提取模式**：识别用户兴趣焦点
- **提高效率**：减少 ML 模型输入长度

### 2.2 UserFeaturesQueryHydrator

```rust
pub struct UserFeaturesQueryHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
    let result = self.strato_client.get_user_features(query.user_id).await?;
    let user_features = decode(&result)?.v.unwrap_or_default();

    Ok(ScoredPostsQuery {
        user_features,
        ..Default::default()
    })
}
```

**UserFeatures 包含**：关注列表、静音/屏蔽列表、语言偏好等。

### 2.3 并行执行

```rust
async fn hydrate_query(&self, query: Q) -> Q {
    let hydrators: Vec<_> = self.query_hydrators()
        .iter()
        .filter(|h| h.enable(&query))
        .collect();

    // 并行执行
    let results = join_all(hydrators.iter().map(|h| h.hydrate(&query))).await;

    // 合并结果
    let mut hydrated_query = query;
    for (hydrator, result) in hydrators.iter().zip(results) {
        match result {
            Ok(hydrated) => hydrator.update(&mut hydrated_query, hydrated),
            Err(err) => error!("hydrator {} failed: {}", hydrator.name(), err),
        }
    }
    hydrated_query
}
```

**性能优势**：并行执行只需 max(50ms, 50ms) = 50ms，而非串行的 100ms。

---

## 3. 实践练习

### 思考题

1. **为什么用户行为序列需要截断到最大长度？**
2. **如果 UserActionSeqQueryHydrator 失败了，推荐请求会失败吗？**
3. **如何处理冷启动用户（无历史行为）？**

### 代码挑战

实现一个 `DeviceContextQueryHydrator`，获取用户的设备信息：

```rust
pub struct DeviceContextQueryHydrator {
    pub device_client: Arc<DeviceClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for DeviceContextQueryHydrator {
    // TODO: 实现 hydrate 和 update 方法
}
```

---

## 小结

本节我们理解了 Query Hydrator 的设计：

1. **目的**：在推荐前获取用户上下文信息
2. **两个 Hydrator**：
   - `UserActionSeqQueryHydrator`：获取用户行为序列（ML 输入）
   - `UserFeaturesQueryHydrator`：获取用户特征（过滤依据）
3. **并行执行**：减少延迟
4. **错误处理**：单个失败不影响整体

下一节，我们将进入 Sources，看系统如何从两个来源获取候选帖子。
