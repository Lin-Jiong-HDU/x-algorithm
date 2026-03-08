# 第 6 节：Sources - 候选来源

> **学习目标**：理解 In-Network 和 Out-of-Network 两种候选来源的工作原理

---

## 1. 概念讲解

### 1.1 两种候选来源

X For You Feed 的候选来自两个来源：

| 来源 | 说明 | 实现方式 |
|------|------|----------|
| **In-Network** | 用户关注的人发布的帖子 | Thunder 服务（内存存储） |
| **Out-of-Network** | 全局发现的相关帖子 | Phoenix Retrieval（ML 检索） |

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Candidate Sources                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────┐      ┌─────────────────────┐              │
│   │    ThunderSource    │      │   PhoenixSource     │              │
│   │    (In-Network)     │      │   (Out-of-Network)  │              │
│   │                     │      │                     │              │
│   │  关注用户的帖子      │      │  ML 检索全局候选     │              │
│   │  ~几百条             │      │  ~几百条             │              │
│   │                     │      │                     │              │
│   │  特点：              │      │  特点：              │              │
│   │  - 时效性好          │      │  - 发现新内容        │              │
│   │  - 用户明确关注      │      │  - 扩展兴趣范围      │              │
│   │  - 实时获取          │      │  - 基于行为相似性    │              │
│   └─────────────────────┘      └─────────────────────┘              │
│                │                          │                         │
│                └──────────┬───────────────┘                         │
│                           ▼                                         │
│                    合并候选（~千条）                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么需要两个来源？

**只有 In-Network 的问题**：
- 信息茧房：只看到关注的内容
- 错过热门/重要内容
- 难以发现新兴趣

**只有 Out-of-Network 的问题**：
- 可能推荐不相关内容
- 忽略用户明确关注的更新
- 缺乏社交连接感

**两者结合**：平衡相关性和多样性。

---

## 2. 代码分析

### 2.1 ThunderSource（In-Network）

**文件**：`home-mixer/sources/thunder_source.rs`

```rust
pub struct ThunderSource {
    pub thunder_client: Arc<ThunderClient>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for ThunderSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        // 如果请求只要求 OON，则禁用
        !query.in_network_only
    }

    async fn get_candidates(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let response = self.thunder_client
            .get_in_network_posts(
                query.user_id,
                query.user_features.following_user_ids.clone(),
                query.seen_ids.clone(),
                false,  // not video only
            )
            .await
            .map_err(|e| format!("Thunder request failed: {}", e))?;

        let candidates: Vec<PostCandidate> = response
            .posts
            .into_iter()
            .map(|post| PostCandidate {
                tweet_id: post.post_id,
                author_id: post.author_id as u64,
                in_reply_to_tweet_id: post.in_reply_to_post_id,
                ancestors: post.conversation_ancestors,
                served_type: Some(ServedType::ForYouInNetwork),
                ..Default::default()
            })
            .collect();

        Ok(candidates)
    }
}
```

**关键点**：
- 使用 `following_user_ids` 获取关注用户的帖子
- 标记 `in_network = true`，后续处理可区分来源

### 2.2 PhoenixSource（Out-of-Network）

**文件**：`home-mixer/sources/phoenix_source.rs`

```rust
pub struct PhoenixSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixSource {
    async fn get_candidates(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id as u64;

        // 必须有用户行为序列才能做 ML 检索
        let sequence = query
            .user_action_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixSource: missing user_action_sequence".to_string())?;

        // 调用 Phoenix Retrieval 服务
        let response = self.phoenix_retrieval_client
            .retrieve(user_id, sequence.clone())
            .await
            .map_err(|e| format!("Phoenix retrieval failed: {}", e))?;

        // 转换为候选
        let candidates: Vec<PostCandidate> = response
            .candidates
            .into_iter()
            .map(|c| PostCandidate {
                tweet_id: c.tweet_id as i64,
                author_id: c.author_id,
                served_type: Some(ServedType::ForYouOutOfNetwork),
                ..Default::default()
            })
            .collect();

        Ok(candidates)
    }
}
```

**关键点**：
- 依赖 `user_action_sequence`（Query Hydrator 填充）
- 调用 Phoenix Retrieval 的 Two-Tower 模型
- 标记 `in_network = false`

### 2.3 并行执行与合并

在 `candidate_pipeline.rs` 中：

```rust
async fn fetch_candidates(&self, query: &Q) -> Vec<C> {
    let sources: Vec<_> = self.sources()
        .iter()
        .filter(|s| s.enable(query))
        .collect();

    // 并行执行所有 Source
    let source_futures = sources.iter().map(|s| s.get_candidates(query));
    let results = join_all(source_futures).await;

    // 合并结果
    let mut collected = Vec::new();
    for (source, result) in sources.iter().zip(results) {
        match result {
            Ok(mut candidates) => {
                info!("source {} fetched {} candidates", source.name(), candidates.len());
                collected.append(&mut candidates);
            }
            Err(err) => {
                error!("source {} failed: {}", source.name(), err);
                // 单个 Source 失败不影响其他
            }
        }
    }
    collected
}
```

**合并策略**：简单地将两个来源的候选合并到一个列表中。

---

## 3. 实践练习

### 思考题

1. **为什么 ThunderSource 有 `enable()` 检查，而 PhoenixSource 没有？**

2. **如果 Thunder 服务不可用，推荐系统会如何表现？**

3. **In-Network 和 Out-of-Network 候选的比例如何控制？是在 Source 阶段还是后续阶段？**

### 代码阅读

查看 `home-mixer/sources/mod.rs`，确认两个 Source 是如何被注册的。

---

## 小结

本节我们理解了两种候选来源：

1. **ThunderSource（In-Network）**：从关注用户获取帖子
2. **PhoenixSource（Out-of-Network）**：ML 检索全局相关帖子
3. **并行执行**：两个 Source 同时获取，减少延迟
4. **容错设计**：单个 Source 失败不影响另一个

下一节，我们将进入 Filters，看系统如何过滤不合适的候选。
