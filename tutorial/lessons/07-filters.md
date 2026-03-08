# 第 7 节：Filters - 过滤系统

> **学习目标**：掌握推荐系统中过滤逻辑的设计，理解 Pre-Scoring 和 Post-Selection 过滤的区别

---

## 1. 概念讲解

### 1.1 为什么需要过滤？

推荐系统获取的候选中，很多是不应该展示给用户的：

- 用户已经看过的帖子
- 用户屏蔽/静音的作者
- 包含用户屏蔽关键词的帖子
- 太老的帖子
- 自己发的帖子
- ...

过滤系统负责移除这些不合适的候选。

### 1.2 两阶段过滤

```
┌─────────────────────────────────────────────────────────────────────┐
│                         过滤系统架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   Pre-Scoring Filters (10个)                  │   │
│   │                                                              │   │
│   │   在 ML 评分之前执行，减少需要评分的候选数量                  │   │
│   │                                                              │   │
│   │   1. DropDuplicatesFilter      - 去重                      │   │
│   │   2. CoreDataHydrationFilter   - 过滤无法获取核心数据的     │   │
│   │   3. AgeFilter                 - 时效性过滤                 │   │
│   │   4. SelfTweetFilter           - 过滤自己的帖子             │   │
│   │   5. RetweetDeduplicationFilter - 转发去重                  │   │
│   │   6. IneligibleSubscriptionFilter - 订阅内容过滤            │   │
│   │   7. PreviouslySeenPostsFilter - 已看过的帖子               │   │
│   │   8. PreviouslyServedPostsFilter - 本次会话已推荐的          │   │
│   │   9. MutedKeywordFilter        - 关键词屏蔽               │   │
│   │   10. AuthorSocialgraphFilter  - 屏蔽/静音用户             │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                         ML 评分                                     │
│                              │                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                  Post-Selection Filters (2个)                 │   │
│   │                                                              │   │
│   │   在选择之后执行，处理需要额外检查的候选                       │   │
│   │                                                              │   │
│   │   1. VFFilter                - 可见性过滤（删除/违规等）      │   │
│   │   2. DedupConversationFilter  - 对话去重                     │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么分两阶段？**

- **Pre-Scoring**：减少需要评分的候选数量，节省 ML 计算成本
- **Post-Selection**：一些检查（如可见性）需要额外信息，且只对最终选中的候选执行

---

## 2. 代码分析

### 2.1 DropDuplicatesFilter

最简单的过滤器——去除重复的帖子 ID。

```rust
pub struct DropDuplicatesFilter;

#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for DropDuplicatesFilter {
    async fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> Result<FilterResult<PostCandidate>, String> {
        let mut seen_ids = HashSet::new();
        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for candidate in candidates {
            if seen_ids.insert(candidate.tweet_id) {
                // 第一次见到，保留
                kept.push(candidate);
            } else {
                // 重复，移除
                removed.push(candidate);
            }
        }

        Ok(FilterResult { kept, removed })
    }
}
```

**设计要点**：
- 使用 `HashSet` 的 `insert` 返回值判断是否重复
- 返回 `FilterResult` 同时包含保留和移除的候选（用于日志）

### 2.2 AgeFilter

过滤超过时效性限制的帖子。

```rust
pub struct AgeFilter {
    pub max_age: Duration,
}

impl AgeFilter {
    fn is_within_age(&self, tweet_id: i64) -> bool {
        // 从 Snowflake ID 解析创建时间
        snowflake::duration_since_creation_opt(tweet_id)
            .map(|age| age <= self.max_age)
            .unwrap_or(false)
    }
}

#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for AgeFilter {
    async fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> Result<FilterResult<PostCandidate>, String> {
        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| self.is_within_age(c.tweet_id));

        Ok(FilterResult { kept, removed })
    }
}
```

**设计要点**：
- 利用 Snowflake ID 的时间戳特性判断帖子年龄
- 使用 `partition` 一次性分割集合

### 2.3 AuthorSocialgraphFilter

过滤用户屏蔽或静音的作者。

```rust
pub struct AuthorSocialgraphFilter;

#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for AuthorSocialgraphFilter {
    async fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 从用户特征中获取屏蔽/静音列表
        let blocked_users: HashSet<_> = query.user_features.blocked_user_ids
            .iter()
            .cloned()
            .collect();

        let muted_users: HashSet<_> = query.user_features.muted_user_ids
            .iter()
            .cloned()
            .collect();

        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| {
                !blocked_users.contains(&(c.author_id as u64))
                    && !muted_users.contains(&(c.author_id as u64))
            });

        Ok(FilterResult { kept, removed })
    }
}
```

**设计要点**：
- 依赖 `QueryHydrator` 获取的用户特征
- 屏蔽和静音都需要检查

### 2.4 Filter 执行顺序的重要性

```rust
let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
    Box::new(DropDuplicatesFilter),           // 1. 先去重，减少后续处理量
    Box::new(CoreDataHydrationFilter),        // 2. 确保有核心数据
    Box::new(AgeFilter::new(Duration::from_secs(params::MAX_POST_AGE))), // 3. 时效性
    Box::new(SelfTweetFilter),               // 4. 过滤自己
    Box::new(RetweetDeduplicationFilter),    // 5. 转发去重
    Box::new(IneligibleSubscriptionFilter),  // 6. 订阅内容
    Box::new(PreviouslySeenPostsFilter),     // 7. 已看过
    Box::new(PreviouslyServedPostsFilter),   // 8. 本次已推荐
    Box::new(MutedKeywordFilter::new()),     // 9. 关键词（需要文本数据）
    Box::new(AuthorSocialgraphFilter),       // 10. 社交关系
];
```

**顺序设计原则**：
1. **减少后续处理量**：去重最先
2. **数据依赖**：需要文本的过滤靠后
3. **用户偏好**：个性化过滤靠后

---

## 3. 实践练习

### 动手任务

**实现 LanguageFilter**：过滤非用户偏好语言的帖子

```rust
pub struct LanguageFilter;

#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for LanguageFilter {
    async fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> Result<FilterResult<PostCandidate>, String> {
        // TODO: 实现
        // 提示：使用 query.language_code 和 candidate.detected_language
    }
}
```

### 思考题

1. **为什么 `DropDuplicatesFilter` 放在最前面？**
2. **`MutedKeywordFilter` 为什么放在 `AuthorSocialgraphFilter` 前面？**
3. **Post-Selection 阶段的 `VFFilter` 能放在 Pre-Scoring 吗？为什么？**

---

## 小结

本节我们理解了过滤系统的设计：

1. **两阶段过滤**：Pre-Scoring（减少计算量）+ Post-Selection（最终检查）
2. **10 个 Pre-Scoring Filters**：去重、时效、用户偏好等
3. **2 个 Post-Selection Filters**：可见性、对话去重
4. **执行顺序很重要**：数据依赖和处理效率决定顺序

下一节，我们将进入 Scorers，看系统如何给候选帖子评分。
