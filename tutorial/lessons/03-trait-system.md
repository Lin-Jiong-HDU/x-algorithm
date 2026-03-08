# 第 3 节：Trait 系统详解

> **学习目标**：掌握 7 个 Trait 的职责定义和实现模式，理解框架的可扩展性设计

---

## 1. 概念讲解

### 1.1 Trait 系统概览

`candidate-pipeline` 定义了 7 个 Trait，每个 Trait 对应管道的一个阶段：

| Trait | 职责 | 执行方式 | 输入 | 输出 |
|-------|------|----------|------|------|
| `QueryHydrator` | 丰富查询信息 | 并行 | `Q` | `Q` |
| `Source` | 获取候选 | 并行 | `Q` | `Vec<C>` |
| `Hydrator` | 丰富候选信息 | 并行 | `Q`, `&[C]` | `Vec<C>` |
| `Filter` | 过滤候选 | 顺序 | `Q`, `Vec<C>` | `FilterResult<C>` |
| `Scorer` | 评分 | 顺序 | `Q`, `&[C]` | `Vec<C>` |
| `Selector` | 排序选择 | 单次 | `Q`, `Vec<C>` | `Vec<C>` |
| `SideEffect` | 副作用 | 并行(异步) | `SideEffectInput<Q,C>` | `()` |

### 1.2 设计模式：策略模式 + 组合模式

这个 Trait 系统是**策略模式**的经典应用：

```rust
// 策略接口
trait Filter<Q, C> {
    async fn filter(&self, query: &Q, candidates: Vec<C>) -> Result<FilterResult<C>, String>;
}

// 具体策略
struct AgeFilter { max_age_seconds: u64 }
struct SelfTweetFilter { }
struct MutedKeywordFilter { }

// 上下文（管道）持有多个策略
fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
```

同时，通过返回多个 Trait 实现（`&[Box<dyn Trait>]`），实现了**组合模式**——管道由多个组件组合而成。

---

## 2. 代码分析

### 2.1 QueryHydrator Trait

**文件**：`candidate-pipeline/query_hydrator.rs`

```rust
#[async_trait]
pub trait QueryHydrator<Q>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
{
    /// 是否启用（默认 true）
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 执行增强，返回增强后的 Query
    async fn hydrate(&self, query: &Q) -> Result<Q, String>;

    /// 将增强结果合并回原 Query
    fn update(&self, query: &mut Q, hydrated: Q);

    /// 组件名称（用于日志）
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

**设计要点**：

1. **`enable()` 方法**：允许根据 Query 动态决定是否执行
2. **`hydrate()` 返回新的 `Q`**：不直接修改，避免并发问题
3. **`update()` 合并结果**：每个 Hydrator 只负责自己关心的字段

**实现示例**：

```rust
struct UserActionSeqHydrator {
    client: Arc<UserActionClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for UserActionSeqHydrator {
    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let sequence = self.client
            .get_user_actions(query.user_id)
            .await
            .map_err(|e| format!("Failed to get user actions: {}", e))?;

        Ok(ScoredPostsQuery {
            user_action_sequence: Some(sequence),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.user_action_sequence = hydrated.user_action_sequence;
    }
}
```

### 2.2 Source Trait

**文件**：`candidate-pipeline/source.rs`

```rust
#[async_trait]
pub trait Source<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 获取候选列表
    async fn get_candidates(&self, query: &Q) -> Result<Vec<C>, String>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

**设计要点**：

- **只读操作**：Source 不修改 Query，只返回候选
- **返回 `Vec<C>`**：多个 Source 的结果会被合并

**实现示例**：

```rust
struct ThunderSource {
    thunder_client: Arc<ThunderClient>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for ThunderSource {
    async fn get_candidates(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let response = self.thunder_client
            .get_in_network_posts(query.user_id, &query.following_user_ids)
            .await
            .map_err(|e| format!("Thunder request failed: {}", e))?;

        Ok(response.posts.into_iter().map(|p| PostCandidate::from(p)).collect())
    }
}
```

### 2.3 Hydrator Trait

**文件**：`candidate-pipeline/hydrator.rs`

```rust
#[async_trait]
pub trait Hydrator<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 增强候选
    /// IMPORTANT: 返回的 Vec 必须与输入顺序一致，长度相同
    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>;

    /// 将增强结果合并回原候选
    fn update(&self, candidate: &mut C, hydrated: C);

    /// 批量更新（默认实现）
    fn update_all(&self, candidates: &mut [C], hydrated: Vec<C>) {
        for (c, h) in candidates.iter_mut().zip(hydrated) {
            self.update(c, h);
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

**关键约束**：

> **IMPORTANT**: 返回的 Vec 必须与输入顺序一致，长度相同

为什么？因为 `update_all` 使用 `zip` 来配对更新：

```rust
for (c, h) in candidates.iter_mut().zip(hydrated) {
    self.update(c, h);
}
```

如果长度不一致，会静默丢失数据。

**实现示例**：

```rust
struct CoreDataHydrator {
    core_data_client: Arc<CoreDataClient>,
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for CoreDataHydrator {
    async fn hydrate(&self, _query: &ScoredPostsQuery, candidates: &[PostCandidate]) -> Result<Vec<PostCandidate>, String> {
        let tweet_ids: Vec<u64> = candidates.iter().map(|c| c.tweet_id).collect();

        let core_data = self.core_data_client
            .get_tweets(&tweet_ids)
            .await
            .map_err(|e| format!("Core data request failed: {}", e))?;

        // 保持顺序！
        let hydrated: Vec<PostCandidate> = candidates
            .iter()
            .map(|c| {
                let data = core_data.get(&c.tweet_id);
                PostCandidate {
                    text: data.map(|d| d.text.clone()),
                    ..Default::default()
                }
            })
            .collect();

        Ok(hydrated)
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.text = hydrated.text;
    }
}
```

### 2.4 Filter Trait

**文件**：`candidate-pipeline/filter.rs`

```rust
pub struct FilterResult<C> {
    pub kept: Vec<C>,      // 保留的候选
    pub removed: Vec<C>,   // 被过滤掉的候选
}

#[async_trait]
pub trait Filter<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 过滤候选，返回保留和移除的集合
    async fn filter(&self, query: &Q, candidates: Vec<C>) -> Result<FilterResult<C>, String>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

**设计要点**：

- **返回 `FilterResult`**：同时保留 kept 和 removed，用于日志和指标
- **顺序执行**：一个 Filter 的输出是下一个 Filter 的输入

**实现示例**：

```rust
struct AgeFilter {
    max_age_seconds: u64,
}

#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for AgeFilter {
    async fn filter(&self, _query: &ScoredPostsQuery, candidates: Vec<PostCandidate>) -> Result<FilterResult<PostCandidate>, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| {
                let age = now - c.created_at;
                age <= self.max_age_seconds as i64
            });

        Ok(FilterResult { kept, removed })
    }
}
```

### 2.5 Scorer Trait

**文件**：`candidate-pipeline/scorer.rs`

```rust
#[async_trait]
pub trait Scorer<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 评分
    /// IMPORTANT: 返回的 Vec 必须与输入顺序一致，长度相同
    async fn score(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>;

    /// 更新单个候选的分数字段
    fn update(&self, candidate: &mut C, scored: C);

    fn update_all(&self, candidates: &mut [C], scored: Vec<C>) {
        for (c, s) in candidates.iter_mut().zip(scored) {
            self.update(c, s);
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

**与 Hydrator 的区别**：

- **Hydrator**：补充信息（如文本、作者名）
- **Scorer**：计算分数（如 ML 预测概率）

**顺序执行的原因**：后一个 Scorer 可能依赖前一个 Scorer 的结果。例如：

1. `PhoenixScorer`：计算 ML 预测概率
2. `WeightedScorer`：基于概率计算加权分数
3. `AuthorDiversityScorer`：基于历史调整分数（衰减重复作者）

### 2.6 Selector Trait

**文件**：`candidate-pipeline/selector.rs`

```rust
pub trait Selector<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 选择（默认：排序 + 截断）
    fn select(&self, _query: &Q, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() {
            sorted.truncate(limit);
        }
        sorted
    }

    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 提取分数用于排序
    fn score(&self, candidate: &C) -> f64;

    /// 按分数降序排序
    fn sort(&self, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// 返回选择的数量（None = 不截断）
    fn size(&self) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

**设计要点**：

- **默认实现**：提供了 `sort + truncate` 的默认行为
- **只需实现 `score()`**：子类只需定义如何提取分数

### 2.7 SideEffect Trait

**文件**：`candidate-pipeline/side_effect.rs`

```rust
#[derive(Clone)]
pub struct SideEffectInput<Q, C> {
    pub query: Arc<Q>,
    pub selected_candidates: Vec<C>,
}

#[async_trait]
pub trait SideEffect<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn enable(&self, _query: Arc<Q>) -> bool {
        true
    }

    /// 执行副作用
    async fn run(&self, input: Arc<SideEffectInput<Q, C>>) -> Result<(), String>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
```

**典型用例**：

- 缓存请求信息（下次请求可用）
- 记录日志到外部系统
- 更新统计数据

---

## 3. 实践练习

### 动手任务

**任务**：实现一个简单的 `DebugFilter`，过滤掉 `tweet_id` 为偶数的候选（仅用于学习）。

```rust
struct DebugFilter;

#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for DebugFilter {
    async fn filter(&self, _query: &ScoredPostsQuery, candidates: Vec<PostCandidate>) -> Result<FilterResult<PostCandidate>, String> {
        // TODO: 实现
    }
}
```

<details>
<summary>参考答案</summary>

```rust
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for DebugFilter {
    async fn filter(&self, _query: &ScoredPostsQuery, candidates: Vec<PostCandidate>) -> Result<FilterResult<PostCandidate>, String> {
        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| c.tweet_id % 2 == 1);

        Ok(FilterResult { kept, removed })
    }
}
```

</details>

### 思考题

1. **为什么 Hydrator 和 Scorer 的 `update` 方法接收 `&mut C` 而不是返回 `C`？**

2. **如果需要在管道中添加一个新阶段（如 `Validator`），需要修改哪些地方？**

3. **`enable()` 方法的默认返回值是 `true`。什么情况下会返回 `false`？举例说明。**

### 扩展阅读

- [Rust Trait 设计模式](https://rust-lang.github.io/async-book/06_multiple_futures/01_chapter.html)
- [策略模式](https://refactoring.guru/design-patterns/strategy)

---

## 小结

本节我们详细分析了 7 个 Trait 的设计：

| Trait | 核心方法 | 返回类型 | 执行方式 |
|-------|----------|----------|----------|
| `QueryHydrator` | `hydrate()` | `Q` | 并行 |
| `Source` | `get_candidates()` | `Vec<C>` | 并行 |
| `Hydrator` | `hydrate()` | `Vec<C>` | 并行 |
| `Filter` | `filter()` | `FilterResult<C>` | 顺序 |
| `Scorer` | `score()` | `Vec<C>` | 顺序 |
| `Selector` | `select()` | `Vec<C>` | 单次 |
| `SideEffect` | `run()` | `()` | 异步 |

**核心设计原则**：

1. **单一职责**：每个 Trait 只做一件事
2. **组合优于继承**：通过组合多个 Trait 实现复杂逻辑
3. **并行优先**：独立操作并行执行
4. **优雅降级**：失败不中断，记录日志继续

下一节，我们将进入 `home-mixer` 模块，看这些 Trait 是如何被组装成完整的推荐管道的。
