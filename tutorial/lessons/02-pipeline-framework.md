# 第 2 节：推荐管道的抽象设计

> **学习目标**：理解 `candidate-pipeline` 框架的核心设计，掌握 Trait 系统如何实现可复用的推荐管道

---

## 1. 概念讲解

### 1.1 为什么需要管道框架？

推荐系统本质上是**一系列数据处理步骤的组合**：

```
获取用户信息 → 获取候选 → 补充信息 → 过滤 → 评分 → 排序选择
```

如果每个推荐场景都从头实现这套流程，会导致：

1. **代码重复**：相似的逻辑在多处实现
2. **难以维护**：修改一处需要同步多处
3. **缺乏一致性**：不同场景的日志、指标、错误处理不一致

**解决方案**：抽象出一个通用的管道框架，定义标准接口（Trait），让各场景只需实现具体逻辑。

### 1.2 管道设计模式

X 采用了经典的**管道模式（Pipeline Pattern）**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CandidatePipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐                                                  │
│   │ Query        │                                                  │
│   │ (用户请求)    │                                                  │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐     ┌──────────────┐                            │
│   │ Query        │ ──▶ │ Source 1     │ ──┐                        │
│   │ Hydrators    │     │ Source 2     │ ──┼──▶ 合并候选            │
│   │ (获取用户信息)│     │ ...          │ ──┘                        │
│   └──────────────┘     └──────────────┘                            │
│          │                    │                                    │
│          └────────────────────┘                                    │
│                    │                                                │
│                    ▼                                                │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │ Hydrator 1   │     │ Filter 1     │     │ Scorer 1     │       │
│   │ Hydrator 2   │ ──▶ │ Filter 2     │ ──▶ │ Scorer 2     │ ──▶   │
│   │ ...          │     │ ...          │     │ ...          │       │
│   └──────────────┘     └──────────────┘     └──────────────┘       │
│          │                    │                    │                │
│      (并行执行)           (顺序执行)           (顺序执行)            │
│                                                    │                │
│                                                    ▼                │
│                                            ┌──────────────┐        │
│                                            │ Selector     │        │
│                                            │ (排序+截断)   │        │
│                                            └──────────────┘        │
│                                                    │                │
│                                                    ▼                │
│                                            ┌──────────────┐        │
│                                            │ SideEffect   │        │
│                                            │ (异步副作用)  │        │
│                                            └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 核心设计原则

1. **关注点分离**：每个 Trait 只负责一件事
2. **可组合性**：通过组合不同实现构建复杂管道
3. **并行优先**：独立操作并行执行，提升性能
4. **优雅降级**：单个组件失败不影响整体流程

---

## 2. 代码分析

### 2.1 CandidatePipeline Trait

打开 `candidate-pipeline/candidate_pipeline.rs`，核心定义在第 37-51 行：

```rust
#[async_trait]
pub trait CandidatePipeline<Q, C>: Send + Sync
where
    Q: HasRequestId + Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>];
    fn sources(&self) -> &[Box<dyn Source<Q, C>>];
    fn hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn scorers(&self) -> &[Box<dyn Scorer<Q, C>>];
    fn selector(&self) -> &dyn Selector<Q, C>;
    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn post_selection_filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<Q, C>>>>;
    fn result_size(&self) -> usize;
    // ...
}
```

**关键点**：

- **泛型参数**：`Q` 是查询类型（用户请求），`C` 是候选类型（帖子）
- **返回切片**：`&[Box<dyn Trait>]` 允许返回多个实现
- **Send + Sync**：支持跨线程共享（异步运行时需要）

### 2.2 execute 方法：管道执行流程

第 53-92 行是核心执行逻辑：

```rust
async fn execute(&self, query: Q) -> PipelineResult<Q, C> {
    // 1. 查询增强（并行）
    let hydrated_query = self.hydrate_query(query).await;

    // 2. 获取候选（并行）
    let candidates = self.fetch_candidates(&hydrated_query).await;

    // 3. 候选增强（并行）
    let hydrated_candidates = self.hydrate(&hydrated_query, candidates).await;

    // 4. 过滤（顺序）
    let (kept_candidates, mut filtered_candidates) = self
        .filter(&hydrated_query, hydrated_candidates.clone())
        .await;

    // 5. 评分（顺序）
    let scored_candidates = self.score(&hydrated_query, kept_candidates).await;

    // 6. 选择（排序+截断）
    let selected_candidates = self.select(&hydrated_query, scored_candidates);

    // 7. 选择后增强（并行）
    let post_selection_hydrated_candidates = self
        .hydrate_post_selection(&hydrated_query, selected_candidates)
        .await;

    // 8. 选择后过滤（顺序）
    let (mut final_candidates, post_selection_filtered_candidates) = self
        .filter_post_selection(&hydrated_query, post_selection_hydrated_candidates)
        .await;

    // 9. 最终截断
    final_candidates.truncate(self.result_size());

    // 10. 异步副作用
    let arc_hydrated_query = Arc::new(hydrated_query);
    let input = Arc::new(SideEffectInput {
        query: arc_hydrated_query.clone(),
        selected_candidates: final_candidates.clone(),
    });
    self.run_side_effects(input);

    PipelineResult {
        retrieved_candidates: hydrated_candidates,
        filtered_candidates,
        selected_candidates: final_candidates,
        query: arc_hydrated_query,
    }
}
```

**执行顺序图**：

```
Query ──▶ hydrate_query ──▶ fetch_candidates ──▶ hydrate
                                                        │
                    ┌───────────────────────────────────┘
                    ▼
              filter (pre-scoring) ──▶ score ──▶ select
                                                        │
                    ┌───────────────────────────────────┘
                    ▼
         hydrate_post_selection ──▶ filter_post_selection
                                                        │
                    ┌───────────────────────────────────┘
                    ▼
                truncate ──▶ side_effects (异步)
```

### 2.3 并行执行模式

查看 `hydrate_query` 方法（第 95-123 行）：

```rust
async fn hydrate_query(&self, query: Q) -> Q {
    let request_id = query.request_id().to_string();

    // 1. 过滤启用的 hydrator
    let hydrators: Vec<_> = self
        .query_hydrators()
        .iter()
        .filter(|h| h.enable(&query))
        .collect();

    // 2. 并行执行所有 hydrator
    let hydrate_futures = hydrators.iter().map(|h| h.hydrate(&query));
    let results = join_all(hydrate_futures).await;

    // 3. 合并结果到 query
    let mut hydrated_query = query;
    for (hydrator, result) in hydrators.iter().zip(results) {
        match result {
            Ok(hydrated) => {
                hydrator.update(&mut hydrated_query, hydrated);
            }
            Err(err) => {
                error!(
                    "request_id={} stage={:?} component={} failed: {}",
                    request_id,
                    PipelineStage::QueryHydrator,
                    hydrator.name(),
                    err
                );
                // 注意：失败不会中断流程，只是记录日志
            }
        }
    }
    hydrated_query
}
```

**设计亮点**：

1. **`join_all` 并行**：所有 hydrator 同时执行，等待全部完成
2. **错误隔离**：单个 hydrator 失败只记录日志，不影响其他
3. **结果合并**：通过 `update` 方法将各 hydrator 的结果合并到 query

### 2.4 顺序执行模式：Filter

查看 `run_filters` 方法（第 237-273 行）：

```rust
async fn run_filters(
    &self,
    query: &Q,
    mut candidates: Vec<C>,
    filters: &[Box<dyn Filter<Q, C>>],
    stage: PipelineStage,
) -> (Vec<C>, Vec<C>) {
    let request_id = query.request_id().to_string();
    let mut all_removed = Vec::new();

    // 顺序执行每个 filter
    for filter in filters.iter().filter(|f| f.enable(query)) {
        let backup = candidates.clone();  // 备份，用于失败恢复
        match filter.filter(query, candidates).await {
            Ok(result) => {
                candidates = result.kept;
                all_removed.extend(result.removed);
            }
            Err(err) => {
                error!(
                    "request_id={} stage={:?} component={} failed: {}",
                    request_id, stage, filter.name(), err
                );
                candidates = backup;  // 失败时恢复备份
            }
        }
    }

    info!(
        "request_id={} stage={:?} kept {}, removed {}",
        request_id, stage, candidates.len(), all_removed.len()
    );
    (candidates, all_removed)
}
```

**为什么 Filter 是顺序执行？**

因为 Filter 有**依赖关系**：比如先去重，再过滤屏蔽用户。如果并行执行，一个 filter 的输出无法作为另一个的输入。

### 2.5 SideEffect：异步副作用

查看 `run_side_effects` 方法（第 319-328 行）：

```rust
fn run_side_effects(&self, input: Arc<SideEffectInput<Q, C>>) {
    let side_effects = self.side_effects();
    tokio::spawn(async move {
        let futures = side_effects
            .iter()
            .filter(|se| se.enable(input.query.clone()))
            .map(|se| se.run(input.clone()));
        let _ = join_all(futures).await;
    });
}
```

**设计亮点**：

- **`tokio::spawn`**：在新任务中执行，不阻塞主流程
- **用户可立即获得响应**：副作用（如缓存写入）在后台完成

---

## 3. 实践练习

### 动手任务

1. **画出管道执行的数据流图**
   - 标注每个阶段的输入输出类型
   - 用不同颜色标注并行和顺序执行

2. **分析错误处理策略**
   - 找出代码中所有 `match result` 的地方
   - 总结错误处理的一致性模式

### 代码阅读任务

阅读以下代码，回答问题：

```rust
// candidate_pipeline.rs 第 160-174 行
async fn hydrate(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
    self.run_hydrators(query, candidates, self.hydrators(), PipelineStage::Hydrator)
        .await
}

async fn hydrate_post_selection(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
    self.run_hydrators(
        query,
        candidates,
        self.post_selection_hydrators(),
        PipelineStage::PostSelectionHydrator,
    )
    .await
}
```

**问题**：为什么 `hydrate` 和 `hydrate_post_selection` 使用相同的方法 `run_hydrators`，但传入不同的 hydrator 列表？

### 思考题

1. **为什么 Hydrator 返回的是 `Vec<C>` 而不是直接修改 `&mut [C]`？**

   <details>
   <summary>提示</summary>

   考虑异步场景：如果多个 hydrator 并行执行，如何避免数据竞争？

   </details>

2. **如果某个 Source 失败了，整个请求会失败吗？查看 `fetch_candidates` 方法验证你的答案。**

3. **SideEffect 的执行结果会影响返回给用户的结果吗？这样设计有什么好处？**

---

## 小结

本节我们深入理解了 `candidate-pipeline` 框架的核心设计：

1. **Trait 系统抽象**：通过 7 个 Trait 定义管道的各个阶段
2. **执行流程**：`hydrate_query → fetch_candidates → hydrate → filter → score → select`
3. **并行策略**：Hydrator/Source 并行，Filter/Scorer 顺序
4. **错误处理**：单个组件失败不中断流程，记录日志继续
5. **异步副作用**：SideEffect 在后台执行，不阻塞响应

下一节，我们将逐一分析 7 个 Trait 的具体定义和实现模式。
