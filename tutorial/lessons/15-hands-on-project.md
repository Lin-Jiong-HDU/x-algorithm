# 第 15 节：实战项目 - 构建迷你推荐系统

> **学习目标**：综合运用所学知识，使用 candidate-pipeline 框架构建一个简化版推荐系统

---

## 1. 项目概述

### 1.1 目标

构建一个**迷你推荐系统**，能够：
1. 从模拟数据源获取候选
2. 获取用户偏好信息
3. 过滤不合适的候选
4. 计算简单的推荐分数
5. 返回 Top-K 推荐结果

### 1.2 项目结构

```
mini-recommender/
├── Cargo.toml
├── src/
│   ├── main.rs              # 入口
│   ├── lib.rs               # 模块导出
│   ├── pipeline.rs          # 管道组装
│   ├── query.rs             # Query 类型
│   ├── candidate.rs         # Candidate 类型
│   ├── query_hydrator.rs    # 用户偏好获取
│   ├── source.rs            # 候选来源
│   ├── hydrator.rs          # 候选增强
│   ├── filter.rs            # 过滤逻辑
│   ├── scorer.rs            # 评分逻辑
│   └── selector.rs          # 选择器
└── data/
    └── mock_data.json       # 模拟数据
```

---

## 2. 步骤一：项目初始化

### 2.1 创建 Cargo.toml

```toml
[package]
name = "mini-recommender"
version = "0.1.0"
edition = "2021"

[dependencies]
candidate-pipeline = { path = "../candidate-pipeline" }
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
log = "0.4"
env_logger = "0.10"
```

### 2.2 创建基本类型

**src/query.rs**：

```rust
use xai_candidate_pipeline::candidate_pipeline::HasRequestId;

#[derive(Clone, Debug, Default)]
pub struct RecommendQuery {
    pub user_id: u64,
    pub request_id: String,

    // Query Hydrator 填充
    pub user_preferences: Option<UserPreferences>,
}

#[derive(Clone, Debug, Default)]
pub struct UserPreferences {
    pub liked_categories: Vec<String>,
    pub excluded_ids: Vec<u64>,
}

impl HasRequestId for RecommendQuery {
    fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl RecommendQuery {
    pub fn new(user_id: u64) -> Self {
        Self {
            user_id,
            request_id: format!("{}-{}", uuid::Uuid::new_v4(), user_id),
            ..Default::default()
        }
    }
}
```

**src/candidate.rs**：

```rust
#[derive(Clone, Debug, Default)]
pub struct ItemCandidate {
    pub item_id: u64,
    pub title: String,
    pub category: String,
    pub created_at: i64,
    pub score: Option<f64>,
    pub author_id: u64,
}
```

---

## 3. 步骤二：实现各组件

### 3.1 Query Hydrator

**src/query_hydrator.rs**：

```rust
use crate::candidate::{RecommendQuery, UserPreferences};
use async_trait::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

pub struct UserPreferenceHydrator {
    data: std::collections::HashMap<u64, UserPreferences>,
}

impl UserPreferenceHydrator {
    pub fn from_json(path: &str) -> Self {
        let json = std::fs::read_to_string(path).expect("Failed to read data file");
        let data: std::collections::HashMap<u64, UserPreferences> =
            serde_json::from_str(&json).expect("Failed to parse JSON");
        Self { data }
    }
}

#[async_trait]
impl QueryHydrator<RecommendQuery> for UserPreferenceHydrator {
    async fn hydrate(&self, query: &RecommendQuery) -> Result<RecommendQuery, String> {
        let preferences = self.data.get(&query.user_id).cloned().unwrap_or_default();

        Ok(RecommendQuery {
            user_preferences: Some(preferences),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut RecommendQuery, hydrated: RecommendQuery) {
        query.user_preferences = hydrated.user_preferences;
    }

    fn name(&self) -> &'static str {
        "UserPreferenceHydrator"
    }
}
```

### 3.2 Source

**src/source.rs**：

```rust
use crate::candidate::{ItemCandidate, RecommendQuery};
use async_trait::async_trait;
use xai_candidate_pipeline::source::Source;

pub struct MockSource {
    items: Vec<ItemCandidate>,
}

impl MockSource {
    pub fn from_json(path: &str) -> Self {
        let json = std::fs::read_to_string(path).expect("Failed to read data file");
        let items: Vec<ItemCandidate> =
            serde_json::from_str(&json).expect("Failed to parse JSON");
        Self { items }
    }
}

#[async_trait]
impl Source<RecommendQuery, ItemCandidate> for MockSource {
    async fn get_candidates(&self, _query: &RecommendQuery) -> Result<Vec<ItemCandidate>, String> {
        // 返回所有候选（实际应用中会根据 query 过滤）
        Ok(self.items.clone())
    }

    fn name(&self) -> &'static str {
        "MockSource"
    }
}
```

### 3.3 Filter

**src/filter.rs**：

```rust
use crate::candidate::{ItemCandidate, RecommendQuery};
use async_trait::async_trait;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// 过滤已排除的物品
pub struct ExcludedFilter;

#[async_trait]
impl Filter<RecommendQuery, ItemCandidate> for ExcludedFilter {
    async fn filter(
        &self,
        query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Result<FilterResult<ItemCandidate>, String> {
        let excluded: std::collections::HashSet<u64> = query
            .user_preferences
            .as_ref()
            .map(|p| p.excluded_ids.iter().copied().collect())
            .unwrap_or_default();

        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| !excluded.contains(&c.item_id));

        Ok(FilterResult { kept, removed })
    }

    fn name(&self) -> &'static str {
        "ExcludedFilter"
    }
}

// 去重
pub struct DedupFilter;

#[async_trait]
impl Filter<RecommendQuery, ItemCandidate> for DedupFilter {
    async fn filter(
        &self,
        _query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Result<FilterResult<ItemCandidate>, String> {
        let mut seen = std::collections::HashSet::new();
        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for candidate in candidates {
            if seen.insert(candidate.item_id) {
                kept.push(candidate);
            } else {
                removed.push(candidate);
            }
        }

        Ok(FilterResult { kept, removed })
    }

    fn name(&self) -> &'static str {
        "DedupFilter"
    }
}
```

### 3.4 Scorer

**src/scorer.rs**：

```rust
use crate::candidate::{ItemCandidate, RecommendQuery};
use async_trait::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

pub struct CategoryMatchScorer;

#[async_trait]
impl Scorer<RecommendQuery, ItemCandidate> for CategoryMatchScorer {
    async fn score(
        &self,
        query: &RecommendQuery,
        candidates: &[ItemCandidate],
    ) -> Result<Vec<ItemCandidate>, String> {
        let liked_categories: std::collections::HashSet<&str> = query
            .user_preferences
            .as_ref()
            .map(|p| p.liked_categories.iter().map(String::as_str).collect())
            .unwrap_or_default();

        let scored = candidates
            .iter()
            .map(|c| {
                // 如果类别匹配，分数 = 1.0，否则 0.5
                let score = if liked_categories.contains(c.category.as_str()) {
                    1.0
                } else {
                    0.5
                };

                ItemCandidate {
                    score: Some(score),
                    ..Default::default()
                }
            })
            .collect();

        Ok(scored)
    }

    fn update(&self, candidate: &mut ItemCandidate, scored: ItemCandidate) {
        candidate.score = scored.score;
    }

    fn name(&self) -> &'static str {
        "CategoryMatchScorer"
    }
}
```

### 3.5 Selector

**src/selector.rs**：

```rust
use crate::candidate::{ItemCandidate, RecommendQuery};
use xai_candidate_pipeline::selector::Selector;

pub struct TopKSelector {
    pub k: usize,
}

impl Selector<RecommendQuery, ItemCandidate> for TopKSelector {
    fn select(&self, _query: &RecommendQuery, mut candidates: Vec<ItemCandidate>) -> Vec<ItemCandidate> {
        // 按分数降序排序
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 截断到 Top-K
        candidates.truncate(self.k);
        candidates
    }

    fn score(&self, candidate: &ItemCandidate) -> f64 {
        candidate.score.unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "TopKSelector"
    }
}
```

---

## 4. 步骤三：组装管道

**src/pipeline.rs**：

```rust
use crate::candidate::{ItemCandidate, RecommendQuery};
use crate::filter::{DedupFilter, ExcludedFilter};
use crate::query_hydrator::UserPreferenceHydrator;
use crate::scorer::CategoryMatchScorer;
use crate::selector::TopKSelector;
use crate::source::MockSource;
use std::sync::Arc;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::source::Source;

pub struct MiniRecommenderPipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<RecommendQuery>>>,
    sources: Vec<Box<dyn Source<RecommendQuery, ItemCandidate>>>,
    filters: Vec<Box<dyn Filter<RecommendQuery, ItemCandidate>>>,
    scorers: Vec<Box<dyn Scorer<RecommendQuery, ItemCandidate>>>,
    selector: TopKSelector,
    result_size: usize,
}

impl MiniRecommenderPipeline {
    pub fn new(data_path: &str, top_k: usize) -> Self {
        Self {
            query_hydrators: vec![Box::new(UserPreferenceHydrator::from_json(
                &format!("{}/user_preferences.json", data_path),
            ))],
            sources: vec![Box::new(MockSource::from_json(
                &format!("{}/items.json", data_path),
            ))],
            filters: vec![
                Box::new(DedupFilter),
                Box::new(ExcludedFilter),
            ],
            scorers: vec![Box::new(CategoryMatchScorer)],
            selector: TopKSelector { k: top_k },
            result_size: top_k,
        }
    }
}

#[async_trait::async_trait]
impl CandidatePipeline<RecommendQuery, ItemCandidate> for MiniRecommenderPipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<RecommendQuery>>] {
        &self.query_hydrators
    }

    fn sources(&self) -> &[Box<dyn Source<RecommendQuery, ItemCandidate>>] {
        &self.sources
    }

    fn hydrators(&self) -> &[Box<dyn xai_candidate_pipeline::hydrator::Hydrator<RecommendQuery, ItemCandidate>>] {
        &[]
    }

    fn filters(&self) -> &[Box<dyn Filter<RecommendQuery, ItemCandidate>>] {
        &self.filters
    }

    fn scorers(&self) -> &[Box<dyn Scorer<RecommendQuery, ItemCandidate>>] {
        &self.scorers
    }

    fn selector(&self) -> &dyn Selector<RecommendQuery, ItemCandidate> {
        &self.selector
    }

    fn post_selection_hydrators(&self) -> &[Box<dyn xai_candidate_pipeline::hydrator::Hydrator<RecommendQuery, ItemCandidate>>] {
        &[]
    }

    fn post_selection_filters(&self) -> &[Box<dyn Filter<RecommendQuery, ItemCandidate>>] {
        &[]
    }

    fn side_effects(&self) -> Arc<Vec<Box<dyn xai_candidate_pipeline::side_effect::SideEffect<RecommendQuery, ItemCandidate>>>> {
        Arc::new(vec![])
    }

    fn result_size(&self) -> usize {
        self.result_size
    }
}
```

---

## 5. 步骤四：创建模拟数据

**data/user_preferences.json**：

```json
{
  "1": {
    "liked_categories": ["technology", "science"],
    "excluded_ids": [101, 102]
  },
  "2": {
    "liked_categories": ["sports", "entertainment"],
    "excluded_ids": []
  }
}
```

**data/items.json**：

```json
[
  {"item_id": 1, "title": "AI Breakthrough", "category": "technology", "created_at": 1700000000, "author_id": 10},
  {"item_id": 2, "title": "New Planet Discovered", "category": "science", "created_at": 1700000100, "author_id": 11},
  {"item_id": 3, "title": "Championship Finals", "category": "sports", "created_at": 1700000200, "author_id": 12},
  {"item_id": 4, "title": "Movie Review", "category": "entertainment", "created_at": 1700000300, "author_id": 13},
  {"item_id": 5, "title": "Tech Giants Report", "category": "technology", "created_at": 1700000400, "author_id": 10},
  {"item_id": 101, "title": "Excluded Item 1", "category": "technology", "created_at": 1700000500, "author_id": 20},
  {"item_id": 102, "title": "Excluded Item 2", "category": "science", "created_at": 1700000600, "author_id": 21}
]
```

---

## 6. 步骤五：编写主程序

**src/main.rs**：

```rust
mod candidate;
mod filter;
mod pipeline;
mod query;
mod query_hydrator;
mod scorer;
mod selector;
mod source;

use candidate::RecommendQuery;
use env_logger;
use log::info;
use pipeline::MiniRecommenderPipeline;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 创建管道
    let pipeline = MiniRecommenderPipeline::new("./data", 5);

    // 用户 1 的推荐请求
    let query = RecommendQuery::new(1);
    info!("Processing recommendation for user {}", query.user_id);

    // 执行管道
    let result = pipeline.execute(query).await;

    // 输出结果
    println!("\n=== Recommendation Results ===");
    println!("Total retrieved: {}", result.retrieved_candidates.len());
    println!("Total filtered: {}", result.filtered_candidates.len());
    println!("Selected items:");

    for (i, candidate) in result.selected_candidates.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.2}, category: {})",
            i + 1,
            candidate.title,
            candidate.score.unwrap_or(0.0),
            candidate.category
        );
    }

    Ok(())
}
```

---

## 7. 运行与测试

```bash
cd mini-recommender
cargo run
```

**预期输出**：

```
=== Recommendation Results ===
Total retrieved: 7
Total filtered: 2
Selected items:
  1. AI Breakthrough (score: 1.00, category: technology)
  2. New Planet Discovered (score: 1.00, category: science)
  3. Tech Giants Report (score: 1.00, category: technology)
  4. Championship Finals (score: 0.50, category: sports)
  5. Movie Review (score: 0.50, category: entertainment)
```

---

## 8. 扩展练习

### 8.1 添加新功能

1. **添加 `RecencyScorer`**：新物品加分
2. **添加 `AuthorDiversityFilter`**：同一作者最多显示 2 个
3. **添加 HTTP API**：使用 Axum 暴露 REST 接口

### 8.2 单元测试

为每个组件编写测试：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dedup_filter() {
        let filter = DedupFilter;
        let query = RecommendQuery::new(1);
        let candidates = vec![
            ItemCandidate { item_id: 1, ..Default::default() },
            ItemCandidate { item_id: 1, ..Default::default() },
            ItemCandidate { item_id: 2, ..Default::default() },
        ];

        let result = filter.filter(&query, candidates).await.unwrap();
        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 1);
    }
}
```

---

## 小结

恭喜你完成了迷你推荐系统的构建！通过这个项目，你实践了：

1. **Query 类型设计**：定义请求和用户偏好
2. **Candidate 类型设计**：定义候选物品
3. **各组件实现**：QueryHydrator、Source、Filter、Scorer、Selector
4. **管道组装**：使用 `CandidatePipeline` Trait 组合所有组件
5. **端到端运行**：从请求到推荐结果的完整流程

这个迷你系统展示了 X For You 推荐系统的核心设计思想，虽然简化了很多细节，但架构是一致的。

---

## 课程总结

通过 15 节课程，我们完整学习了 X For You 推荐系统：

- **第一阶段**：系统概览与架构理解
- **第二阶段**：Home Mixer 深入
- **第三阶段**：Thunder 实时数据层
- **第四阶段**：Phoenix ML 模型
- **第五阶段**：设计总结与实战

希望这个课程能帮助你理解工业级推荐系统的设计，并将这些知识应用到实际项目中！
