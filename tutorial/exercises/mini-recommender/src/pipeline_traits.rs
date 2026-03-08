// 为了简化，我们在这里重新定义 candidate-pipeline 的核心 trait
// 实际项目中应该直接依赖 candidate-pipeline crate

use async_trait::async_trait;
use std::sync::Arc;

use crate::candidate::ItemCandidate;
use crate::query::RecommendQuery;

/// Query Hydrator Trait
#[async_trait]
pub trait QueryHydrator: Send + Sync {
    fn enable(&self, _query: &RecommendQuery) -> bool {
        true
    }

    async fn hydrate(&self, query: &RecommendQuery) -> Result<RecommendQuery, String>;

    fn update(&self, query: &mut RecommendQuery, hydrated: RecommendQuery);

    fn name(&self) -> &'static str;
}

/// Source Trait
#[async_trait]
pub trait Source: Send + Sync {
    fn enable(&self, _query: &RecommendQuery) -> bool {
        true
    }

    async fn get_candidates(&self, query: &RecommendQuery) -> Result<Vec<ItemCandidate>, String>;

    fn name(&self) -> &'static str;
}

/// Hydrator Trait
#[async_trait]
pub trait Hydrator: Send + Sync {
    fn enable(&self, _query: &RecommendQuery) -> bool {
        true
    }

    async fn hydrate(
        &self,
        query: &RecommendQuery,
        candidates: &[ItemCandidate],
    ) -> Result<Vec<ItemCandidate>, String>;

    fn update(&self, candidate: &mut ItemCandidate, hydrated: ItemCandidate);

    fn update_all(&self, candidates: &mut [ItemCandidate], hydrated: Vec<ItemCandidate>) {
        for (c, h) in candidates.iter_mut().zip(hydrated) {
            self.update(c, h);
        }
    }

    fn name(&self) -> &'static str;
}

/// Filter Result
pub struct FilterResult<C> {
    pub kept: Vec<C>,
    pub removed: Vec<C>,
}

/// Filter Trait
#[async_trait]
pub trait Filter: Send + Sync {
    fn enable(&self, _query: &RecommendQuery) -> bool {
        true
    }

    async fn filter(
        &self,
        query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Result<FilterResult<ItemCandidate>, String>;

    fn name(&self) -> &'static str;
}

/// Scorer Trait
#[async_trait]
pub trait Scorer: Send + Sync {
    fn enable(&self, _query: &RecommendQuery) -> bool {
        true
    }

    async fn score(
        &self,
        query: &RecommendQuery,
        candidates: &[ItemCandidate],
    ) -> Result<Vec<ItemCandidate>, String>;

    fn update(&self, candidate: &mut ItemCandidate, scored: ItemCandidate);

    fn update_all(&self, candidates: &mut [ItemCandidate], scored: Vec<ItemCandidate>) {
        for (c, s) in candidates.iter_mut().zip(scored) {
            self.update(c, s);
        }
    }

    fn name(&self) -> &'static str;
}

/// Selector Trait
pub trait Selector: Send + Sync {
    fn enable(&self, _query: &RecommendQuery) -> bool {
        true
    }

    fn select(&self, query: &RecommendQuery, candidates: Vec<ItemCandidate>) -> Vec<ItemCandidate> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() {
            sorted.truncate(limit);
        }
        sorted
    }

    fn score(&self, candidate: &ItemCandidate) -> f64;

    fn sort(&self, candidates: Vec<ItemCandidate>) -> Vec<ItemCandidate> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    fn size(&self) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str;
}

/// Side Effect Trait
#[async_trait]
pub trait SideEffect: Send + Sync {
    fn enable(&self, _query: Arc<RecommendQuery>) -> bool {
        true
    }

    async fn run(
        &self,
        query: Arc<RecommendQuery>,
        selected_candidates: Vec<ItemCandidate>,
    ) -> Result<(), String>;

    fn name(&self) -> &'static str;
}
