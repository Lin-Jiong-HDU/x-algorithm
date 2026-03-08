use crate::candidate::ItemCandidate;
use crate::query::RecommendQuery;
use futures::future::join_all;
use log::{error, info, warn};
use std::sync::Arc;
use tokio::spawn;

use crate::pipeline_traits::{
    Filter, FilterResult, Hydrator, QueryHydrator, Scorer, Selector, SideEffect, Source,
};

/// 推荐结果
pub struct RecommendResult {
    pub retrieved_candidates: Vec<ItemCandidate>,
    pub filtered_candidates: Vec<ItemCandidate>,
    pub selected_candidates: Vec<ItemCandidate>,
    pub query: Arc<RecommendQuery>,
}

/// Mini 推荐管道
pub struct MiniRecommenderPipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator>>,
    sources: Vec<Box<dyn Source>>,
    hydrators: Vec<Box<dyn Hydrator>>,
    filters: Vec<Box<dyn Filter>>,
    scorers: Vec<Box<dyn Scorer>>,
    selector: Box<dyn Selector>,
    side_effects: Arc<Vec<Box<dyn SideEffect>>>,
    result_size: usize,
}

impl MiniRecommenderPipeline {
    pub fn new(
        query_hydrators: Vec<Box<dyn QueryHydrator>>,
        sources: Vec<Box<dyn Source>>,
        hydrators: Vec<Box<dyn Hydrator>>,
        filters: Vec<Box<dyn Filter>>,
        scorers: Vec<Box<dyn Scorer>>,
        selector: Box<dyn Selector>,
        side_effects: Vec<Box<dyn SideEffect>>,
        result_size: usize,
    ) -> Self {
        Self {
            query_hydrators,
            sources,
            hydrators,
            filters,
            scorers,
            selector,
            side_effects: Arc::new(side_effects),
            result_size,
        }
    }

    /// 执行推荐管道
    pub async fn execute(&self, mut query: RecommendQuery) -> RecommendResult {
        let request_id = query.request_id.clone();
        info!("request_id={} Starting recommendation pipeline", request_id);

        // 1. Query Hydration（并行）
        query = self.hydrate_query(&query).await;

        // 2. Fetch Candidates（并行）
        let candidates = self.fetch_candidates(&query).await;
        let retrieved_count = candidates.len();
        info!("request_id={} Retrieved {} candidates", request_id, retrieved_count);

        // 3. Hydrate Candidates（并行）
        let candidates = self.hydrate_candidates(&query, candidates).await;

        // 4. Filter（顺序）
        let (candidates, filtered) = self.apply_filters(&query, candidates).await;
        info!(
            "request_id={} After filtering: {} kept, {} removed",
            request_id,
            candidates.len(),
            filtered.len()
        );

        // 5. Score（顺序）
        let candidates = self.apply_scorers(&query, candidates).await;

        // 6. Select
        let selected = self.select(&query, candidates);

        // 7. Run side effects（异步，不阻塞）
        self.run_side_effects(Arc::new(query.clone()), selected.clone());

        // 8. Truncate to result size
        let selected: Vec<_> = selected.into_iter().take(self.result_size).collect();

        info!("request_id={} Selected {} items", request_id, selected.len());

        RecommendResult {
            retrieved_candidates: vec![], // 简化版不跟踪
            filtered_candidates: filtered,
            selected_candidates: selected,
            query: Arc::new(query),
        }
    }

    /// Query Hydration（并行执行）
    async fn hydrate_query(&self, query: &RecommendQuery) -> RecommendQuery {
        let request_id = query.request_id.clone();
        let hydrators: Vec<_> = self
            .query_hydrators
            .iter()
            .filter(|h| h.enable(query))
            .collect();

        // 并行执行
        let futures = hydrators.iter().map(|h| h.hydrate(query));
        let results = join_all(futures).await;

        // 合并结果
        let mut hydrated_query = query.clone();
        for (hydrator, result) in hydrators.iter().zip(results) {
            match result {
                Ok(hydrated) => {
                    hydrator.update(&mut hydrated_query, hydrated);
                }
                Err(err) => {
                    error!(
                        "request_id={} hydrator {} failed: {}",
                        request_id,
                        hydrator.name(),
                        err
                    );
                }
            }
        }
        hydrated_query
    }

    /// Fetch Candidates（并行执行）
    async fn fetch_candidates(&self, query: &RecommendQuery) -> Vec<ItemCandidate> {
        let request_id = query.request_id.clone();
        let sources: Vec<_> = self.sources.iter().filter(|s| s.enable(query)).collect();

        // 并行执行
        let futures = sources.iter().map(|s| s.get_candidates(query));
        let results = join_all(futures).await;

        // 合并结果
        let mut collected = Vec::new();
        for (source, result) in sources.iter().zip(results) {
            match result {
                Ok(mut candidates) => {
                    info!(
                        "request_id={} source {} fetched {} candidates",
                        request_id,
                        source.name(),
                        candidates.len()
                    );
                    collected.append(&mut candidates);
                }
                Err(err) => {
                    error!(
                        "request_id={} source {} failed: {}",
                        request_id,
                        source.name(),
                        err
                    );
                }
            }
        }
        collected
    }

    /// Hydrate Candidates（并行执行）
    async fn hydrate_candidates(
        &self,
        query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Vec<ItemCandidate> {
        if candidates.is_empty() || self.hydrators.is_empty() {
            return candidates;
        }

        let request_id = query.request_id.clone();
        let hydrators: Vec<_> = self.hydrators.iter().filter(|h| h.enable(query)).collect();
        let expected_len = candidates.len();

        // 并行执行
        let futures = hydrators.iter().map(|h| h.hydrate(query, &candidates));
        let results = join_all(futures).await;

        // 合并结果
        let mut hydrated = candidates;
        for (hydrator, result) in hydrators.iter().zip(results) {
            match result {
                Ok(h) => {
                    if h.len() == expected_len {
                        hydrator.update_all(&mut hydrated, h);
                    } else {
                        warn!(
                            "request_id={} hydrator {} length mismatch",
                            request_id,
                            hydrator.name()
                        );
                    }
                }
                Err(err) => {
                    error!(
                        "request_id={} hydrator {} failed: {}",
                        request_id,
                        hydrator.name(),
                        err
                    );
                }
            }
        }
        hydrated
    }

    /// Apply Filters（顺序执行）
    async fn apply_filters(
        &self,
        query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> (Vec<ItemCandidate>, Vec<ItemCandidate>) {
        let request_id = query.request_id.clone();
        let mut current = candidates;
        let mut all_removed = Vec::new();

        for filter in self.filters.iter().filter(|f| f.enable(query)) {
            let backup = current.clone();
            match filter.filter(query, current).await {
                Ok(FilterResult { kept, removed }) => {
                    info!(
                        "request_id={} filter {} kept {} removed {}",
                        request_id,
                        filter.name(),
                        kept.len(),
                        removed.len()
                    );
                    current = kept;
                    all_removed.extend(removed);
                }
                Err(err) => {
                    error!(
                        "request_id={} filter {} failed: {}",
                        request_id,
                        filter.name(),
                        err
                    );
                    current = backup;
                }
            }
        }
        (current, all_removed)
    }

    /// Apply Scorers（顺序执行）
    async fn apply_scorers(
        &self,
        query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Vec<ItemCandidate> {
        if candidates.is_empty() {
            return candidates;
        }

        let request_id = query.request_id.clone();
        let mut scored = candidates;

        for scorer in self.scorers.iter().filter(|s| s.enable(query)) {
            match scorer.score(query, &scored).await {
                Ok(s) => scorer.update_all(&mut scored, s),
                Err(err) => {
                    error!(
                        "request_id={} scorer {} failed: {}",
                        request_id,
                        scorer.name(),
                        err
                    );
                }
            }
        }
        scored
    }

    /// Select
    fn select(&self, query: &RecommendQuery, candidates: Vec<ItemCandidate>) -> Vec<ItemCandidate> {
        self.selector.select(query, candidates)
    }

    /// Run Side Effects（异步，不阻塞）
    fn run_side_effects(&self, query: Arc<RecommendQuery>, selected: Vec<ItemCandidate>) {
        let side_effects = Arc::clone(&self.side_effects);
        spawn(async move {
            let futures = side_effects
                .iter()
                .filter(|se| se.enable(Arc::clone(&query)))
                .map(|se| se.run(Arc::clone(&query), selected.clone()));
            let _ = join_all(futures).await;
        });
    }
}
