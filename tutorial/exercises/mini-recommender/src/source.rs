use crate::candidate::ItemCandidate;
use crate::pipeline_traits::Source;
use crate::query::RecommendQuery;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// 模拟的候选数据存储
pub struct MockCandidateStore {
    // category -> candidates
    store: RwLock<HashMap<String, Vec<ItemCandidate>>>,
}

impl MockCandidateStore {
    pub fn new() -> Self {
        let mut store = HashMap::new();

        // 添加模拟数据
        store.insert(
            "technology".to_string(),
            vec![
                ItemCandidate::new(101, "AI Breakthrough", "technology", 1),
                ItemCandidate::new(102, "New Programming Language", "technology", 2),
                ItemCandidate::new(103, "Tech Giants Report", "technology", 3),
                ItemCandidate::new(104, "Cloud Computing Update", "technology", 4),
            ],
        );

        store.insert(
            "science".to_string(),
            vec![
                ItemCandidate::new(201, "New Planet Discovered", "science", 1),
                ItemCandidate::new(202, "Climate Research Update", "science", 2),
                ItemCandidate::new(203, "Medical Breakthrough", "science", 3),
            ],
        );

        store.insert(
            "sports".to_string(),
            vec![
                ItemCandidate::new(301, "Championship Finals", "sports", 1),
                ItemCandidate::new(302, "Transfer News", "sports", 2),
            ],
        );

        store.insert(
            "entertainment".to_string(),
            vec![
                ItemCandidate::new(401, "Movie Review", "entertainment", 1),
                ItemCandidate::new(402, "Music Awards", "entertainment", 2),
            ],
        );

        Self {
            store: RwLock::new(store),
        }
    }

    pub fn get_all_candidates(&self) -> Vec<ItemCandidate> {
        let store = self.store.read().unwrap();
        store.values().flat_map(|v| v.clone()).collect()
    }

    pub fn get_candidates_by_categories(&self, categories: &[String]) -> Vec<ItemCandidate> {
        let store = self.store.read().unwrap();
        categories
            .iter()
            .filter_map(|cat| store.get(cat).cloned())
            .flat_map(|v| v)
            .collect()
    }
}

/// 偏好型候选来源
pub struct PreferenceBasedSource {
    store: Arc<MockCandidateStore>,
}

impl PreferenceBasedSource {
    pub fn new(store: Arc<MockCandidateStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Source for PreferenceBasedSource {
    fn enable(&self, query: &RecommendQuery) -> bool {
        query.user_preferences.is_some()
    }

    async fn get_candidates(&self, query: &RecommendQuery) -> Result<Vec<ItemCandidate>, String> {
        let prefs = query
            .user_preferences
            .as_ref()
            .ok_or("No user preferences available")?;

        let candidates = self.store.get_candidates_by_categories(&prefs.liked_categories);

        // 过滤掉已排除的
        let excluded: std::collections::HashSet<u64> = prefs.excluded_ids.iter().copied().collect();
        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|c| !excluded.contains(&c.item_id))
            .collect();

        log::info!(
            "request_id={} source={} fetched {} candidates",
            query.request_id,
            self.name(),
            candidates.len()
        );

        Ok(candidates)
    }

    fn name(&self) -> &'static str {
        "PreferenceBasedSource"
    }
}

/// 热门内容来源
pub struct TrendingSource {
    store: Arc<MockCandidateStore>,
}

impl TrendingSource {
    pub fn new(store: Arc<MockCandidateStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Source for TrendingSource {
    async fn get_candidates(&self, query: &RecommendQuery) -> Result<Vec<ItemCandidate>, String> {
        let all_candidates = self.store.get_all_candidates();
        let candidates: Vec<_> = all_candidates
            .into_iter()
            .take(5)
            .map(|mut c| {
                c.is_trending = Some(true);
                c
            })
            .collect();

        log::info!(
            "request_id={} source={} fetched {} candidates",
            query.request_id,
            self.name(),
            candidates.len()
        );

        Ok(candidates)
    }

    fn name(&self) -> &'static str {
        "TrendingSource"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_preference_based_source() {
        let store = Arc::new(MockCandidateStore::new());
        let source = PreferenceBasedSource::new(store);

        let mut query = RecommendQuery::new(1);
        query.user_preferences = Some(crate::query::UserPreferences {
            liked_categories: vec!["technology".to_string()],
            excluded_ids: vec![],
            preferred_category: None,
        });

        let candidates = source.get_candidates(&query).await.unwrap();
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|c| c.category == "technology"));
    }
}
