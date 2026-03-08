use crate::candidate::ItemCandidate;
use crate::pipeline_traits::{Filter, FilterResult};
use crate::query::RecommendQuery;
use async_trait::async_trait;
use std::collections::HashSet;

/// 去重过滤器
pub struct DedupFilter;

#[async_trait]
impl Filter for DedupFilter {
    async fn filter(
        &self,
        _query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Result<FilterResult<ItemCandidate>, String> {
        let mut seen = HashSet::new();
        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for candidate in candidates {
            if seen.insert(candidate.item_id) {
                kept.push(candidate);
            } else {
                removed.push(candidate);
            }
        }

        log::info!(
            "DedupFilter: kept {}, removed {}",
            kept.len(),
            removed.len()
        );

        Ok(FilterResult { kept, removed })
    }

    fn name(&self) -> &'static str {
        "DedupFilter"
    }
}

/// 排除列表过滤器
pub struct ExclusionFilter;

#[async_trait]
impl Filter for ExclusionFilter {
    async fn filter(
        &self,
        query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Result<FilterResult<ItemCandidate>, String> {
        let excluded: HashSet<_> = query
            .user_preferences
            .as_ref()
            .map(|p| p.excluded_ids.iter().copied().collect())
            .unwrap_or_default();

        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for candidate in candidates {
            if excluded.contains(&candidate.item_id) {
                removed.push(candidate);
            } else {
                kept.push(candidate);
            }
        }

        log::info!(
            "ExclusionFilter: kept {}, removed {}",
            kept.len(),
            removed.len()
        );

        Ok(FilterResult { kept, removed })
    }

    fn name(&self) -> &'static str {
        "ExclusionFilter"
    }
}

/// 类别匹配过滤器
pub struct CategoryFilter;

#[async_trait]
impl Filter for CategoryFilter {
    async fn filter(
        &self,
        query: &RecommendQuery,
        candidates: Vec<ItemCandidate>,
    ) -> Result<FilterResult<ItemCandidate>, String> {
        let liked_categories: HashSet<_> = query
            .user_preferences
            .as_ref()
            .map(|p| p.liked_categories.iter().cloned().collect())
            .unwrap_or_default();

        // 如果没有偏好设置，保留所有
        if liked_categories.is_empty() {
            return Ok(FilterResult {
                kept: candidates,
                removed: vec![],
            });
        }

        let mut kept = Vec::new();
        let mut removed = Vec::new();

        for candidate in candidates {
            if liked_categories.contains(&candidate.category) {
                kept.push(candidate);
            } else {
                removed.push(candidate);
            }
        }

        log::info!(
            "CategoryFilter: kept {}, removed {}",
            kept.len(),
            removed.len()
        );

        Ok(FilterResult { kept, removed })
    }

    fn name(&self) -> &'static str {
        "CategoryFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dedup_filter() {
        let filter = DedupFilter;
        let query = RecommendQuery::new(1);

        let candidates = vec![
            ItemCandidate::new(1, "A", "tech", 1),
            ItemCandidate::new(1, "A", "tech", 1), // 重复
            ItemCandidate::new(2, "B", "tech", 1),
        ];

        let result = filter.filter(&query, candidates).await.unwrap();
        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 1);
    }
}
