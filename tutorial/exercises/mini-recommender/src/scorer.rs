use crate::candidate::ItemCandidate;
use crate::pipeline_traits::Scorer;
use crate::query::RecommendQuery;
use async_trait::async_trait;

/// 简单评分器：基于分类偏好和热度
pub struct SimpleScorer {
    category_weight: f64,
    trending_weight: f64,
}

impl SimpleScorer {
    pub fn new() -> Self {
        Self {
            category_weight: 1.0,
            trending_weight: 0.5,
        }
    }

    fn compute_score(&self, query: &RecommendQuery, candidate: &ItemCandidate) -> f64 {
        let mut score = 0.5; // 基础分

        // 分类偏好加分
        if let Some(prefs) = &query.user_preferences {
            if prefs.liked_categories.contains(&candidate.category) {
                score += self.category_weight;
            }
        }

        // 热门内容加分
        if candidate.is_trending.unwrap_or(false) {
            score += self.trending_weight;
        }

        score
    }
}

#[async_trait]
impl Scorer for SimpleScorer {
    async fn score(
        &self,
        query: &RecommendQuery,
        candidates: &[ItemCandidate],
    ) -> Result<Vec<ItemCandidate>, String> {
        let scored: Vec<ItemCandidate> = candidates
            .iter()
            .map(|c| {
                let score = self.compute_score(query, c);
                ItemCandidate {
                    score: Some(score),
                    ..c.clone()
                }
            })
            .collect();

        log::info!(
            "request_id={} scorer={} scored {} candidates",
            query.request_id,
            self.name(),
            scored.len()
        );

        Ok(scored)
    }

    fn update(&self, candidate: &mut ItemCandidate, scored: ItemCandidate) {
        candidate.score = scored.score;
    }

    fn name(&self) -> &'static str {
        "SimpleScorer"
    }
}

/// 作者多样性评分器：对同一作者的后续内容降权
pub struct AuthorDiversityScorer {
    decay_factor: f64,
    floor: f64,
}

impl Default for AuthorDiversityScorer {
    fn default() -> Self {
        Self {
            decay_factor: 0.5,
            floor: 0.1,
        }
    }
}

impl AuthorDiversityScorer {
    pub fn new(decay_factor: f64, floor: f64) -> Self {
        Self { decay_factor, floor }
    }

    fn multiplier(&self, position: usize) -> f64 {
        (1.0 - self.floor) * self.decay_factor.powf(position as f64) + self.floor
    }
}

#[async_trait]
impl Scorer for AuthorDiversityScorer {
    async fn score(
        &self,
        _query: &RecommendQuery,
        candidates: &[ItemCandidate],
    ) -> Result<Vec<ItemCandidate>, String> {
        use std::cmp::Ordering;
        use std::collections::HashMap;

        let mut author_counts: HashMap<u64, usize> = HashMap::new();
        let mut scored = vec![ItemCandidate::default(); candidates.len()];

        // 按分数排序后处理（高分优先）
        let mut ordered: Vec<(usize, &ItemCandidate)> = candidates
            .iter()
            .enumerate()
            .collect();
        ordered.sort_by(|(_, a), (_, b)| {
            let a_score = a.score.unwrap_or(f64::NEG_INFINITY);
            let b_score = b.score.unwrap_or(f64::NEG_INFINITY);
            b_score.partial_cmp(&a_score).unwrap_or(Ordering::Equal)
        });

        for (original_idx, candidate) in ordered {
            let entry = author_counts.entry(candidate.author_id).or_insert(0);
            let position = *entry;
            *entry += 1;

            let multiplier = self.multiplier(position);
            let adjusted_score = candidate.score.map(|s| s * multiplier);

            scored[original_idx] = ItemCandidate {
                score: adjusted_score,
                ..Default::default()
            };
        }

        Ok(scored)
    }

    fn update(&self, candidate: &mut ItemCandidate, scored: ItemCandidate) {
        candidate.score = scored.score;
    }

    fn name(&self) -> &'static str {
        "AuthorDiversityScorer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_scorer() {
        let scorer = SimpleScorer::new();
        let mut query = RecommendQuery::new(1);
        query.user_preferences = Some(crate::query::UserPreferences {
            liked_categories: vec!["technology".to_string()],
            excluded_ids: vec![],
            preferred_category: None,
        });

        let candidates = vec![
            ItemCandidate::new(1, "Tech Post", "technology", 1),
            ItemCandidate::new(2, "Sports Post", "sports", 2),
        ];

        let result = scorer.score(&query, &candidates).await.unwrap();

        // 技术类应该有更高的分数
        assert!(result[0].score > result[1].score);
    }
}
