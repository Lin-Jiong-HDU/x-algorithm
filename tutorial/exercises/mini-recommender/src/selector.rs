use crate::candidate::ItemCandidate;
use crate::pipeline_traits::Selector;
use crate::query::RecommendQuery;

/// Top-K 分数选择器
pub struct TopKScoreSelector {
    k: usize,
}

impl TopKScoreSelector {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Selector for TopKScoreSelector {
    fn score(&self, candidate: &ItemCandidate) -> f64 {
        candidate.score.unwrap_or(0.0)
    }

    fn size(&self) -> Option<usize> {
        Some(self.k)
    }

    fn name(&self) -> &'static str {
        "TopKScoreSelector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_selector() {
        let selector = TopKScoreSelector::new(3);

        let candidates = vec![
            ItemCandidate {
                item_id: 1,
                score: Some(0.9),
                ..Default::default()
            },
            ItemCandidate {
                item_id: 2,
                score: Some(0.8),
                ..Default::default()
            },
            ItemCandidate {
                item_id: 3,
                score: Some(0.7),
                ..Default::default()
            },
            ItemCandidate {
                item_id: 4,
                score: Some(0.6),
                ..Default::default()
            },
            ItemCandidate {
                item_id: 5,
                score: Some(0.5),
                ..Default::default()
            },
        ];

        let query = RecommendQuery::new(1);
        let result = selector.select(&query, candidates);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].item_id, 1);
        assert_eq!(result[1].item_id, 2);
        assert_eq!(result[2].item_id, 3);
    }
}
