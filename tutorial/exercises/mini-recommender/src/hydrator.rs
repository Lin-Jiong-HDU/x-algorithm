use crate::candidate::ItemCandidate;
use crate::pipeline_traits::Hydrator;
use crate::query::RecommendQuery;
use async_trait::async_trait;

/// 热门标记 Hydrator
pub struct TrendingHydrator;

#[async_trait]
impl Hydrator for TrendingHydrator {
    async fn hydrate(
        &self,
        _query: &RecommendQuery,
        candidates: &[ItemCandidate],
    ) -> Result<Vec<ItemCandidate>, String> {
        // 模拟：标记 ID 为奇数的为热门
        let hydrated: Vec<ItemCandidate> = candidates
            .iter()
            .map(|c| {
                let is_trending = Some(c.item_id % 2 == 1);
                ItemCandidate {
                    is_trending,
                    ..Default::default()
                }
            })
            .collect();

        Ok(hydrated)
    }

    fn update(&self, candidate: &mut ItemCandidate, hydrated: ItemCandidate) {
        candidate.is_trending = hydrated.is_trending;
    }

    fn name(&self) -> &'static str {
        "TrendingHydrator"
    }
}
