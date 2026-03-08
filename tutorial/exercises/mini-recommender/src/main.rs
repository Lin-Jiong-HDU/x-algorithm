use log::info;
use std::sync::Arc;

mod candidate;
mod filter;
mod hydrator;
mod pipeline;
mod pipeline_traits;
mod query;
mod query_hydrator;
mod scorer;
mod selector;
mod source;

pub use candidate::ItemCandidate;
pub use pipeline::MiniRecommenderPipeline;
pub use query::RecommendQuery;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // 1. 创建管道
    let pipeline = create_pipeline();

    info!("Mini Recommender started");

    // 2. 创建查询
    let query = RecommendQuery::new(1);

    // 3. 执行推荐
    info!("Running recommendation for user_id={}", query.user_id);
    let result = pipeline.execute(query).await;

    // 4. 输出结果
    info!("\n=== Recommendation Results ===");
    info!("Retrieved: {}", result.retrieved_candidates.len());
    info!("Filtered: {}", result.filtered_candidates.len());
    info!("Selected: {}", result.selected_candidates.len());

    for (i, item) in result.selected_candidates.iter().enumerate() {
        let score = item.score.unwrap_or(0.0);
        info!(
            "  {}. {} (category: {}, score: {:.3})",
            i + 1,
            item.title,
            item.category,
            score
        );
    }
}

fn create_pipeline() -> MiniRecommenderPipeline {
    use filter::{CategoryFilter, DedupFilter};
    use hydrator::{ExcludedItemsHydrator, UserPreferenceHydrator};
    use scorer::{AuthorDiversityScorer, SimpleScorer};
    use selector::TopKScoreSelector;
    use source::{MockCandidateStore, PreferenceBasedSource, TrendingSource};

    // 创建模拟数据存储
    let store = Arc::new(MockCandidateStore::new());

    MiniRecommenderPipeline::new(
        // Query Hydrators
        vec![
            Box::new(UserPreferenceHydrator::new()),
            Box::new(ExcludedItemsHydrator),
        ],

        // Sources
        vec![
            Box::new(PreferenceBasedSource::new(Arc::clone(&store))),
            Box::new(TrendingSource::new(store)),
        ],

        // Hydrators (简化版不使用)
        vec![],

        // Filters
        vec![
            Box::new(DedupFilter),
            Box::new(CategoryFilter),
        ],

        // Scorers
        vec![
            Box::new(SimpleScorer::new()),
            Box::new(AuthorDiversityScorer::new(0.5)),
        ],

        // Selector
        Box::new(TopKScoreSelector::new(5)),

        // Side Effects
        vec![],

        // Result size
        5,
    )
}
