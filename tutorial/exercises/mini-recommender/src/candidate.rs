use serde::{Deserialize, Serialize};

/// 推荐候选项
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ItemCandidate {
    pub item_id: u64,
    pub title: String,
    pub category: String,
    pub created_at: i64,
    pub author_id: u64,

    // 各阶段填充的字段
    pub is_trending: Option<bool>,      // Hydrator 填充
    pub score: Option<f64>,             // Scorer 填充
}

impl ItemCandidate {
    pub fn new(
        item_id: u64,
        title: &str,
        category: &str,
        author_id: u64,
    ) -> Self {
        Self {
            item_id,
            title: title.to_string(),
            category: category.to_string(),
            created_at: chrono::Utc::now().timestamp() as i64,
            author_id,
            is_trending: None,
            score: None,
        }
    }

    pub fn with_created_at(mut self, created_at: i64) -> Self {
        self.created_at = created_at;
        self
    }
}
