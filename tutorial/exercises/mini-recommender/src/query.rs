use xai_candidate_pipeline::candidate_pipeline::HasRequestId;

/// 用户偏好设置
#[derive(Clone, Debug, Default)]
pub struct UserPreferences {
    pub liked_categories: Vec<String>,
    pub excluded_author_ids: Vec<u64>,
    pub preferred_category: Option<String>,
}

/// 推荐查询请求
#[derive(Clone, Debug, Default)]
pub struct RecommendQuery {
    pub user_id: u64,
    pub request_id: String,

    // Query Hydrator 填充
    pub user_preferences: Option<UserPreferences>,

    // 上下文信息
    pub max_results: usize,
    pub exclude_ids: Vec<u64>,
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
            user_preferences: None,
            max_results: 10,
            exclude_ids: Vec::new(),
        }
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    pub fn with_exclude_ids(mut self, exclude_ids: Vec<u64>) -> Self {
        self.exclude_ids = exclude_ids;
        self
    }
}
