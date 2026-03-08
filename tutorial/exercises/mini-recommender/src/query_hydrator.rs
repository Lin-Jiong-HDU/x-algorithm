use crate::candidate::ItemCandidate;
use crate::pipeline_traits::QueryHydrator;
use crate::query::{RecommendQuery, UserPreferences};
use async_trait::async_trait;
use std::collections::HashSet;

/// 模拟的用户偏好数据
pub struct MockUserPreferenceClient {
    // user_id -> liked_categories
    preferences: std::collections::HashMap<u64, Vec<String>>,
}

impl MockUserPreferenceClient {
    pub fn new() -> Self {
        let mut preferences = std::collections::HashMap::new();

        // 添加一些模拟数据
        preferences.insert(1, vec!["technology".to_string(), "science".to_string()]);
        preferences.insert(2, vec!["sports".to_string(), "entertainment".to_string()]);
        preferences.insert(3, vec!["technology".to_string()]);

        Self { preferences }
    }

    pub fn get_preferences(&self, user_id: u64) -> Option<Vec<String>> {
        self.preferences.get(&user_id).cloned()
    }
}

/// 用户偏好 Hydrator
pub struct UserPreferenceHydrator {
    client: MockUserPreferenceClient,
}

impl UserPreferenceHydrator {
    pub fn new() -> Self {
        Self {
            client: MockUserPreferenceClient::new(),
        }
    }
}

#[async_trait]
impl QueryHydrator for UserPreferenceHydrator {
    async fn hydrate(&self, query: &RecommendQuery) -> Result<RecommendQuery, String> {
        let liked_categories = self
            .client
            .get_preferences(query.user_id)
            .unwrap_or_default();

        Ok(RecommendQuery {
            user_preferences: Some(UserPreferences {
                liked_categories,
                excluded_author_ids: vec![],
                preferred_category: None,
            }),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut RecommendQuery, hydrated: RecommendQuery) {
        query.user_preferences = hydrated.user_preferences;
    }

    fn name(&self) -> &'static str {
        "UserPreferenceHydrator"
    }
}

/// 排除列表 Hydrator
pub struct ExcludedItemsHydrator;

impl ExcludedItemsHydrator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl QueryHydrator for ExcludedItemsHydrator {
    async fn hydrate(&self, query: &RecommendQuery) -> Result<RecommendQuery, String> {
        // 模拟：假设用户已经看过某些内容
        let excluded_ids = match query.user_id {
            1 => vec![101, 102],
            2 => vec![201, 202],
            _ => vec![],
        };

        let mut prefs = query.user_preferences.clone().unwrap_or_default();
        prefs.excluded_ids = excluded_ids;

        Ok(RecommendQuery {
            user_preferences: Some(prefs),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut RecommendQuery, hydrated: RecommendQuery) {
        query.user_preferences = hydrated.user_preferences;
    }

    fn name(&self) -> &'static str {
        "ExcludedItemsHydrator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_user_preference_hydrator() {
        let hydrator = UserPreferenceHydrator::new();
        let query = RecommendQuery::new(1);

        let result = hydrator.hydrate(&query).await.unwrap();
        assert!(result.user_preferences.is_some());
        assert_eq!(
            result.user_preferences.unwrap().liked_categories,
            vec!["technology", "science"]
        );
    }
}
