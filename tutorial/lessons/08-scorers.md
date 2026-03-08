# 第 8 节：Scorers - 评分系统

> **学习目标**：理解多 Scorer 组合的评分机制，掌握如何将 ML 预测转换为最终分数

---

## 1. 概念讲解

### 1.1 评分系统的设计

X 的评分系统采用**多 Scorer 串行**的设计：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Scoring Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐                                               │
│   │ PhoenixScorer   │  ML 模型预测                                  │
│   │                 │  → P(like), P(reply), P(repost), ...          │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │ WeightedScorer  │  加权组合                                     │
│   │                 │  → weighted_score = Σ (weight × P(action))    │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────────┐                                           │
│   │AuthorDiversityScorer│  多样性控制                               │
│   │                     │  → 重复作者分数衰减                        │
│   └────────┬────────────┘                                           │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │   OONScorer     │  Out-of-Network 调整                         │
│   │                 │  → 对非关注内容的特殊处理                     │
│   └─────────────────┘                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么是串行而非并行？

与 Hydrator 不同，Scorer 是**顺序执行**的，因为：

1. **数据依赖**：`WeightedScorer` 需要 `PhoenixScorer` 的输出
2. **逐步调整**：每个 Scorer 基于前一个的分数进行调整

---

## 2. 代码分析

### 2.1 PhoenixScorer

**文件**：`home-mixer/scorers/phoenix_scorer.rs`

调用 ML 模型获取各种行为的预测概率。

```rust
pub struct PhoenixScorer {
    pub phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
}

async fn score(&self, query: &ScoredPostsQuery, candidates: &[PostCandidate])
    -> Result<Vec<PostCandidate>, String>
{
    let user_id = query.user_id as u64;

    // 准备请求
    let tweet_infos: Vec<TweetInfo> = candidates.iter().map(|c| {
        TweetInfo {
            tweet_id: c.retweeted_tweet_id.unwrap_or(c.tweet_id as u64),
            author_id: c.retweeted_user_id.unwrap_or(c.author_id),
        }
    }).collect();

    // 调用 ML 模型
    let result = self.phoenix_client
        .predict(user_id, sequence.clone(), tweet_infos)
        .await?;

    // 提取分数
    let predictions_map = self.build_predictions_map(&response);

    let scored_candidates = candidates.iter().map(|c| {
        let lookup_tweet_id = c.retweeted_tweet_id.unwrap_or(c.tweet_id as u64);
        let phoenix_scores = predictions_map
            .get(&lookup_tweet_id)
            .map(|preds| self.extract_phoenix_scores(preds))
            .unwrap_or_default();

        PostCandidate {
            phoenix_scores,
            prediction_request_id: Some(request_id),
            last_scored_at_ms,
            ..Default::default()
        }
    }).collect();

    Ok(scored_candidates)
}
```

**PhoenixScores 包含 19 种行为预测**：

```rust
pub struct PhoenixScores {
    // 正向行为
    pub favorite_score: Option<f64>,      // 点赞
    pub reply_score: Option<f64>,         // 回复
    pub retweet_score: Option<f64>,       // 转发
    pub click_score: Option<f64>,         // 点击
    pub share_score: Option<f64>,         // 分享
    pub dwell_score: Option<f64>,         // 停留
    pub follow_author_score: Option<f64>, // 关注作者

    // 负向行为
    pub not_interested_score: Option<f64>,  // 不感兴趣
    pub block_author_score: Option<f64>,    // 屏蔽作者
    pub mute_author_score: Option<f64>,     // 静音作者
    pub report_score: Option<f64>,          // 举报

    // ... 更多行为
}
```

### 2.2 WeightedScorer

**文件**：`home-mixer/scorers/weighted_scorer.rs`

将多种行为预测**加权组合**成单一分数。

```rust
fn compute_weighted_score(candidate: &PostCandidate) -> f64 {
    let s = &candidate.phoenix_scores;

    // 正向行为（正权重）
    let positive = Self::apply(s.favorite_score, FAVORITE_WEIGHT)
        + Self::apply(s.reply_score, REPLY_WEIGHT)
        + Self::apply(s.retweet_score, RETWEET_WEIGHT)
        + Self::apply(s.click_score, CLICK_WEIGHT)
        // ... 更多正向行为

    // 负向行为（负权重）
    let negative = Self::apply(s.not_interested_score, NOT_INTERESTED_WEIGHT)  // 负数
        + Self::apply(s.block_author_score, BLOCK_AUTHOR_WEIGHT)      // 负数
        + Self::apply(s.mute_author_score, MUTE_AUTHOR_WEIGHT)         // 负数
        + Self::apply(s.report_score, REPORT_WEIGHT);                   // 负数

    positive + negative
}
```

**公式**：

```
Final Score = Σ (weight_i × P(action_i))

其中：
- 正向行为（like, reply）使用正权重
- 负向行为（block, report）使用负权重
```

**视频特殊处理**：

```rust
fn vqv_weight_eligibility(candidate: &PostCandidate) -> f64 {
    // 只有视频时长超过阈值才有 VQV 分数
    if candidate.video_duration_ms.is_some_and(|ms| ms > MIN_VIDEO_DURATION_MS) {
        VQV_WEIGHT
    } else {
        0.0
    }
}
```

### 2.3 AuthorDiversityScorer

**文件**：`home-mixer/scorers/author_diversity_scorer.rs`

确保 Feed 不会充斥同一个作者的内容。

```rust
pub struct AuthorDiversityScorer {
    decay_factor: f64,  // 衰减因子
    floor: f64,          // 最低乘数
}

fn multiplier(&self, position: usize) -> f64 {
    // 指数衰减：第 N 个相同作者的帖子分数乘以 decay^N
    (1.0 - self.floor) * self.decay_factor.powf(position as f64) + self.floor
}
```

**处理流程**：

```rust
async fn score(&self, candidates: &[PostCandidate]) -> Result<Vec<PostCandidate>, String> {
    let mut author_counts: HashMap<u64, usize> = HashMap::new();

    // 按分数排序后处理（高分优先）
    let mut ordered: Vec<_> = candidates.iter().enumerate().collect();
    ordered.sort_by(|(_, a), (_, b)| {
        b.weighted_score.partial_cmp(&a.weighted_score).unwrap()
    });

    for (original_idx, candidate) in ordered {
        // 获取这是该作者的第几个帖子
        let position = author_counts.entry(candidate.author_id).or_insert(0);
        let multiplier = self.multiplier(*position);
        *position += 1;

        // 应用衰减
        let adjusted_score = candidate.weighted_score.map(|s| s * multiplier);
        scored[original_idx] = PostCandidate { score: adjusted_score, .. };
    }

    Ok(scored)
}
```

**效果示例**：

```
假设 decay_factor = 0.5, floor = 0.1

作者 A 的帖子：
  第 1 个：score × 1.0   (multiplier = (1-0.1) * 0.5^0 + 0.1 = 1.0)
  第 2 个：score × 0.55  (multiplier = (1-0.1) * 0.5^1 + 0.1 = 0.55)
  第 3 个：score × 0.325 (multiplier = (1-0.1) * 0.5^2 + 0.1 = 0.325)
  ...
```

### 2.4 OONScorer

对 Out-of-Network 内容的特殊调整，确保 Feed 中有适当的非关注内容比例。

---

## 3. 实践练习

### 思考题

1. **为什么负向行为（如 block）使用负权重？如果使用 0 权重会怎样？**

2. **AuthorDiversityScorer 为什么按分数排序后再计算衰减？如果不排序会怎样？**

3. **如果要添加一个新的行为预测（如"收藏到书签"），需要修改哪些地方？**

### 代码挑战

**实现一个简单的 `RecencyBoostScorer`**：给新帖子加分

```rust
pub struct RecencyBoostScorer {
    pub boost_factor: f64,      // 每小时衰减
    pub max_boost: f64,         // 最大加成
}

// 提示：使用 candidate.created_at 和当前时间计算年龄
// 年龄越小，加成越大
```

---

## 小结

本节我们理解了评分系统的设计：

1. **4 个 Scorer 串行执行**：Phoenix → Weighted → Diversity → OON
2. **PhoenixScorer**：ML 模型预测 19 种行为概率
3. **WeightedScorer**：加权组合成单一分数，正向行为正权重，负向行为负权重
4. **AuthorDiversityScorer**：通过指数衰减避免同一作者垄断 Feed
5. **顺序执行的原因**：后续 Scorer 依赖前一个的输出

下一节，我们将进入 Thunder 模块，理解实时数据层的设计。
