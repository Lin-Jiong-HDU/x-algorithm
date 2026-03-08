# 第 9 节：Thunder - 实时数据层

> **学习目标**：理解高性能实时数据层的设计，掌握内存存储和并发控制

---

## 1. 概念讲解

### 1.1 Thunder 的职责

Thunder 是 X 推荐系统的**实时数据层**，负责：

1. **实时存储**：维护最近发布的帖子（内存中）
2. **数据消费**：从 Kafka 消费帖子创建/删除事件
3. **快速查询**：支持亚毫秒级的 In-Network 候选获取

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Thunder 架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐        ┌─────────────────────────────────────┐   │
│   │   Kafka     │ ──────▶│           PostStore                 │   │
│   │  帖子事件   │        │                                     │   │
│   └─────────────┘        │   ┌─────────────────────────────┐   │   │
│                          │   │    user_id → Vec<Post>       │   │   │
│                          │   │    user_id → Vec<Reply>      │   │   │
│                          │   │    user_id → Vec<VideoPost>  │   │   │
│                          │   └─────────────────────────────┘   │   │
│                          │                                     │   │
│                          │   特点：                            │   │
│                          │   - 全内存存储                       │   │
│                          │   - 按用户组织                       │   │
│                          │   - 自动过期清理                     │   │
│                          └─────────────────────────────────────┘   │
│                                        │                            │
│                                        ▼                            │
│                          ┌─────────────────────────────────────┐   │
│                          │         gRPC Service                 │   │
│                          │   get_in_network_posts(user_id)      │   │
│                          │   延迟 < 1ms                         │   │
│                          └─────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么需要 Thunder？

**问题**：如果每次推荐请求都从数据库查询关注用户的帖子，会有什么问题？

1. **延迟高**：数据库查询通常需要 10-100ms
2. **并发压力大**：每秒数万请求会压垮数据库
3. **数据不新鲜**：数据库同步有延迟

**Thunder 的解决方案**：

- **内存存储**：数据在 RAM 中，访问极快
- **按用户组织**：直接根据关注列表批量获取
- **实时更新**：Kafka 消费确保数据新鲜

---

## 2. 代码分析

### 2.1 服务结构

```rust
pub struct ThunderServiceImpl {
    post_store: Arc<PostStore>,        // 内存存储
    strato_client: Arc<StratoClient>,  // 获取关注列表
    request_semaphore: Arc<Semaphore>, // 并发控制
}
```

### 2.2 核心请求处理

```rust
async fn get_in_network_posts(&self, request: Request<GetInNetworkPostsRequest>)
    -> Result<Response<GetInNetworkPostsResponse>, Status>
{
    // 1. 并发控制：获取信号量
    let _permit = match self.request_semaphore.try_acquire() {
        Ok(permit) => {
            IN_FLIGHT_REQUESTS.inc();
            permit
        }
        Err(_) => {
            REJECTED_REQUESTS.inc();
            return Err(Status::resource_exhausted("Server at capacity"));
        }
    };

    // 2. 获取关注列表（如果请求中没有）
    let following_user_ids = if req.following_user_ids.is_empty() {
        self.strato_client
            .fetch_following_list(req.user_id, MAX_INPUT_LIST_SIZE)
            .await?
    } else {
        req.following_user_ids
    };

    // 3. 使用 spawn_blocking 避免阻塞异步运行时
    let post_store = Arc::clone(&self.post_store);
    let proto_posts = tokio::task::spawn_blocking(move || {
        // 创建排除集合
        let exclude_tweet_ids: HashSet<i64> = req.exclude_tweet_ids.iter()
            .map(|&id| id as i64)
            .collect();

        // 从 PostStore 获取帖子
        let all_posts = if req.is_video_request {
            post_store.get_videos_by_users(&following_user_ids, &exclude_tweet_ids, ...)
        } else {
            post_store.get_all_posts_by_users(&following_user_ids, &exclude_tweet_ids, ...)
        };

        // 按时间排序并截断
        score_recent(all_posts, max_results)
    }).await?;

    Ok(Response::new(GetInNetworkPostsResponse { posts: proto_posts }))
}
```

### 2.3 并发控制

```rust
// 初始化时设置最大并发数
request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests))

// 请求处理时
let _permit = match self.request_semaphore.try_acquire() {
    Ok(permit) => permit,
    Err(_) => return Err(Status::resource_exhausted("Server at capacity")),
};
```

**设计要点**：

- **`try_acquire`**：非阻塞，立即返回结果
- **资源耗尽时快速失败**：返回 `RESOURCE_EXHAUSTED` 而不是等待
- **Guard 模式**：`_permit` 离开作用域时自动释放信号量

### 2.4 避免阻塞异步运行时

```rust
tokio::task::spawn_blocking(move || {
    // 在这里执行可能阻塞的操作
    post_store.get_all_posts_by_users(...)
}).await
```

**为什么需要 `spawn_blocking`？**

- Tokio 的异步运行时使用工作线程池
- 阻塞操作会占用工作线程，影响其他任务
- `spawn_blocking` 将阻塞操作移到专门的阻塞线程池

### 2.5 统计分析与可观测性

```rust
fn analyze_and_report_post_statistics(posts: &[LightPost], stage: &str) {
    // 计算各种统计指标
    let time_since_most_recent = posts.iter().map(|p| p.created_at).max();
    let unique_authors: HashSet<_> = posts.iter().map(|p| p.author_id).collect();
    let reply_ratio = reply_count as f64 / posts.len() as f64;

    // 上报指标
    GET_IN_NETWORK_POSTS_FOUND_FRESHNESS_SECONDS
        .with_label_values(&[stage])
        .observe(freshness);

    // 记录日志
    debug!(
        "[{}] total={}, unique_authors={}, reply_ratio={:.2}",
        stage, posts.len(), unique_authors.len(), reply_ratio
    );
}
```

**两个阶段的统计**：
1. `retrieved`：刚从 PostStore 获取时
2. `scored`：按时间排序后

### 2.6 简单评分策略

```rust
fn score_recent(mut light_posts: Vec<LightPost>, max_results: usize) -> Vec<LightPost> {
    // 按创建时间降序排序
    light_posts.sort_unstable_by_key(|post| Reverse(post.created_at));

    // 截断到最大结果数
    light_posts.into_iter().take(max_results).collect()
}
```

Thunder 只做简单的**时间排序**，复杂的 ML 评分在后续的 Phoenix Scorer 中完成。

---

## 3. 实践练习

### 思考题

1. **为什么 Thunder 使用信号量而不是队列来控制并发？**

2. **`spawn_blocking` 和直接在异步函数中执行阻塞操作有什么区别？**

3. **如果 PostStore 中的数据量增长到内存无法容纳，应该如何处理？**

### 性能分析

**计算 Thunder 的理论延迟**：

假设：
- 关注用户数：500
- 每用户平均帖子：50
- 内存访问延迟：100ns
- 排序算法：O(n log n)

计算总延迟并分析瓶颈。

### 代码挑战

**实现一个简单的 `PostStore` 模拟**：

```rust
use std::collections::HashMap;
use std::sync::RwLock;

pub struct Post {
    pub post_id: u64,
    pub author_id: u64,
    pub created_at: i64,
}

pub struct PostStore {
    // user_id -> posts
    store: RwLock<HashMap<u64, Vec<Post>>>,
    max_posts_per_user: usize,
}

impl PostStore {
    pub fn add_post(&self, post: Post) {
        // TODO: 实现
    }

    pub fn get_posts_by_users(&self, user_ids: &[u64], exclude_ids: &HashSet<u64>) -> Vec<Post> {
        // TODO: 实现
    }

    pub fn cleanup_old_posts(&self, max_age_seconds: i64) {
        // TODO: 实现
    }
}
```

---

## 小结

本节我们理解了 Thunder 实时数据层的设计：

1. **职责**：内存存储最近帖子，支持亚毫秒级查询
2. **数据来源**：Kafka 消费帖子事件
3. **并发控制**：Semaphore 限制最大并发，资源耗尽时快速失败
4. **异步处理**：`spawn_blocking` 避免阻塞异步运行时
5. **可观测性**：详细的统计指标和日志

下一节，我们将进入 Phoenix ML 模块，理解推荐系统的机器学习部分。
