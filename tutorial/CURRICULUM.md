# X For You 推荐算法学习课程大纲

> 基于 X 开源推荐系统项目的 15 节课程，从基础概念到工业级实现

---

## 课程设计理念

- **循序渐进**: 从宏观架构到微观实现，逐步深入
- **理论实践结合**: 每节课包含原理讲解 + 代码阅读 + 动手练习
- **全栈视角**: 涵盖系统设计 (Rust) 和机器学习 (Python/JAX) 两个维度

---

## 第一阶段：系统概览与架构理解 (第 1-3 节)

### 第 1 节：推荐系统基础与 X 算法概览

**学习目标**: 理解现代推荐系统的核心问题，建立对 X For You 算法的宏观认知

**课程内容**:
1. 推荐系统的核心挑战
   - 海量候选集的高效筛选 (百万 → 千 → 百)
   - 个性化与多样性的平衡
   - 实时性要求

2. X For You 算法整体架构
   - In-Network vs Out-of-Network 内容来源
   - 两阶段推荐：Retrieval + Ranking
   - 无手工特征的端到端学习

3. 项目结构导览
   - `candidate-pipeline/`: 通用管道框架
   - `home-mixer/`: 核心编排层
   - `thunder/`: 实时数据层
   - `phoenix/`: ML 模型

**阅读材料**:
- `README.md` 全文
- `phoenix/README.md` 全文

**思考题**:
- 为什么推荐系统需要 Retrieval + Ranking 两阶段？
- "无手工特征"的设计有什么优缺点？

---

### 第 2 节：推荐管道的抽象设计

**学习目标**: 理解可复用推荐管道框架的设计模式

**课程内容**:
1. Candidate Pipeline 框架概览
   - Trait 设计哲学：Source, Hydrator, Filter, Scorer, Selector
   - 泛型设计：`CandidatePipeline<Q, C>`

2. 管道执行流程
   ```
   hydrate_query → fetch_candidates → hydrate → filter → score → select
   ```

3. 并行执行与错误处理
   - `join_all` 并行化
   - 组件失败时的优雅降级

**阅读材料**:
- `candidate-pipeline/candidate_pipeline.rs` (核心文件)
- `candidate-pipeline/lib.rs`

**代码重点**:
- 第 37-51 行：`CandidatePipeline` trait 定义
- 第 53-92 行：`execute` 方法的主流程
- 第 95-123 行：`hydrate_query` 并行执行模式

**动手练习**:
- 画出管道执行的数据流图
- 标注哪些阶段是并行执行的

---

### 第 3 节：Trait 系统详解

**学习目标**: 掌握每个 Trait 的职责和实现模式

**课程内容**:
1. **Source Trait**: 候选来源
   - `get_candidates()`: 获取原始候选
   - 多 Source 并行合并

2. **QueryHydrator Trait**: 查询增强
   - 丰富用户上下文信息
   - 用户行为序列获取

3. **Hydrator Trait**: 候选增强
   - 补充帖子元数据
   - 作者信息、媒体信息等

4. **Filter Trait**: 过滤
   - Pre-scoring vs Post-selection 过滤
   - 过滤逻辑的业务考量

5. **Scorer Trait**: 评分
   - 多 Scorer 串行组合
   - 分数更新机制

6. **Selector Trait**: 选择
   - 排序与截断
   - Top-K 选择策略

7. **SideEffect Trait**: 副作用
   - 异步执行，不阻塞主流程
   - 缓存、日志等用途

**阅读材料**:
- `candidate-pipeline/source.rs`
- `candidate-pipeline/query_hydrator.rs`
- `candidate-pipeline/hydrator.rs`
- `candidate-pipeline/filter.rs`
- `candidate-pipeline/scorer.rs`
- `candidate-pipeline/selector.rs`
- `candidate-pipeline/side_effect.rs`

**动手练习**:
- 为每个 Trait 写一个最小实现示例

---

## 第二阶段：Home Mixer 深入 (第 4-8 节)

### 第 4 节：Home Mixer 整体架构

**学习目标**: 理解核心编排层的组织方式

**课程内容**:
1. Home Mixer 的职责
   - 作为推荐请求的入口
   - 组装各种组件形成完整管道

2. 目录结构解析
   ```
   home-mixer/
   ├── query_hydrators/     # 查询增强器
   ├── sources/             # 候选来源
   ├── candidate_hydrators/ # 候选增强器
   ├── filters/             # 过滤器
   ├── scorers/             # 评分器
   ├── selectors/           # 选择器
   └── side_effects/        # 副作用
   ```

3. gRPC 服务设计
   - `ScoredPostsService` 接口
   - 请求/响应协议

**阅读材料**:
- `home-mixer/main.rs`: 服务启动
- `home-mixer/server.rs`: 服务实现
- `home-mixer/lib.rs`: 模块导出

**代码重点**:
- `main.rs:48-56`: gRPC 服务配置

---

### 第 5 节：Query Hydrators - 用户上下文获取

**学习目标**: 理解如何获取和丰富用户信息

**课程内容**:
1. **UserActionSeqQueryHydrator**
   - 获取用户历史行为序列
   - 行为类型：like, reply, repost, click 等
   - 为 ML 模型提供输入

2. **UserFeaturesQueryHydrator**
   - 用户关注列表
   - 用户偏好设置

**阅读材料**:
- `home-mixer/query_hydrators/user_action_seq_query_hydrator.rs`
- `home-mixer/query_hydrators/user_features_query_hydrator.rs`

**思考题**:
- 用户行为序列对推荐有什么作用？
- 如何处理冷启动用户（无历史行为）？

---

### 第 6 节：Sources - 候选来源

**学习目标**: 理解两种候选来源的工作原理

**课程内容**:
1. **ThunderSource - In-Network 候选**
   - 从关注用户获取帖子
   - 实时性要求

2. **PhoenixSource - Out-of-Network 候选**
   - ML 检索获取全局候选
   - Two-Tower 模型调用

**阅读材料**:
- `home-mixer/sources/thunder_source.rs`
- `home-mixer/sources/phoenix_source.rs`

**代码重点**:
- 两种 Source 如何并行获取候选

---

### 第 7 节：Filters - 过滤系统

**学习目标**: 掌握推荐系统中的过滤逻辑设计

**课程内容**:
1. **Pre-Scoring Filters** (评分前过滤)
   - `DropDuplicatesFilter`: 去重
   - `AgeFilter`: 时效性过滤
   - `SelfTweetFilter`: 过滤自己的帖子
   - `AuthorSocialgraphFilter`: 屏蔽/静音过滤
   - `MutedKeywordFilter`: 关键词过滤
   - `PreviouslySeenPostsFilter`: 已看过的帖子
   - `IneligibleSubscriptionFilter`: 订阅内容过滤

2. **Post-Selection Filters** (选择后过滤)
   - `VFFilter`: 可见性过滤 (删除/违规/暴力等)
   - `DedupConversationFilter`: 对话去重

**阅读材料**:
- `home-mixer/filters/` 目录下所有文件
- 重点关注每个 Filter 的 `filter()` 方法

**动手练习**:
- 设计一个新的 Filter：`LanguageFilter`，过滤非用户语言偏好的帖子

---

### 第 8 节：Scorers - 评分系统

**学习目标**: 理解多 Scorer 组合的评分机制

**课程内容**:
1. **PhoenixScorer - ML 模型评分**
   - 调用 Phoenix 模型获取预测概率
   - 多种行为概率：like, reply, repost, click 等

2. **WeightedScorer - 加权组合**
   ```
   Final Score = Σ (weight_i × P(action_i))
   ```
   - 正向行为：正权重
   - 负向行为：负权重

3. **AuthorDiversityScorer - 多样性控制**
   - 重复作者的分数衰减
   - 避免信息茧房

4. **OONScorer - Out-of-Network 调整**
   - 对网络外内容的特殊处理

**阅读材料**:
- `home-mixer/scorers/phoenix_scorer.rs`
- `home-mixer/scorers/weighted_scorer.rs`
- `home-mixer/scorers/author_diversity_scorer.rs`
- `home-mixer/scorers/oon_scorer.rs`

**代码重点**:
- `phoenix_scorer.rs:129-151`: 如何提取各种行为分数

**思考题**:
- 为什么要用多 Scorer 串行而非单一 Scorer？
- 如何平衡个性化与多样性？

---

## 第三阶段：Thunder 实时数据层 (第 9 节)

### 第 9 节：Thunder - 内存存储与实时消费

**学习目标**: 理解高性能实时数据层的设计

**课程内容**:
1. **架构设计**
   - 内存中的 PostStore
   - Kafka 消息消费
   - 自动过期清理

2. **数据结构**
   - 按用户组织的帖子存储
   - 原帖 vs 回复/转发的分离
   - 视频帖子的特殊处理

3. **并发控制**
   - Semaphore 限流
   - `spawn_blocking` 避免阻塞异步运行时

4. **可观测性**
   - 详细的指标上报
   - 请求统计与分析

**阅读材料**:
- `thunder/thunder_service.rs`
- `thunder/main.rs`
- `thunder/posts/` 目录

**代码重点**:
- `thunder_service.rs:154-330`: 核心请求处理逻辑
- `thunder_service.rs:64-148`: 统计分析与指标上报

**动手练习**:
- 分析 Thunder 如何实现亚毫秒级查询

---

## 第四阶段：Phoenix ML 模型 (第 10-13 节)

### 第 10 节：Phoenix 模型架构总览

**学习目标**: 理解推荐系统中 ML 模型的整体设计

**课程内容**:
1. **两阶段 ML 架构**
   - Retrieval: Two-Tower 模型
   - Ranking: Transformer 模型

2. **JAX/Haiku 框架**
   - 函数式神经网络定义
   - 参数管理

3. **Grok-1 架构移植**
   - 从语言模型到推荐模型的适配
   - 关键改动点

**阅读材料**:
- `phoenix/README.md`
- `phoenix/grok.py`: Transformer 基础实现
- `phoenix/pyproject.toml`: 依赖管理

**动手练习**:
```bash
cd phoenix
uv run run_ranker.py
uv run run_retrieval.py
```

---

### 第 11 节：Retrieval - Two-Tower 模型

**学习目标**: 理解大规模检索的 Two-Tower 架构

**课程内容**:
1. **Two-Tower 架构原理**
   - User Tower: 用户特征编码
   - Candidate Tower: 候选特征编码
   - 点积相似度检索

2. **Embedding 设计**
   - Hash-based Embeddings
   - 多哈希组合

3. **ANN 检索**
   - 从百万候选中快速检索
   - 近似最近邻搜索

**阅读材料**:
- `phoenix/recsys_retrieval_model.py`

**代码重点**:
- User Tower 的 Transformer 编码
- Embedding 归一化与点积相似度

---

### 第 12 节：Ranking - Transformer 排序模型

**学习目标**: 深入理解排序模型的输入输出设计

**课程内容**:
1. **输入设计**
   ```
   [User Embedding] + [History Sequence] + [Candidate Sequence]
        [B, 1, D]        [B, S, D]            [B, C, D]
   ```

2. **Candidate Isolation**
   - 候选之间不能相互 attend
   - 确保评分独立可缓存

3. **多任务预测**
   - 同时预测多种行为概率
   - like, reply, repost, click, dwell 等

4. **输出设计**
   ```
   logits: [B, num_candidates, num_actions]
   ```

**阅读材料**:
- `phoenix/recsys_model.py`

**代码重点**:
- 第 79-119 行：`block_user_reduce` 用户 embedding 组合
- 第 122-182 行：`block_history_reduce` 历史 embedding 组合
- 第 185-242 行：`block_candidate_reduce` 候选 embedding 组合
- 第 439-474 行：`__call__` 前向传播

---

### 第 13 节：Attention Mask 与 Candidate Isolation

**学习目标**: 深入理解排序模型的核心创新点

**课程内容**:
1. **Attention Mask 设计**
   ```
   │ User │    History    │   Candidates   │
   ├──────┼───────────────┼────────────────┤
   │  U   │  ✓ ✓ ✓ ✓ ✓ ✓  │  ✗ ✗ ✗ ✗ ✗ ✗  │  ← User 可 attend History
   │  H   │  ✓ ✓ ✓ ✓ ✓ ✓  │  ✗ ✗ ✗ ✗ ✗ ✗  │  ← History 双向 attention
   │  C   │  ✓ ✓ ✓ ✓ ✓ ✓  │  diagonal only │  ← Candidate 只能 attend 自己
   ```

2. **为什么需要 Candidate Isolation**
   - 分数不受其他候选影响
   - 可缓存性
   - 批处理一致性

3. **实现方式**
   - 在 `grok.py` 中查看 attention mask 生成

**阅读材料**:
- `phoenix/README.md` 第 119-155 行的 mask 可视化
- `phoenix/grok.py` 中的 attention 实现

**思考题**:
- 如果允许候选相互 attend，会有什么问题？
- 这种设计对模型训练有什么影响？

---

## 第五阶段：高级主题与实战 (第 14-15 节)

### 第 14 节：系统设计亮点总结

**学习目标**: 提炼可复用的系统设计模式

**课程内容**:
1. **架构模式**
   - 管道模式 (Pipeline Pattern)
   - 策略模式 (Strategy Pattern) - 各 Trait 的实现
   - 组合模式 (Composite Pattern) - 多 Scorer/Filter 组合

2. **性能优化**
   - 并行执行 (`join_all`)
   - 异步副作用
   - 内存缓存 (Thunder)

3. **可扩展性**
   - 新增 Source/Filter/Scorer 只需实现 Trait
   - 配置驱动的组件启用/禁用

4. **可靠性**
   - 组件失败时的优雅降级
   - 详细的日志和指标

**思考题**:
- 这个架构可以应用到哪些其他场景？
- 有哪些可以改进的地方？

---

### 第 15 节：实战项目 - 构建迷你推荐系统

**学习目标**: 综合运用所学知识，构建一个简化版推荐系统

**项目要求**:
1. 使用 `candidate-pipeline` 框架
2. 实现以下组件：
   - 1 个 Source: 从模拟数据获取候选
   - 1 个 QueryHydrator: 获取用户偏好
   - 1 个 Filter: 过滤不符合条件的候选
   - 1 个 Scorer: 简单评分逻辑
   - 1 个 Selector: Top-K 选择

3. 输出: 给定用户 ID，返回推荐的候选列表

**参考结构**:
```
mini-recommender/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── source.rs
│   ├── query_hydrator.rs
│   ├── filter.rs
│   ├── scorer.rs
│   └── selector.rs
└── data/
    └── mock_data.json
```

**评分标准**:
- 正确实现 CandidatePipeline trait
- 代码结构与原项目一致
- 有基本的单元测试

---

## 附录

### 推荐阅读顺序

```
README.md
    ↓
candidate-pipeline/  (理解框架)
    ↓
home-mixer/          (理解应用)
    ↓
thunder/             (理解数据层)
    ↓
phoenix/             (理解 ML 模型)
```

### 关键文件清单

| 优先级 | 文件 | 说明 |
|--------|------|------|
| ⭐⭐⭐ | `README.md` | 项目总览 |
| ⭐⭐⭐ | `candidate-pipeline/candidate_pipeline.rs` | 核心框架 |
| ⭐⭐⭐ | `phoenix/README.md` | ML 架构说明 |
| ⭐⭐ | `phoenix/recsys_model.py` | 排序模型 |
| ⭐⭐ | `home-mixer/scorers/phoenix_scorer.rs` | ML 调用 |
| ⭐⭐ | `thunder/thunder_service.rs` | 实时数据层 |
| ⭐ | `home-mixer/filters/*.rs` | 过滤逻辑 |
| ⭐ | `phoenix/grok.py` | Transformer 基础 |

### 扩展学习资源

1. **推荐系统理论**
   - [Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091)
   - [Two-Tower Model for Retrieval](https://arxiv.org/abs/2005.07465)

2. **Transformer 架构**
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [Grok-1 Open Source](https://github.com/xai-org/grok-1)

3. **Rust 异步编程**
   - [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
   - [Async Book](https://rust-lang.github.io/async-book/)

4. **JAX/Haiku**
   - [JAX Documentation](https://jax.readthedocs.io/)
   - [Haiku Documentation](https://dm-haiku.readthedocs.io/)

---

## 学习时间估算

| 阶段 | 节数 | 预计时间 |
|------|------|----------|
| 第一阶段：概览与架构 | 3 节 | 6-9 小时 |
| 第二阶段：Home Mixer | 5 节 | 10-15 小时 |
| 第三阶段：Thunder | 1 节 | 3-4 小时 |
| 第四阶段：Phoenix ML | 4 节 | 8-12 小时 |
| 第五阶段：实战 | 2 节 | 6-10 小时 |
| **总计** | **15 节** | **33-50 小时** |

---

*Generated for X For You Recommendation Algorithm Study*
