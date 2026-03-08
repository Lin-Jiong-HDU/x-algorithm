# 第 1 节：推荐系统基础与 X 算法概览

> **学习目标**：理解现代推荐系统的核心挑战，建立对 X For You 算法的宏观认知

---

## 1. 概念讲解

### 1.1 推荐系统的核心挑战

推荐系统本质上是一个**信息过滤**问题：从海量内容中筛选出用户最可能感兴趣的内容。在实际工程中，这面临三个核心挑战：

#### 挑战一：海量候选集的高效筛选

社交媒体平台每天产生数以亿计的内容，但用户每次刷新只能看到几十条。如何从百万级候选中快速筛选出最相关的几百条？

**工业界的解决方案**：两阶段架构

```
┌─────────────────────────────────────────────────────────────┐
│                    推荐系统漏斗                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   百万级候选 ──▶ Retrieval ──▶ 千级候选 ──▶ Ranking ──▶ 百级 │
│                 (粗筛)                    (精排)            │
│                                                             │
│   Retrieval: 简单模型，极快速度，召回率优先                  │
│   Ranking:  复杂模型，较慢速度，准确率优先                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

这种设计的关键洞察是：**不是所有候选都需要精确评分**。先用轻量级模型快速召回可能相关的候选，再用复杂模型对少量候选进行精确排序。

#### 挑战二：个性化与多样性的平衡

纯粹的个性化会导致"信息茧房"——用户只看到与过去相似的内容。好的推荐系统需要：

- **相关性**：推荐用户可能喜欢的内容
- **多样性**：避免内容同质化
- **新颖性**：偶尔推荐用户未曾接触的领域
- **时效性**：优先展示新鲜内容

X 的做法是同时考虑 **In-Network**（关注的人）和 **Out-of-Network**（全局发现）两个来源，确保用户既能看到熟悉的内容，也能发现新的兴趣点。

#### 挑战三：实时性要求

用户刷新页面时，推荐结果必须在几百毫秒内返回。这意味着：

- 数据获取要快（内存缓存 > 数据库）
- 模型推理要快（模型不能太复杂）
- 系统要能处理高并发（异步、并行）

---

### 1.2 X For You 算法整体架构

X 的 For You 算法采用了现代化的推荐系统架构，核心设计理念是**端到端学习**——系统几乎不依赖手工设计的特征，而是让模型从用户行为中自动学习。

#### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FOR YOU FEED 请求                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           HOME MIXER                                 │
│                        (编排协调层)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────┐      ┌─────────────────────┐              │
│   │      THUNDER        │      │   PHOENIX RETRIEVAL │              │
│   │   (In-Network)      │      │   (Out-of-Network)  │              │
│   │                     │      │                     │              │
│   │  关注用户的帖子      │      │  ML 检索全局候选     │              │
│   └─────────────────────┘      └─────────────────────┘              │
│                │                          │                         │
│                └──────────┬───────────────┘                         │
│                           ▼                                         │
│                    合并 + 过滤                                        │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    PHOENIX SCORER                            │   │
│   │              (Grok-based Transformer)                        │   │
│   │                                                              │   │
│   │   输入: 用户行为序列 + 候选帖子                                │   │
│   │   输出: P(like), P(reply), P(repost), P(click)...           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│                    加权评分 + 排序                                   │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            ▼
                   返回排序后的 Feed
```

#### 四大核心组件

| 组件                   | 语言       | 职责                                                                   |
| ---------------------- | ---------- | ---------------------------------------------------------------------- |
| **candidate-pipeline** | Rust       | 通用推荐管道框架，定义 Source/Hydrator/Filter/Scorer/Selector 等 Trait |
| **home-mixer**         | Rust       | 核心编排层，组装完整推荐管道，暴露 gRPC 服务                           |
| **thunder**            | Rust       | 实时数据层，内存存储最近帖子，支持亚毫秒级查询                         |
| **phoenix**            | Python/JAX | ML 模型，包含 Retrieval（Two-Tower）和 Ranking（Transformer）          |

#### 关键设计决策

1. **无手工特征**：系统不依赖人工设计的特征（如"用户过去 7 天点赞数"），而是让 Transformer 从用户行为序列中自动学习

2. **Candidate Isolation**：排序时候选帖子之间不能相互 attend，确保每个帖子的评分独立，可缓存

3. **多任务预测**：模型同时预测多种行为概率（点赞、回复、转发、点击等），最终分数是加权组合

4. **两阶段 Retrieval**：
   - In-Network：从关注用户获取（Thunder）
   - Out-of-Network：ML 检索全局候选（Phoenix Retrieval）

---

## 2. 代码分析

### 2.1 项目结构

```
x-algorithm/
├── README.md                 # 项目总览，必读
├── candidate-pipeline/       # 核心框架
│   ├── lib.rs
│   ├── candidate_pipeline.rs # CandidatePipeline trait 定义
│   ├── source.rs             # Source trait
│   ├── hydrator.rs           # Hydrator trait
│   ├── filter.rs             # Filter trait
│   ├── scorer.rs             # Scorer trait
│   ├── selector.rs           # Selector trait
│   ├── query_hydrator.rs     # QueryHydrator trait
│   └── side_effect.rs        # SideEffect trait
├── home-mixer/               # 编排层实现
│   ├── main.rs
│   ├── server.rs
│   ├── query_hydrators/      # 用户上下文获取
│   ├── sources/              # 候选来源
│   ├── candidate_hydrators/  # 候选信息补充
│   ├── filters/              # 过滤逻辑
│   ├── scorers/              # 评分逻辑
│   └── side_effects/         # 副作用（缓存等）
├── thunder/                  # 实时数据层
│   ├── main.rs
│   ├── thunder_service.rs    # gRPC 服务
│   └── posts/                # 帖子存储
└── phoenix/                  # ML 模型
    ├── README.md             # 模型架构详解
    ├── grok.py               # Transformer 基础实现
    ├── recsys_model.py       # 排序模型
    └── recsys_retrieval_model.py  # 检索模型
```

### 2.2 README 架构图解读

打开 `README.md`，你会看到完整的系统架构图（第 40-122 行）。这个图值得仔细研究，它展示了：

1. **请求入口**：`FOR YOU FEED REQUEST` 进入 `HOME MIXER`

2. **Query Hydration**：获取用户上下文
   - User Action Sequence：用户历史行为
   - User Features：关注列表、偏好等

3. **Candidate Sources**：两个来源并行获取
   - Thunder：关注用户的帖子
   - Phoenix Retrieval：ML 检索

4. **Hydration**：补充候选信息（作者、媒体等）

5. **Filtering**：两阶段过滤
   - Pre-Scoring：评分前过滤（去重、时效、屏蔽等）
   - Post-Selection：选择后过滤（可见性检查）

6. **Scoring**：三层评分
   - Phoenix Scorer：ML 预测
   - Weighted Scorer：加权组合
   - Author Diversity Scorer：多样性控制

7. **Selection**：排序 + Top-K 选择

### 2.3 阅读建议

对于有 Rust 和 ML 基础的开发者，建议的阅读顺序：

```
1. README.md (全文)
       ↓
2. candidate-pipeline/candidate_pipeline.rs (理解框架)
       ↓
3. home-mixer/main.rs + server.rs (理解入口)
       ↓
4. phoenix/README.md (理解 ML 架构)
       ↓
5. 按需深入各模块
```

---

## 3. 实践练习

### 动手任务

1. **运行 Phoenix 模型**

   ```bash
   cd phoenix
   uv run run_ranker.py      # 运行排序模型
   uv run run_retrieval.py   # 运行检索模型
   ```

2. **绘制数据流图**
   - 在纸上或用工具画出完整的数据流
   - 标注每个阶段的输入输出
   - 标出哪些阶段是并行执行的

### 思考题

1. **为什么推荐系统需要 Retrieval + Ranking 两阶段？为什么不用一个复杂模型直接处理所有候选？**

   <details>
   <summary>提示</summary>

   考虑计算复杂度：假设有 1000 万候选，一个复杂模型需要 10ms 处理一个候选...

   </details>

2. **"无手工特征"的设计有什么优缺点？**

   <details>
   <summary>提示</summary>

   优点：减少工程复杂度，模型自动发现模式
   缺点：需要大量训练数据，可解释性降低

   </details>

3. **为什么需要同时考虑 In-Network 和 Out-of-Network 内容？**

### 扩展阅读

- [Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091) - Facebook 的推荐模型
- [Twitter 如何设计推荐系统](https://blog.twitter.com/engineering/en_us/topics/insights/2018/twitter-recommendations) - 官方博客

---

## 小结

本节我们建立了对推荐系统和 X For You 算法的宏观认知：

1. **推荐系统三大挑战**：海量筛选、个性化与多样性平衡、实时性
2. **两阶段架构**：Retrieval（粗筛）+ Ranking（精排）
3. **X 的四大组件**：candidate-pipeline、home-mixer、thunder、phoenix
4. **核心设计理念**：无手工特征、Candidate Isolation、多任务预测

下一节，我们将深入 `candidate-pipeline` 框架，理解它是如何通过 Trait 系统实现可复用的推荐管道。
