# 第 10 节：Phoenix 模型架构总览

> **学习目标**：理解推荐系统中 ML 模型的整体设计，掌握两阶段 ML 架构

---

## 1. 概念讲解

### 1.1 Phoenix 的两个模型

Phoenix 包含两个独立的 ML 模型：

| 模型 | 用途 | 输入规模 | 输出 |
|------|------|----------|------|
| **Retrieval** | 从百万候选中快速召回 | 百万级 | 千级候选 ID |
| **Ranking** | 对候选进行精确排序 | 千级 | 行为概率 |

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Phoenix ML Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    STAGE 1: RETRIEVAL                        │   │
│   │                                                              │   │
│   │   ┌───────────────┐         ┌───────────────┐               │   │
│   │   │  User Tower   │         │ Candidate     │               │   │
│   │   │               │         │  Tower        │               │   │
│   │   │ 用户行为序列   │         │ 所有帖子       │               │   │
│   │   │      ↓        │         │      ↓        │               │   │
│   │   │ [1, D] 向量   │  ⊙ dot  │ [N, D] 向量   │               │   │
│   │   └───────────────┘         └───────────────┘               │   │
│   │           │                        │                         │   │
│   │           └────────────────────────┘                         │   │
│   │                        ↓                                      │   │
│   │              Top-K 相似候选 ID                                │   │
│   │              (从百万中召回千级)                                │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    STAGE 2: RANKING                          │   │
│   │                                                              │   │
│   │   输入: 用户行为序列 + 候选帖子                               │   │
│   │                     ↓                                        │   │
│   │              ┌───────────────┐                               │   │
│   │              │  Transformer  │                               │   │
│   │              │  (Grok-based) │                               │   │
│   │              └───────────────┘                               │   │
│   │                     ↓                                        │   │
│   │   输出: [候选数 × 行为数] 的概率矩阵                          │   │
│   │         P(like), P(reply), P(repost), ...                    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么是两阶段？

| 考量 | Retrieval | Ranking |
|------|-----------|---------|
| **目标** | 召回率（不漏掉好内容） | 准确率（精确排序） |
| **模型复杂度** | 简单（Two-Tower） | 复杂（Transformer） |
| **计算量** | O(N) 向量检索 | O(C × D) Transformer |
| **延迟** | 极快（ANN 索引） | 较慢（需要 GPU） |
| **处理规模** | 百万 → 千 | 千 → 排序 |

**核心洞察**：用简单模型快速缩小候选集，再用复杂模型精确排序。

### 1.3 技术栈

- **框架**：JAX + Haiku（函数式神经网络）
- **基础架构**：Grok-1 Transformer（从 xAI 移植）
- **运行方式**：
  ```bash
  cd phoenix
  uv run run_ranker.py      # 运行排序模型
  uv run run_retrieval.py   # 运行检索模型
  uv run pytest             # 运行测试
  ```

---

## 2. 代码分析

### 2.1 项目结构

```
phoenix/
├── README.md                    # 架构文档（必读）
├── pyproject.toml               # 依赖管理
├── grok.py                      # Transformer 基础实现
├── recsys_model.py              # 排序模型
├── recsys_retrieval_model.py    # 检索模型
├── run_ranker.py                # 排序模型入口
├── run_retrieval.py             # 检索模型入口
├── runners.py                   # 推理运行器
└── test_*.py                    # 测试文件
```

### 2.2 Grok Transformer 基础

`grok.py` 包含从 Grok-1 移植的 Transformer 实现：

```python
@dataclass
class TransformerConfig:
    attention_output_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    output_multiplier: float = 1.0
    # ... 更多配置

class Transformer(hk.Module):
    def __call__(self, inputs, padding_mask, candidate_start_offset=None):
        # 1. Embedding 处理
        # 2. 多层 Transformer Block
        # 3. 特殊的 Attention Mask（Candidate Isolation）
        # 4. 输出
```

**关键适配**：
- 原始 Grok-1 是语言模型
- Phoenix 适配为推荐模型，关键改动是 **Candidate Isolation**

### 2.3 排序模型核心配置

```python
@dataclass
class PhoenixModelConfig:
    model: TransformerConfig       # Transformer 配置
    emb_size: int                   # Embedding 维度
    num_actions: int                # 预测的行为数量
    history_seq_len: int = 128      # 历史序列长度
    candidate_seq_len: int = 32     # 候选序列长度
    hash_config: HashConfig         # 哈希配置
```

### 2.4 运行模型

```bash
# 运行排序模型
cd phoenix
uv run run_ranker.py

# 输出示例：
# Loading model...
# Input: user_id=123, 5 candidates
# Output:
#   Candidate 0: P(like)=0.12, P(reply)=0.03, ...
#   Candidate 1: P(like)=0.08, P(reply)=0.01, ...
```

---

## 3. 实践练习

### 动手任务

1. **运行 Phoenix 模型**
   ```bash
   cd phoenix
   uv run run_ranker.py
   uv run run_retrieval.py
   ```

2. **阅读 `phoenix/README.md`**
   - 理解架构图
   - 找到 Attention Mask 的可视化

### 思考题

1. **为什么 Retrieval 使用 Two-Tower 而不是 Transformer？**

   <details>
   <summary>提示</summary>

   考虑：候选规模、计算复杂度、是否需要候选间的交互

   </details>

2. **Grok-1 是语言模型，如何适配为推荐模型？关键改动是什么？**

3. **为什么需要 `history_seq_len` 和 `candidate_seq_len` 两个参数？**

### 代码阅读

打开 `phoenix/grok.py`，找到：

1. `TransformerConfig` 的定义
2. `Transformer` 类的 `__call__` 方法
3. `candidate_start_offset` 参数的作用

---

## 小结

本节我们建立了对 Phoenix ML 模块的宏观认知：

1. **两阶段 ML 架构**：Retrieval（召回）+ Ranking（排序）
2. **技术栈**：JAX + Haiku + Grok-based Transformer
3. **Retrieval**：Two-Tower 模型，快速从百万候选中召回
4. **Ranking**：Transformer 模型，精确预测多种行为概率

下一节，我们将深入 Retrieval 模型的 Two-Tower 架构。
