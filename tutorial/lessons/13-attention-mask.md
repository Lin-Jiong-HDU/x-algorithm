# 第 13 节：Attention Mask 与 Candidate Isolation

> **学习目标**：深入理解排序模型的核心创新点——Candidate Isolation 设计

---

## 1. 概念讲解

### 1.1 问题：候选之间的相互影响

在标准 Transformer 中，所有 Token 可以相互 Attend：

```
标准 Attention（所有位置可以相互 attend）：

位置:   [User] [H1] [H2] [H3] [C1] [C2] [C3]
        ┌─────────────────────────────────────┐
User  │  ✓    ✓    ✓    ✓    ✓    ✓    ✓   │
H1    │  ✓    ✓    ✓    ✓    ✓    ✓    ✓   │
H2    │  ✓    ✓    ✓    ✓    ✓    ✓    ✓   │
H3    │  ✓    ✓    ✓    ✓    ✓    ✓    ✓   │
C1    │  ✓    ✓    ✓    ✓    ✓    ✓    ✓   │  ← C1 可以 attend C2, C3
C2    │  ✓    ✓    ✓    ✓    ✓    ✓    ✓   │  ← C2 可以 attend C1, C3
C3    │  ✓    ✓    ✓    ✓    ✓    ✓    ✓   │  ← C3 可以 attend C1, C2
        └─────────────────────────────────────┘

问题：C1 的分数会依赖于 C2 和 C3 是什么！
```

**这有什么问题？**

1. **分数不稳定**：同一个候选，在不同批次的分数会不同
2. **不可缓存**：必须每次都重新计算，不能复用
3. **难以解释**：用户为什么对 C1 感兴趣？可能是因为 C2 也很吸引人

### 1.2 解决方案：Candidate Isolation

```
Candidate Isolation Attention Mask：

位置:   [User] [H1] [H2] [H3] [C1] [C2] [C3]
        ┌─────────────────────────────────────┐
User  │  ✓    ✓    ✓    ✓    ✗    ✗    ✗   │  User/Hist 不能 attend 候选
H1    │  ✓    ✓    ✓    ✓    ✗    ✗    ✗   │
H2    │  ✓    ✓    ✓    ✓    ✗    ✗    ✗   │
H3    │  ✓    ✓    ✓    ✓    ✗    ✗    ✗   │
        ├─────────────────────────────────────┤
C1    │  ✓    ✓    ✓    ✓    ✓    ✗    ✗   │  C1 只能 attend User+Hist+自己
C2    │  ✓    ✓    ✓    ✓    ✗    ✓    ✗   │  C2 只能 attend User+Hist+自己
C3    │  ✓    ✓    ✓    ✓    ✗    ✗    ✓   │  C3 只能 attend User+Hist+自己
        └─────────────────────────────────────┘

✓ = Can attend    ✗ = Cannot attend
```

**关键设计**：
- User 和 History 之间：**双向 Attention**
- 候选 → User/History：**可以 Attend**
- 候选 → 候选：**只能 Attend 自己**（对角线）

---

## 2. 代码分析

### 2.1 Attention Mask 生成

在 `grok.py` 中的 Transformer 实现：

```python
def create_attention_mask(padding_mask, candidate_start_offset, seq_len):
    """创建 Candidate Isolation 的 Attention Mask

    Args:
        padding_mask: [B, T] - 1 表示有效，0 表示 padding
        candidate_start_offset: int - 候选开始的位置
        seq_len: int - 序列长度

    Returns:
        attention_mask: [B, 1, T, T] - 用于 scaled_dot_product_attention
    """
    B = padding_mask.shape[0]

    # 1. 创建基础的因果 mask（下三角）
    # 但我们要的不是标准因果，而是特殊的隔离模式

    # 2. 创建二维 mask
    mask = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)

    # 3. 候选区域：只允许对角线（自己 attend 自己）
    if candidate_start_offset is not None:
        candidate_len = seq_len - candidate_start_offset

        # 候选之间不能相互 attend
        candidate_region = jnp.zeros((candidate_len, candidate_len), dtype=jnp.bool_)
        # 除了对角线
        candidate_region = candidate_region.at[jnp.diag_indices(candidate_len)].set(True)

        # 放入整体 mask
        mask = mask.at[candidate_start_offset:, candidate_start_offset:].set(candidate_region)

    # 4. 应用 padding mask
    # 如果某个位置是 padding，它不能被任何位置 attend
    # 也不能 attend 任何位置

    # 5. 扩展为 [B, 1, T, T]
    attention_mask = jnp.broadcast_to(mask[None, None, :, :], (B, 1, seq_len, seq_len))

    return attention_mask
```

### 2.2 可视化

```python
# 示例：1 个用户 + 3 个历史 + 2 个候选
seq_len = 6
candidate_start_offset = 4  # 候选从位置 4 开始

# 生成的 mask:
#       0    1    2    3    4    5
#    ┌────────────────────────────────┐
# 0  │  1    1    1    1    0    0   │  User 可以 attend User + Hist
# 1  │  1    1    1    1    0    0   │  Hist 可以相互 attend
# 2  │  1    1    1    1    0    0   │
# 3  │  1    1    1    1    0    0   │
#    ├────────────────────────────────┤
# 4  │  1    1    1    1    1    0   │  C1 可以 attend User + Hist + 自己
# 5  │  1    1    1    1    0    1   │  C2 可以 attend User + Hist + 自己
#    └────────────────────────────────┘
```

### 2.3 在 Transformer 中的应用

```python
class Transformer(hk.Module):
    def __call__(self, inputs, padding_mask, candidate_start_offset=None):
        seq_len = inputs.shape[1]

        # 创建 attention mask
        attention_mask = self.create_attention_mask(
            padding_mask,
            candidate_start_offset,
            seq_len
        )

        # Transformer blocks
        x = inputs
        for block in self.blocks:
            x = block(x, attention_mask)

        return TransformerOutput(embeddings=x)
```

---

## 3. 为什么需要 Candidate Isolation？

### 3.1 分数一致性

**没有 Isolation**：
```
批次 1: 候选 = [A, B, C]
  → A 的分数 = f(A, B, C)  ← 依赖于 B, C

批次 2: 候选 = [A, D, E]
  → A 的分数 = f(A, D, E)  ← 依赖于 D, E

问题：同一个 A，不同批次分数不同！
```

**有 Isolation**：
```
批次 1: 候选 = [A, B, C]
  → A 的分数 = f(A, User, History)  ← 只依赖 User + History

批次 2: 候选 = [A, D, E]
  → A 的分数 = f(A, User, History)  ← 相同！

优势：分数稳定一致
```

### 3.2 可缓存性

```python
# 没有 Isolation：必须每次完整计算
def score_without_isolation(user, candidates):
    # 候选数量变化时，需要重新计算所有
    return model(concat(user, history, candidates))

# 有 Isolation：可以缓存候选的中间表示
def score_with_isolation(user, candidates):
    # 1. 计算 User + History 的表示（可缓存）
    user_repr = model.user_encoder(concat(user, history))

    # 2. 每个候选独立计算（可并行、可缓存）
    scores = [model.candidate_scorer(user_repr, c) for c in candidates]

    return scores
```

### 3.3 批处理一致性

在训练时，不同批次可能包含不同的候选组合。Candidate Isolation 确保模型学到的模式不依赖于候选的排列组合。

---

## 4. 实践练习

### 思考题

1. **如果允许候选相互 Attend，会有什么好处？会有什么坏处？**

2. **Candidate Isolation 对模型训练有什么影响？训练会更难还是更容易？**

3. **为什么 User 和 History 之间可以相互 Attend，但候选不能 Attend 候选？**

### 代码阅读

打开 `phoenix/grok.py`，找到：

1. `create_attention_mask` 或类似函数的实现
2. Attention Mask 如何被应用到 `scaled_dot_product_attention`

### 可视化练习

**手绘 Attention Mask**：

假设配置：
- User: 1 个位置
- History: 4 个位置
- Candidates: 3 个位置

画出 8x8 的 Attention Mask 矩阵。

---

## 小结

本节我们深入理解了 Candidate Isolation 的设计：

1. **问题**：标准 Attention 让候选相互影响，导致分数不稳定
2. **解决方案**：Candidate Isolation——候选只能 Attend User + History + 自己
3. **好处**：
   - 分数一致性：同一候选在不同批次分数相同
   - 可缓存性：User/History 表示可以缓存复用
   - 批处理一致性：训练更稳定
4. **实现**：通过特殊的 Attention Mask 矩阵

这是 Phoenix 排序模型的**核心创新点**，使得模型既保持了 Transformer 的表达能力，又获得了工程上的可扩展性。

下一节，我们将总结整个系统的设计模式。
