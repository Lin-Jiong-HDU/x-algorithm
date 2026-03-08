# 第 12 节：Ranking - Transformer 排序模型

> **学习目标**：深入理解排序模型的输入输出设计，掌握 Embedding 组合方式

---

## 1. 概念讲解

### 1.1 Ranking 模型的输入

排序模型将用户上下文和候选帖子一起输入 Transformer：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ranking Model Input                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   位置:    [0]     [1 ... S]         [S+1 ... S+C]                 │
│           ┌───┐   ┌─────────┐       ┌─────────────┐                │
│           │ U │   │ History │       │ Candidates  │                │
│           │   │   │ S 个位置 │       │  C 个位置   │                │
│           └───┘   └─────────┘       └─────────────┘                │
│              │          │                   │                       │
│              │          │                   │                       │
│   包含:   User      历史行为            候选帖子                    │
│           Hashes    Post+Author        Post+Author                  │
│                     +Action            +ProductSurface              │
│                                                                     │
│   维度:  [B, 1, D]  [B, S, D]        [B, C, D]                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 输出设计

模型预测**每个候选**的**多种行为概率**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ranking Model Output                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   logits: [B, num_candidates, num_actions]                         │
│                                                                     │
│   示例（B=1, C=3, A=5）：                                           │
│                                                                     │
│              like    reply   repost  click   dwell                  │
│           ┌────────────────────────────────────────┐               │
│   候选 0  │ 0.12   0.03    0.01    0.45    0.30   │               │
│   候选 1  │ 0.08   0.01    0.00    0.32    0.22   │               │
│   候选 2  │ 0.25   0.05    0.03    0.60    0.45   │               │
│           └────────────────────────────────────────┘               │
│                                                                     │
│   每个值是 P(action | user, candidate)                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 代码分析

### 2.1 输入构建：build_inputs

```python
def build_inputs(self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings)
    -> Tuple[jax.Array, jax.Array, int]:
{
    // 1. 用户 Embedding 组合
    user_embeddings, user_padding_mask = block_user_reduce(
        batch.user_hashes,
        recsys_embeddings.user_embeddings,
        hash_config.num_user_hashes,
        config.emb_size,
    );
    // 输出: [B, 1, D]

    // 2. 历史序列 Embedding 组合
    history_embeddings, history_padding_mask = block_history_reduce(
        batch.history_post_hashes,
        recsys_embeddings.history_post_embeddings,
        recsys_embeddings.history_author_embeddings,
        history_product_surface_embeddings,
        history_actions_embeddings,
        hash_config.num_item_hashes,
        hash_config.num_author_hashes,
    );
    // 输出: [B, S, D]

    // 3. 候选 Embedding 组合
    candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
        batch.candidate_post_hashes,
        recsys_embeddings.candidate_post_embeddings,
        recsys_embeddings.candidate_author_embeddings,
        candidate_product_surface_embeddings,
        hash_config.num_item_hashes,
        hash_config.num_author_hashes,
    );
    // 输出: [B, C, D]

    // 4. 拼接
    embeddings = jnp.concatenate([user_embeddings, history_embeddings, candidate_embeddings], axis=1);
    padding_mask = jnp.concatenate([user_padding_mask, history_padding_mask, candidate_padding_mask], axis=1);

    // 5. 计算候选起始位置
    candidate_start_offset = user_padding_mask.shape[1] + history_padding_mask.shape[1];

    return embeddings, padding_mask, candidate_start_offset;
}
```

### 2.2 Embedding 组合函数

#### block_user_reduce

```python
def block_user_reduce(user_hashes, user_embeddings, num_user_hashes, emb_size):
    # user_hashes: [B, num_user_hashes]
    # user_embeddings: [B, num_user_hashes, D]

    B = user_embeddings.shape[0]
    D = emb_size

    # 1. 展平多个哈希的 Embedding
    user_embedding = user_embeddings.reshape((B, 1, num_user_hashes * D))

    # 2. 投影回 D 维
    proj_mat = hk.get_parameter("proj_mat_1", [num_user_hashes * D, D])
    user_embedding = jnp.dot(user_embedding, proj_mat)

    # 3. 生成 padding mask（hash 0 表示无效）
    user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1)

    return user_embedding, user_padding_mask
```

#### block_history_reduce

```python
def block_history_reduce(
    history_post_hashes,
    history_post_embeddings,
    history_author_embeddings,
    history_product_surface_embeddings,
    history_actions_embeddings,
    num_item_hashes,
    num_author_hashes,
):
    B, S, _, D = history_post_embeddings.shape

    # 1. 展平多个哈希
    post_emb = history_post_embeddings.reshape((B, S, num_item_hashes * D))
    author_emb = history_author_embeddings.reshape((B, S, num_author_hashes * D))

    # 2. 拼接所有特征
    combined = jnp.concatenate([
        post_emb,
        author_emb,
        history_actions_embeddings,      # 行为类型
        history_product_surface_embeddings,  # 产品表面
    ], axis=-1)

    # 3. 投影回 D 维
    proj_mat = hk.get_parameter("proj_mat_3", [combined.shape[-1], D])
    history_embedding = jnp.dot(combined, proj_mat)

    # 4. padding mask
    history_padding_mask = (history_post_hashes[:, :, 0] != 0)

    return history_embedding, history_padding_mask
```

### 2.3 前向传播

```python
def __call__(self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings) -> RecsysModelOutput:
    // 1. 构建输入
    embeddings, padding_mask, candidate_start_offset = self.build_inputs(batch, recsys_embeddings);

    // 2. Transformer 编码
    model_output = self.model(
        embeddings,
        padding_mask,
        candidate_start_offset=candidate_start_offset,  // 关键！
    );

    // 3. Layer Norm
    out_embeddings = layer_norm(model_output.embeddings);

    // 4. 提取候选位置的输出
    candidate_embeddings = out_embeddings[:, candidate_start_offset:, :];

    // 5. Unembedding：投影到行为空间
    unembeddings = hk.get_parameter("unembeddings", [config.emb_size, config.num_actions]);
    logits = jnp.dot(candidate_embeddings, unembeddings);

    return RecsysModelOutput(logits=logits);
```

### 2.4 行为 Embedding

```python
def _get_action_embeddings(self, actions):
    # actions: [B, S, num_actions] - multi-hot 向量
    # 例如：[1, 0, 1, 0] 表示同时有 action 0 和 action 2

    D = config.emb_size
    num_actions = actions.shape[-1]

    # 1. 学习行为投影矩阵
    action_projection = hk.get_parameter("action_projection", [num_actions, D])

    # 2. 转换为有符号向量（0→-1, 1→+1）
    actions_signed = (2 * actions - 1)  # [-1, +1]

    # 3. 投影
    action_emb = jnp.dot(actions_signed, action_projection)

    # 4. 对无效行为置零
    valid_mask = jnp.any(actions, axis=-1, keepdims=True)
    action_emb = action_emb * valid_mask

    return action_emb
```

---

## 3. 实践练习

### 思考题

1. **为什么要用 Layer Norm 而不是 Batch Norm？**

2. **`candidate_start_offset` 参数的作用是什么？如果传 None 会怎样？**

3. **为什么行为用 multi-hot 而不是 single-hot？**

### 代码阅读

打开 `phoenix/recsys_model.py`，找到：

1. `PhoenixModelConfig` 的定义
2. `block_candidate_reduce` 函数（与 `block_history_reduce` 对比）

### 代码挑战

**分析 Embedding 维度变化**：

```python
# 假设配置
emb_size = 128
num_item_hashes = 2
num_author_hashes = 2
history_seq_len = 128
candidate_seq_len = 32
num_actions = 14

# 计算：
# 1. block_user_reduce 输入维度
# 2. block_history_reduce 拼接后维度（投影前）
# 3. build_inputs 最终序列长度
```

---

## 小结

本节我们理解了 Ranking Transformer 的设计：

1. **输入**：User + History + Candidates 拼接
2. **Embedding 组合**：多哈希展平 → 投影 → 拼接 → 再投影
3. **行为 Embedding**：multi-hot → 有符号向量 → 投影
4. **输出**：每个候选的多种行为概率
5. **关键参数**：`candidate_start_offset` 用于 Candidate Isolation

下一节，我们将深入 Attention Mask 和 Candidate Isolation 的设计。
