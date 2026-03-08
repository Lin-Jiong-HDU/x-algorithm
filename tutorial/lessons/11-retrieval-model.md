# 第 11 节：Retrieval - Two-Tower 棑索模型

> **学习目标**：理解大规模检索的 Two-Tower 架构，掌握 Embedding 设计和相似度检索

---

## 1. 概念讲解

### 1.1 Two-Tower 架构

Two-Tower 是推荐系统检索阶段的经典架构：

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Two-Tower Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                        USER TOWER                             │   │
│   │                                                              │   │
│   │   用户特征                用户行为序列                         │   │
│   │      ↓                        ↓                               │   │
│   │   User Embedding    +    History Embeddings                  │   │
│   │      └────────────────────┬────────────────────┘              │   │
│   │                           ↓                                   │   │
│   │                    Transformer                                │   │
│   │                           ↓                                   │   │
│   │                   Mean Pooling                               │   │
│   │                           ↓                                   │   │
│   │                  L2 Normalize                                │   │
│   │                           ↓                                   │   │
│   │              user_representation [B, D]                       │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                    │                                │
│                                    │  dot product                  │
│                                    ▼                                │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                     CANDIDATE TOWER                          │   │
│   │                                                              │   │
│   │   帖子 Embedding      +      作者 Embedding                  │   │
│   │           ↓                       ↓                          │   │
│   │           └───────────┬───────────┘                          │   │
│   │                       ↓                                      │   │
│   │                   MLP 投影                                   │   │
│   │                       ↓                                      │   │
│   │                  L2 Normalize                                │   │
│   │                       ↓                                      │   │
│   │           candidate_representation [N, D]                    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么叫 "Two-Tower"？

因为有两个独立的**编码器**（Tower）：
- **User Tower**：编码用户信息
- **Candidate Tower**：编码候选帖子

两个 Tower **独立运行**，最终通过**点积**计算相似度。

### 1.3 核心优势

1. **可分离计算**：
   - User Tower 只在用户请求时运行
   - Candidate Tower 可以**预先计算**并索引

2. **高效检索**：
   - 归一化后的点积 = 余弦相似度
   - 支持 ANN（近似最近邻）索引

3. **可扩展**：
   - 候选数量增加不影响 User Tower 复杂度

---

## 2. 代码分析

### 2.1 User Tower

**文件**：`phoenix/recsys_retrieval_model.py`

```python
def build_user_representation(self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings)
    -> Tuple[jax.Array, jax.Array]:
{
    config = self.config
    hash_config = config.hash_config

    // 1. 获取各种 Embedding
    history_product_surface_embeddings = self._single_hot_to_embeddings(
        batch.history_product_surface, vocab_size, emb_size, "product_surface_embedding_table"
    )
    history_actions_embeddings = self._get_action_embeddings(batch.history_actions)

    // 2. 组合 User Embedding
    user_embeddings, user_padding_mask = block_user_reduce(
        batch.user_hashes,
        recsys_embeddings.user_embeddings,
        hash_config.num_user_hashes,
        config.emb_size,
    )

    // 3. 组合 History Embedding
    history_embeddings, history_padding_mask = block_history_reduce(
        batch.history_post_hashes,
        recsys_embeddings.history_post_embeddings,
        recsys_embeddings.history_author_embeddings,
        history_product_surface_embeddings,
        history_actions_embeddings,
        ...
    )

    // 4. 拼接 User + History
    embeddings = jnp.concatenate([user_embeddings, history_embeddings], axis=1)
    padding_mask = jnp.concatenate([user_padding_mask, history_padding_mask], axis=1)

    // 5. Transformer 编码
    model_output = self.model(
        embeddings.astype(self.fprop_dtype),
        padding_mask,
        candidate_start_offset=None,  // 没有 candidate，只有 user + history
    )

    // 6. Mean Pooling
    user_outputs = model_output.embeddings
    mask_float = padding_mask.astype(jnp.float32)[:, :, None]
    user_embeddings_masked = user_outputs * mask_float
    user_embedding_sum = jnp.sum(user_embeddings_masked, axis=1)  // [B, D]
    mask_sum = jnp.sum(mask_float, axis=1)  // [B, 1]
    user_representation = user_embedding_sum / jnp.maximum(mask_sum, 1.0)

    // 7. L2 归一化
    user_norm_sq = jnp.sum(user_representation**2, axis=-1, keepdims=True)
    user_norm = jnp.sqrt(jnp.maximum(user_norm_sq, EPS))
    user_representation = user_representation / user_norm

    return user_representation, user_norm
}
```

**关键点**：
- 使用 Transformer 编码用户和历史行为
- Mean Pooling 聚合成单一向量
- L2 归一化确保点积 = 余弦相似度

### 2.2 Candidate Tower

```python
@dataclass
class CandidateTower(hk.Module):
    emb_size: int

    def __call__(self, post_author_embedding: jax.Array) -> jax.Array:
        // 1. 展平多哈希
        // [B, C, num_hashes, D] -> [B, C, num_hashes * D]
        post_author_embedding = jnp.reshape(post_author_embedding, (B, C, -1))

        // 2. MLP 投影
        proj_1 = hk.get_parameter("candidate_tower_projection_1", [input_dim, emb_size * 2])
        proj_2 = hk.get_parameter("candidate_tower_projection_2", [emb_size * 2, emb_size])

        hidden = jnp.dot(post_author_embedding, proj_1)
        hidden = jax.nn.silu(hidden)  // 激活函数
        candidate_embeddings = jnp.dot(hidden, proj_2)

        // 3. L2 归一化
        candidate_norm_sq = jnp.sum(candidate_embeddings**2, axis=-1, keepdims=True)
        candidate_norm = jnp.sqrt(jnp.maximum(candidate_norm_sq, EPS))
        candidate_representation = candidate_embeddings / candidate_norm

        return candidate_representation
```

**为什么 Candidate Tower 更简单？**

- User Tower 需要 Transformer 理解**用户兴趣模式**
- Candidate Tower 只需要把帖子特征**投影到共享空间**

### 2.3 Top-K 检索

```python
def _retrieve_top_k(self, user_representation, corpus_embeddings, top_k, corpus_mask):
    // 1. 计算相似度矩阵
    // user_representation: [B, D]
    // corpus_embeddings: [N, D]
    // scores: [B, N]
    scores = jnp.matmul(user_representation, corpus_embeddings.T)

    // 2. 应用 mask（过滤无效候选）
    if corpus_mask is not None:
        scores = jnp.where(corpus_mask[None, :], scores, -INF)

    // 3. 取 Top-K
    top_k_scores, top_k_indices = jax.lax.top_k(scores, top_k)

    return top_k_indices, top_k_scores
```

**计算复杂度**：
- 点积：O(B × N × D)
- Top-K：O(B × N × log K)

对于百万级候选（N = 1,000,000），这仍然很慢。**生产环境使用 ANN 索引**。

### 2.4 Hash-Based Embeddings

```python
@dataclass
class HashConfig:
    num_user_hashes: int = 2   // 用户哈希数量
    num_item_hashes: int = 2   // 帖子哈希数量
    num_author_hashes: int = 2 // 作者哈希数量
```

**为什么用多个哈希？**

```python
// 单哈希：hash(user_id) -> embedding_index
// 多哈希：
embedding = concat([
    embedding_table_1[hash(user_id, seed_1)],
    embedding_table_2[hash(user_id, seed_2)],
])
```

**优势**：
- 减少哈希冲突
- 更丰富的表示
- 参数效率更高

---

## 3. 实践练习

### 思考题

1. **为什么 User Tower 和 Candidate Tower 使用不同的架构？**

2. **为什么要 L2 归一化？不归一化会怎样？**

3. **如果候选数量是 1000 万，`_retrieve_top_k` 会很慢。生产环境如何解决？**

### 代码阅读

1. 打开 `phoenix/recsys_model.py`，找到 `block_user_reduce` 和 `block_history_reduce` 函数
2. 理解它们如何组合多个哈希的 Embedding

### 代码挑战

**实现一个简单的 Two-Tower 模型**：

```python
import jax.numpy as jnp
import haiku as hk

class SimpleUserTower(hk.Module):
    def __call__(self, user_features):
        # TODO: 实现
        # 输入: [B, feature_dim]
        # 输出: [B, emb_dim] (L2 normalized)
        pass

class SimpleCandidateTower(hk.Module):
    def __call__(self, candidate_features):
        # TODO: 实现
        pass

def retrieve(user_repr, candidate_repr, top_k):
    # TODO: 实现点积 + Top-K
    pass
```

---

## 小结

本节我们理解了 Two-Tower 检索架构：

1. **Two-Tower 设计**：User Tower + Candidate Tower 独立编码
2. **User Tower**：Transformer + Mean Pooling + L2 Normalize
3. **Candidate Tower**：MLP 投影 + L2 Normalize
4. **相似度检索**：点积（归一化后 = 余弦相似度）
5. **Hash-Based Embeddings**：多哈希减少冲突

下一节，我们将深入 Ranking Transformer 模型。
