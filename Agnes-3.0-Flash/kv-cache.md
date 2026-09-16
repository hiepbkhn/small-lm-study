# Agnes-3.0-Flash KV Cache Efficiency

## 1. Architecture Context

Agnes-3.0-Flash is a hybrid-attention decoder with 72 layers:

| Layer type | Count | Attention |
|---|---|---|
| Delta-rule recurrent | 54 | Per-layer fixed-size state, **no KV cache growth** |
| Global attention | 18 | Standard GQA, **KV cache grows with context** |

Only 18 of 72 layers (25%) hold a KV cache that scales with sequence length.

## 2. KV Cache Size Calculation

### 2.1 Per-token, per-layer KV cache

The 18 global-attention layers use GQA: 24 query heads, 4 KV heads, head dim 256.

Per token, per layer:
- K: 4 heads x 256 dims x 2 bytes (bf16) = 2,048 bytes
- V: 4 heads x 256 dims x 2 bytes = 2,048 bytes
- Total: **4,096 bytes/token/layer**

### 2.2 Total KV cache

| Context length | Per-token | 18 layers | Total |
|---|---|---|---|
| 8,192 | 4,096 B | 73,728 B/token | 576 MB |
| 32,768 | 4,096 B | 73,728 B/token | 2.25 GB |
| 131,072 | 4,096 B | 73,728 B/token | 9.00 GB |
| 262,144 | 4,096 B | 73,728 B/token | 18.00 GB |

At the full 262,144-token context, the KV cache alone is **~18 GB**, before any model weights (66 GB in bf16).

### 2.3 Delta-rule recurrent state

The 54 delta-rule layers use a fixed-size recurrent state in fp32, independent of sequence length:
- 16 key heads x 48 value heads, head dim 128
- State size: 16 x 48 x 128 x 4 bytes = **3.93 MB/layer**, x 54 layers = **~212 MB total**

This is constant regardless of context length, unlike the linearly-growing KV cache.

## 3. Efficiency Analysis

### 3.1 Cache Reduction vs. Full-Attention Baseline

A standard full-attention 72-layer transformer with the same GQA config (24Q/4KV, dim 256) would have:

- 72 layers x 4,096 B/token = 294,912 B/token
- At 262,144 tokens: 72 x 18.00 GB = **~36 GB total KV cache**

Agnes-3.0-Flash uses only 18 of 72 layers for global attention:

| | Full-attention (72 layers) | Agnes hybrid (18 global) | Reduction |
|---|---|---|---|
| KV cache @ 262K | ~36 GB | ~18 GB | **50%** |
| Cache growth | Linear in L | Linear in L, but 1/4 the slope | 4x smaller slope |

The delta-rule layers replace 75% of the attention layers with fixed-memory recurrent units, cutting the growing KV cache footprint by half while preserving all 72 layers of model capacity.

### 3.2 Effective Cache Per Byte of Model Capacity

The model has 33B parameters (66 GB bf16). At full context:

| Component | Size | % of total memory |
|---|---|---|
| Weights (bf16) | 66 GB | 78% |
| KV cache @ 262K | 18 GB | 21% |
| Delta-rule state (fp32) | 0.2 GB | <1% |
| **Total** | **~84 GB** | 100% |

The KV cache is the second-largest memory consumer. Because only 18 layers contribute to it, the cache is manageable even on a single H100 (80 GB) at shorter context lengths, though 262K tokens requires ~84 GB total, exceeding 80 GB.

### 3.3 Memory vs. Context Length

| Context | KV cache | Total (weights + cache + state) | GPU fit |
|---|---|---|---|
| 8,192 | 0.576 GB | ~67 GB | H100 80 GB OK |
| 32,768 | 2.25 GB | ~68.5 GB | H100 80 GB OK |
| 131,072 | 9.0 GB | ~75 GB | H100 80 GB tight |
| 262,144 | 18 GB | ~84 GB | H100 80 GB exceeds, H200 141 GB OK |

For 262K context, `--tp 2` (tensor parallel across 2 GPUs) or an H200 141 GB card is recommended.

### 3.4 Comparison with Other Long-Context Strategies

| Strategy | KV cache @ 262K | Notes |
|---|---|---|
| Full attention, 72 layers, GQA 24/4 | ~36 GB | 2x the Agnes cache |
| Sliding window (e.g. 4K window) | ~1 GB | Loses long-range dependency |
| Agnes hybrid (18 global + 54 delta) | ~18 GB | Retains full 72-layer depth |
| FlashAttention (algorithmic) | Same as above | Reduces compute, not memory |

The hybrid approach achieves long-context efficiency without sacrificing layer depth. The delta-rule layers are the key innovation: they encode sequence information into a fixed-size recurrent state, amortizing the memory cost across any context length.

## 4. Practical Implications

### 4.1 Deployment

- **Single GPU (H100 80 GB):** Supports ~131K context tokens with KV cache + weights
- **Single GPU (H200 141 GB):** Full 262K context fits comfortably
- **Tensor parallel `--tp 2`:** Splits both weights and KV cache, enabling 262K on 2x H100

### 4.2 Batch Serving

Since the delta-rule state is fixed-size, the per-request memory overhead beyond the first token is constant for the 54 recurrent layers. Only the 18 global layers grow per-token. This means:

- **Prefill:** Memory scales linearly with prompt length (KV cache for all 18 layers)
- **Decode:** Per-token cost is small: 18 layers x 4,096 B = 73.7 KB/token (plus the constant 212 MB delta state)

At 128 concurrent requests with 32K context each:
- 128 x 2.25 GB = 288 GB KV cache, requiring `--tp 2` or more

### 4.3 Why This Matters for 33B-Class Models

Typical 33B full-attention models at 262K context would need ~36 GB of KV cache. By using 18 global + 54 delta layers, Agnes halves this to ~18 GB while:

1. Preserving all 72 layers of representational depth
2. Keeping recurrent state at a fixed 212 MB regardless of sequence length
3. Enabling deployment on a single H200 with room for batching

The architectural choice trades some attention expressiveness (only 25% of layers see the full sequence via GQA) for a 50% reduction in growing memory, which is the dominant scaling factor for long-context serving.
