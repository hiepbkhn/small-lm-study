# Spark-X2.5-4B — KV Cache Analysis

## Hybrid attention reduces effective KV cost

Spark-X2.5-4B uses a **3:1 sliding-window-to-full-attention** layer ratio
(27 sliding, 9 full out of 36 layers). This is the key KV-cache efficiency
feature.

### Relevant config values

| Key | Value |
|---|---|
| `num_hidden_layers` | 36 |
| `num_key_value_heads` | 4 |
| `head_dim` | 256 |
| `sliding_window` | 512 |
| `max_position_embeddings` | 1,048,576 (1M) |
| `torch_dtype` | bfloat16 (2 bytes) |
| full-attention layers | 9 (indices 3,7,11,15,19,23,27,31,35) |
| sliding-attention layers | 27 |

### Per-token KV footprint

Per token, per sequence, K+V combined per layer:

```
bytes/layer/token = 2 (bf16) × 2 (K+V) × num_kv_heads × head_dim
                  = 2 × 2 × 4 × 256
                  = 4,096 bytes = 4 KB
```

| Layer type | Number | Retained context | Cache per token |
|---|---|---|---|
| Full attention | 9 | entire seq | 4 KB × S |
| Sliding attention | 27 | min(S, 512) | 4 KB × min(S,512) |

### Total KV cache vs. sequence length

For sequence length `S`:

```
bytes = (9 × S + 27 × min(S, 512)) × 4,096
```

For `S ≤ 512`: `(9S + 27S) × 4096 = 36S × 4096` — all layers at full cost.

For `S > 512` (the realistic regime):

```
bytes = (9S + 27×512) × 4096
      = (9S + 13,824) × 4096
      = 36,864·S + 56,623,104  bytes
```

### Comparison with a uniform-36-layer GQA model

A non-hybrid model with the same 36 layers / 4 KV heads / head_dim 256 would
cost `36 × 4096 × S = 147,456 × S` bytes (no sliding cap).

| Seq length | Uniform 36-layer | Spark-X2.5 hybrid | Savings |
|---|---|---|---|
| 8,192 | 1.21 GB | 0.34 GB | ~72% |
| 32,768 | 4.83 GB | 1.21 GB | ~75% |
| 131,072 | 19.3 GB | 4.84 GB | ~75% |
| 1,048,576 | 154.6 GB | 38.6 GB | ~75% |

At 1M context the hybrid saves **~116 GB** of KV cache vs. a uniform model.
This is what makes native 1M context feasible on on-device / single-GPU
hardware.

### KV cache @ 1M context (single sequence, bf16)

```
full:    9 × 1,048,576 × 4,096 = 38.65 GB
sliding: 27 × 512 × 4,096      = 0.057 GB
total:                          ≈ 38.7 GB
```

For comparison (same size class, other models in this study):

| Model | KV @ max ctx |
|---|---|
| Spark-X2.5-4B | 38.7 GB @ 1M (hybrid) |
| K2-Horizon-7B | 72 GB @ 512K (uniform) |
| MiniCPM5-2B | 5.25 GB @ 128K (uniform) |

### Notes

- Sliding-window layers use `create_sliding_window_causal_mask` with window
  512; KV for those layers is evicted after 512 tokens in the cache.
- The two RoPE configs (`full`: θ=5e6, prf=0.25; `sliding`: θ=1e4,
  prf=1.0) mean sliding layers get stronger local positional signal while
  full layers encode long-range position sparsely (only 25% of dims rotated).
- `head_dim=256` is large (vs typical 128), which raises per-layer cost,
  but GQA 4:1 keeps it modest; the sliding ratio is what dominates savings.
- `tie_word_embeddings=true` → no separate LM head weights.

## Summary

- The 3:1 sliding:full ratio cuts effective KV cache to ~25% of a uniform
  model at long context.
- Native 1M context enabled on-device (≈38.7 GB KV at 1M seq, bf16).
- GQA 4:1, head_dim 256, headwise sigmoid output gate.
