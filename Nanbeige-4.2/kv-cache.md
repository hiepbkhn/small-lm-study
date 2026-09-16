# Nanbeige4.2-3B — KV Cache Analysis

## Overview

Nanbeige4.2-3B uses a **looped Transformer** design (`num_loops=2`, 22 physical
layers). The central KV-cache question is: *does the loop multiply the cache?*
It does — the model caches KV per **(layer, loop)** slot, so the effective
cache depth is `num_hidden_layers × num_loops = 44` slots. This is the main
cost of the loop design.

## Relevant config values

| Key | Value |
|---|---|
| `num_hidden_layers` | 22 (physical layers) |
| `num_loops` | 2 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `max_position_embeddings` | 262,144 |
| `torch_dtype` | bfloat16 (2 bytes) |
| `loop_share_kv` | false (not enabled) |

## Effective cache depth

Cache layer index is computed as:

```python
# modeling_nanbeige.py:143
def _get_loop_cache_layer_idx(layer_idx, loop_idx, num_hidden_layers, cache_layer_idx=None):
    if cache_layer_idx is not None:
        return cache_layer_idx
    return layer_idx + loop_idx * num_hidden_layers
```

So for `num_loops=2`, physical layer `L` (0..21) writes to slots:
- loop 0 → slot `L`
- loop 1 → slot `L + 22`

Total slots = 44. Note the cache slot layout is *per loop, full 22-layer
span*, not a single shared slot per physical layer.

## Memory formula

Per token, per sequence, for K and V combined:

```
bytes = 2 (bf16) × 2 (K+V) × num_kv_heads × head_dim × effective_depth
      = 2 × 2 × 8 × 128 × 44
      = 184,320 bytes ≈ 180 KB  / token / seq
```

### Cache size at common lengths (single sequence)

| Sequence length | KV cache (≈) |
|---|---|
| 4,096 | 735 MB |
| 8,192 | 1.47 GB |
| 32,768 | 5.86 GB |
| 131,072 | 23.4 GB |
| 262,144 (max) | 46.9 GB |

(Compute: 184,320 × S bytes; at 262,144 → 46.9 GB before adding model
weights ~6.5 GB bf16.)

This is roughly **2× a non-looped model** with 22 layers and the same GQA
config. For comparison, a standard 22-layer GQA (8 KV heads, 128 dim)
model costs ~92,160 B/token (90 KB/token), i.e. half.

## Loop-sharing KV (`loop_share_kv`)

`config.json` has `loop_share_kv` unset → **disabled** for this checkpoint.
When enabled (LoopSplit path only), a middle block's first-pass K/V is reused
on repeat executions instead of re-deriving, halving the repeated layers'
cache:

```python
# modeling_nanbeige.py:258
def _apply_loop_shared_kv(...):
    if mhc_loop_idx == 0:
        loop_share_kv_cache[layer_idx] = (key_states, value_states)
        return key_states, value_states
    return loop_share_kv_cache[layer_idx]   # reuse
```

With `loop_share_kv=True`, only the first execution of shared layers stores
KV; repeats read the cache. This would bring cache cost back toward the
single-pass 22-layer figure. It requires `enable_double_loop_split=True`
and is incompatible with `enable_depth_attention`.

## Depth attention cache

When `enable_depth_attention=True`, KV from anchor layers is additionally
kept in a small `depth_attention_kv_cache` list (every
`depth_attention_stride` layers) for value-mixing before token-level
attention. Anchor-only mode (`depth_attention_recent_window=0`) and static
anchors (`depth_attention_static_anchor_once=True`) keep this footprint
bounded (O(number of anchors), not O(seq_len)).

## N-gram cache

If N-gram embeddings are enabled, an `NgramCache` extends `DynamicCache`
with a rolling `ngram_context` of `emb_neighbor_num - 1` tokens. Negligible
footprint vs KV.

## Constraints on caching

- `StaticCache` is **not supported** with loop-aware caching
  (`num_loops > 1` or double-loop split). Use `DynamicCache`.
- `loop_share_kv` does not support gradient checkpointing during training.
- `enable_depth_attention` + `loop_share_kv` does not support
  `use_cache=True` / generation.
- vLLM expands `config.num_hidden_layers` to `physical × num_loops` to size
  KV slots (`_prepare_config_for_vllm`), pinning `num_physical_layers` so
  the ModuleList length stays stable.

## Summary

- Effective KV cache is ~180 KB/token/seq (46.9 GB at 256K single seq) —
  ~2× a non-looped equivalent due to `num_loops=2`.
- The loop *adds capacity at no extra weight cost*, but *doubles the KV
  cache*; `loop_share_kv` (when enabled) halves that on shared layers.
- GQA (8/48 heads, head_dim 128) keeps per-layer KV small; the loop is the
  dominant multiplier.
- Static cache unsupported; DynamicCache required.
