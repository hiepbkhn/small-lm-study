# K2-Horizon-7B — KV-Cache Analysis

## Why K2-Horizon-7B Is Not KV-Cache Efficient

### 1. 512K Full-Attention Context (No Sliding Window)

`sliding_window=null` and `use_sliding_window=false` — every token must attend to **every prior token** at every layer. There is no window limit, no sparse attention, no local attention to bound per-layer cache size. The cache grows linearly with context and the model is designed to use that full 512K window.

### 2. GQA But Not MQA/MLA

K2-Horizon uses GQA with **8 KV heads** (32 query heads / 4:1 ratio). GQA is a good choice, but:
- **MQA (1 KV head)** would reduce cache 8× (72 GB → 9 GB at 512K)
- **Multi-Latent Attention (MLA)** (DeepSeek-style low-rank KV compression) would reduce cache ~30–40×

GQA was likely chosen as a quality-preserving middle ground for a dense 7B model; MQA/MLA would be more aggressive but risk higher quality loss at the scale.

### 3. No KV Cache Quantization in the Architecture

The model has no built-in support for INT8/INT4/FP8 KV-cache quantization. The `DynamicCache` and `FlashAttention` backends assume the same dtype as the model (BF16). External quantized KV-cache schemes (e.g., `k-quant`/`v-quant` in vLLM) can reduce this, but they are post-hoc and not part of the model design.

### 4. No Attention Gating or Layer-Local Attention

`attention_gate_func=null` and `query_key_norm=false`. The architecture does not use a lightweight gating mechanism that could let early or late layers use shorter effective windows, which is another way to bound cache.

### 5. 36 Layers × Full-Attention

The 7B-class dense model has **36 decoder layers**, all with full-attention + dense MLP. Every layer contributes its own K and V tensors to the cache. Fewer layers (e.g., 24 or 28) or a deeper-but-narrower topology would reduce cache, but the design trade-off favors per-layer expressiveness.

---

## Detailed Memory Estimation

### Per-Token Cache Cost

| Component | Calculation | Bytes |
|-----------|-------------|-------|
| Per KV head | head_dim × 2 (K + V) | 128 × 2 = 256 B |
| Per layer (8 KV heads) | 8 × 256 × 2 (BF16) | 4,096 B |
| Per token (36 layers) | 36 × 4,096 | **144,000 B ≈ 144 KB** |

```
per_token = 36 layers × 8 kv_heads × 128 head_dim × 2 (K+V) × 2 bytes (BF16)
          = 147,456 bytes = 144 KB
```

### Cache Size vs. Context Length

| Context | Cache Size (BF16) | Notes |
|---------|-------------------|-------|
| 4,096 | **0.56 GB** | Typical short prompt |
| 8,192 | **1.12 GB** | |
| 32,768 | **4.50 GB** | |
| 64,512 | **9.00 GB** | |
| 131,072 | **18.00 GB** | Midtrain Stage 2 seq len |
| 262,144 | **36.00 GB** | |
| 524,288 | **72.00 GB** | Full 512K context |

**Formula:**
```
cache_bytes = L × H_kv × d_head × 2 × 2 × ctx
            = 36 × 8 × 128 × 4 × ctx
            = 147,456 × ctx
```

### Comparison with Model Weights

| Item | Size |
|------|------|
| Model weights (BF16, non-tied) | ~16.8 GB |
| KV cache @ 512K | **72.0 GB** |
| KV cache / weights ratio @ 512K | **4.3 : 1** |

At 512K context, the KV cache is **4.3× larger than the model itself**. Even at 32K context the cache (4.5 GB) is a meaningful fraction of the 16.8 GB weights.

### Total Memory for Serving (1 GPU, BF16, TP=1)

| Context | Weights | KV Cache | Total | Fits on |
|---------|---------|----------|-------|---------|
| 8K | 16.8 GB | 1.12 GB | ~18 GB | A100 40 GB (tight) |
| 32K | 16.8 GB | 4.50 GB | ~21 GB | A100 40 GB |
| 128K | 16.8 GB | 18.0 GB | ~35 GB | A100 40 GB (tight) |
| 256K | 16.8 GB | 36.0 GB | ~53 GB | H100 80 GB |
| 512K | 16.8 GB | 72.0 GB | ~89 GB | H100 80 GB (TP=2) or H200 |

> The validated serving recipe (SGLang: BF16, TP=1, FA3) targets **H100/H200** for full 512K context. On A100-40GB the full 512K window is infeasible without TP.

### Multi-Batch / Concurrent Sequences

Cache cost multiplies with batch size:

| Batch | Context | Total KV | GPU Needed |
|-------|---------|----------|------------|
| 1 | 32K | 4.5 GB | A100 40 GB |
| 4 | 32K | 18.0 GB | A100 40 GB |
| 8 | 32K | 36.0 GB | H100 80 GB |
| 16 | 32K | 72.0 GB | H100 80 GB (tight) |
| 1 | 512K | 72.0 GB | H100/H200 TP=1–2 |
| 4 | 512K | 288 GB | H200 × 2+ or FP8/INT8 KV quant |

### With KV Quantization (post-hoc, vLLM/SGLang)

| KV dtype | Cache @ 512K | Cache @ 32K |
|----------|-------------|-------------|
| BF16 (default) | 72.0 GB | 4.50 GB |
| FP8 (Q8/K8) | 36.0 GB | 2.25 GB |
| INT4 (Q4/K4) | 18.0 GB | 1.12 GB |

INT4 KV-cache quantization at 512K brings the cache to ~18 GB, roughly the same as BF16 at 128K — making 512K viable on a single A100 40GB.

---

## Summary of Design Trade-offs

| Aspect | K2-Horizon-7B | Most Efficient Alternative | Cache Ratio |
|--------|--------------|---------------------------|-------------|
| Attention type | GQA 8 kv-heads | MQA / MLA | 8× / ~30–40× smaller |
| Sliding window | none (full 512K) | 4K/8K window | bounds cache to ~5.75 MB/layer |
| KV quantization | none (BF16 only) | INT4 KV | 4× smaller |
| Layers | 36 | 24 | 1.5× smaller |
| head_dim | 128 | 64 | 2× smaller |

The 512K full-attention + GQA + dense-MLP combination is a deliberate quality-vs-capability choice: the model optimizes for maximum long-context reasoning quality at 7B scale, at the cost of a KV cache that scales linearly to **72 GB** at the full 512K window.
