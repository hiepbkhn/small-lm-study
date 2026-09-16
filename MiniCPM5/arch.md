# MiniCPM5-2B Architecture

## Overview

MiniCPM5-2B is a **2B-class dense** decoder-only LLM from the MiniCPM5 series (OpenBMB), designed for on-device, local deployment, and resource-constrained scenarios. It uses the **standard `LlamaForCausalLM`** architecture — no custom code, no custom kernels.

- **Total params:** 2,516,756,480 (~2.5B, including embeddings)
- **Non-embedding params:** 1,981,982,720 (~1.98B)
- **Architecture:** Standard Llama (`LlamaForCausalLM`)
- **Context:** 131,072 tokens (128K)
- **Dtype:** BF16
- **License:** Apache 2.0
- **Inference:** day-0 support in vLLM, SGLang, llama.cpp, Ollama, LM Studio, MLX, LiteRT-LM

## Configuration

| Parameter | Value |
|-----------|-------|
| `model_type` | `llama` |
| `architectures` | `LlamaForCausalLM` |
| `hidden_size` | 2048 |
| `intermediate_size` | 6144 |
| `num_hidden_layers` | 42 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 (GQA, ratio 8:1) |
| `head_dim` | 128 |
| `vocab_size` | 130,560 |
| `max_position_embeddings` | 131,072 |
| `hidden_act` | silu |
| `rms_norm_eps` | 1e-06 |
| `attention_bias` | false (implicit; no `attention_bias` key) |
| `rope_theta` | 5,000,000 |
| `rope_scaling` | null |
| `tie_word_embeddings` | false |
| `use_cache` | true |
| `bos_token_id` | 0 |
| `eos_token_id` | [1, 130073] |
| `pad_token_id` | 1 |
| `torch_dtype` | bfloat16 |

## Model Components

### `LlamaModel` (backbone)

```
embed_tokens: Embedding(vocab=130560, hidden=2048)
layers:        42 x LlamaDecoderLayer
norm:          LlamaRMSNorm(hidden=2048, eps=1e-6)
rotary_emb:    LlamaRotaryEmbedding(rope_theta=5e6, head_dim=128)
```

### `LlamaDecoderLayer`

Standard pre-norm residual Transformer block:

```
residual = hidden
hidden = input_layernorm(hidden)
hidden = self_attn(hidden) + residual
residual = hidden
hidden = post_attention_layernorm(hidden)
hidden = mlp(hidden) + residual
```

- `input_layernorm` / `post_attention_layernorm`: `LlamaRMSNorm(hidden=2048, eps=1e-6)`
- `self_attn`: `LlamaAttention` (GQA, standard)
- `mlp`: `LlamaMLP` (SwiGLU)

### `LlamaAttention`

Grouped-Query Attention with RoPE (standard Llama 3 style):

- `q_proj`: Linear(2048 → 16×128 = 2048), no bias
- `k_proj`: Linear(2048 → 2×128 = 256), no bias
- `v_proj`: Linear(2048 → 2×128 = 256), no bias
- `o_proj`: Linear(2048 → 2048), no bias
- GQA ratio: 16 q-heads / 2 kv-heads = **8:1**
- RoPE: `rope_theta=5e6`, no scaling, no sliding window
- No query/key norm (standard Llama)

### `LlamaMLP`

SwiGLU-style dense MLP (no bias):

- `gate_proj`: Linear(2048 → 6144)
- `up_proj`:   Linear(2048 → 6144)
- `down_proj`: Linear(6144 → 2048)
- `act_fn`: silu
- Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`

### `LlamaRMSNorm`

Standard RMSNorm:

- `hidden_size=2048`, `eps=1e-6`
- No grouping (unlike K2-Horizon's grouped RMSNorm)

### `LlamaRotaryEmbedding`

Standard RoPE:

- `rope_theta=5,000,000` (lower than K2-Horizon's 10M, typical for 2B-class)
- `head_dim=128`
- No sliding window, no RoPE scaling

## Weight Estimate

| Component | Calculation | Size |
|-----------|-------------|------|
| Embedding | 130,560 × 2048 × 2 B | 510 MB |
| LM Head | 130,560 × 2048 × 2 B | 510 MB |
| Per layer (q+k+v+o proj + MLP) | ~360 MB / 4 | ~90 MB |
| 42 layers | 42 × 90 MB | 3,780 MB |
| **Total (BF16)** | | **~4.7 GB** |

## KV-Cache Notes

Per token (all 42 layers, BF16):
```
42 layers × 2 kv_heads × 128 head_dim × 2 (K+V) × 2 bytes = 42.0 KB
```

| Context | KV Cache |
|---------|----------|
| 4K | 168 MB |
| 8K | 336 MB |
| 32K | 1.31 GB |
| 64K | 2.62 GB |
| 128K | 5.25 GB |

The 8:1 GQA ratio (2 KV heads) makes KV cache significantly more compact than K2-Horizon-7B's 4:1 ratio — at 128K context, MiniCPM5-2B's cache (~5.25 GB) is roughly 1/14th of K2-Horizon-7B at 512K (72 GB).

## Training Recipe

Three stages: **base → mid → post-training** (SFT + RL + OPD).

| Stage | Details |
|-------|---------|
| Base training | Stable + decay training, ~22T-class corpus |
| Mid-training | Target capability strengthening, data-distribution adaptation |
| SFT | 400B tokens deep-thinking SFT (UltraData-SFT-2605) |
| RL | Specialized teachers: math, code, agentic, writing; JustRL II critic-based |
| OPD | On-Policy Distillation: 16 expert models (incl. 5 agentic) merged via full-vocab reverse-KL |

**Open data released:** Ultra-FineWeb, Ultra-FineWeb-L3, UltraX, UltraData-Code, UltraData-Math, UltraData-SFT-2605, UltraData-SFT-Agent-2609, UltraData-RL-2609

## Inference Notes

- **Sampling:** `temperature=1.0`, `top_p=0.95`, `min_p=0.0` (critical: llama.cpp's default `min_p=0.05` causes repetition)
- **Thinking mode:** `enable_thinking=True` in chat template
- **Tool calls:** XML-style, SGLang's `minicpm5` parser recommended
- **Speculative decoding:** DSpark draft model available (`MiniCPM5-2B-DSpark`, block size 7)

## Model Variants

| Variant | Purpose |
|---------|---------|
| `MiniCPM5-2B` | Final release (BF16, post-trained with RL + OPD) |
| `MiniCPM5-2B-SFT` | SFT-only checkpoint (before RL/OPD) |
| `MiniCPM5-2B-Midtrain` | Mid-training checkpoint (before SFT) |
| `MiniCPM5-2B-Base` | Base pre-training only |
| `MiniCPM5-2B-GGUF` | llama.cpp / Ollama / LM Studio |
| `MiniCPM5-2B-MLX` | Apple Silicon (MLX / 4-bit) |
| `MiniCPM5-2B-GPTQ` | GPTQ / 4-bit quantized |
| `MiniCPM5-2B-DSpark` | Draft model for speculative decoding |
| `MiniCPM5-2B-LiteRT` | On-device (Android/iOS/IoT) |

## References

- Model card: https://huggingface.co/openbmb/MiniCPM5-2B
- Tech report: https://arxiv.org/abs/2506.07900
- GitHub: https://github.com/OpenBMB/MiniCPM
- UltraData: https://ultradata.openbmb.cn/
