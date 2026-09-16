# K2-Horizon-7B Architecture

## Overview

K2-Horizon-7B is a **7B-class dense** decoder-only transformer from the K2-Horizon family (Institute of Foundation Models), with a **512K native context window**. It belongs to the K2-Horizon fleet of six models (0.9B, 3.7B, 7B, 32B, 36B-A4B, 375B-A23B) and is the dense 7B member.

- **Total params:** 9B (includes tied/non-tied embeddings; `tie_word_embeddings=false`)
- **Core architecture:** dense 7B decoder-only
- **Context:** 524,288 tokens (512K)
- **Dtype:** BF16
- **Tokenizer:** tokenizers backend (BPE-style)
- **Base:** Apache 2.0 open weights
- **Model type (HF):** `k2_horizon`
- **HF class:** `K2HorizonForCausalLM` (custom code, `trust_remote_code=True`)
- **Inference:** day-0 support in vLLM, SGLang, Ollama, llama.cpp, LM Studio

## Configuration

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 4096 |
| `intermediate_size` | 12288 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA, ratio 4:1) |
| `head_dim` | 128 |
| `vocab_size` | 250,624 |
| `max_position_embeddings` | 524,288 |
| `hidden_act` | silu |
| `rms_norm_eps` | 1e-06 |
| `layernorm_num_groups` | 4 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `rope_theta` | 10,000,000.0 |
| `rope_type` | default |
| `query_key_norm` | false |
| `tie_word_embeddings` | false |
| `use_cache` | true |
| `sliding_window` | null |
| `num_experts` | 0 (dense; MoE disabled) |
| `num_experts_per_tok` | 0 |
| `mova_num_experts` | 0 (MoVA disabled) |
| `mlp_only_layers` | [0..35] (all 36 layers are dense MLP) |
| `moe_gate_bias` | false |
| `moe_intermediate_size` | 0 |
| `router_score_func` | sigmoid |
| `norm_topk_prob` | true |
| `router_scaling_factor` | 1.0 |
| `router_aux_loss_coef` | 0.001 |
| `bos_token_id` | 0 |
| `eos_token_id` | 1 |
| `attention_gate_func` | null |
| `rope_head_dim` | 128 |
| `num_shared_experts` | 0 |
| `decoder_sparse_step` | 1 |

## Model Components

### `K2HorizonModel` (backbone)

```
embed_tokens: Embedding(vocab=250624, hidden=4096)
layers:        36 x K2HorizonDecoderLayer
norm:          K2HorizonRMSNorm(hidden=4096, groups=4, eps=1e-6)
rotary_emb:    K2HorizonRotaryEmbedding(rope_theta=1e7, head_dim=128)
```

### `K2HorizonDecoderLayer`

Each layer follows a **pre-norm, residual** Transformer block:

```
residual = hidden
hidden = input_layernorm(hidden)
hidden = self_attn(hidden) + residual
residual = hidden
hidden = post_attention_layernorm(hidden)
hidden = mlp(hidden) + residual
```

- `input_layernorm` / `post_attention_layernorm`: `K2HorizonRMSNorm(hidden=4096, groups=4)`
- `self_attn`: `K2HorizonAttention` (standard GQA) — MoVA attention is disabled since `mova_num_experts=0`
- `mlp`: `K2HorizonMLP` (dense) — MoE is disabled since all 36 layers are in `mlp_only_layers`

### `K2HorizonAttention`

Grouped-Query Attention (GQA) with RoPE:

- `q_proj`: Linear(4096 → 32×128 = 4096), no bias
- `k_proj`: Linear(4096 → 8×128 = 1024), no bias
- `v_proj`: Linear(4096 → 8×128 = 1024), no bias
- `o_proj`: Linear(32×128 = 4096 → 4096), no bias
- RoPE: `apply_rotary_pos_emb` on q/k with `cos`/`sin` from `K2HorizonRotaryEmbedding`
- Optional `gate_proj` (disabled; `attention_gate_func=null`)
- Optional `q_norm`/`k_norm` (disabled; `query_key_norm=false`)
- No sliding window

### `K2HorizonMLP`

SwiGLU-style dense MLP (no bias):

- `gate_proj`: Linear(4096 → 12288)
- `up_proj`:   Linear(4096 → 12288)
- `down_proj`: Linear(12288 → 4096)
- `act_fn`: silu
- Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`

### `K2HorizonRMSNorm`

Grouped RMSNorm with `n_groups=4`:

- Reshapes hidden to `(..., n_groups, hidden/n_groups)`
- Computes per-group variance, normalizes, then scales by learnable `weight`
- Casts to FP32 for stability, returns original dtype

### `K2HorizonRotaryEmbedding`

Standard RoPE:

- `rope_theta=1e7`, `rope_type=default`
- Inverse frequencies computed over `rope_head_dim=128`
- Supports dynamic RoPE updates via `@dynamic_rope_update`
- `cos`/`sin` computed in FP32, cast back to input dtype

### `K2HorizonSparseMoeBlock` (disabled in this model)

Mixture-of-Experts FFN block. Present in the codebase but not instantiated for the 7B model (`num_experts=0`). Uses a router with configurable `score_func` (`softmax`/`sigmoid`), top-k expert selection, optional shared experts, and load-balancing auxiliary loss.

### `K2HorizonMoVAAttention` (disabled in this model)

Mixture-of-Value Attention. Routed value experts in attention (`mova_num_experts`). Not instantiated for the 7B model. Used by the 36B-A4B MoVA variant.

## Forward Pass (inference)

```
input_ids → embed_tokens → [36 x DecoderLayer] → final RMSNorm → lm_head → logits
```

- Causal mask via `create_causal_mask` (full-attention, no sliding window)
- KV cache supported (`use_cache=true`, `DynamicCache`)
- Attention backends: eager, FlashAttention (FA1/FA2/FA3), SDPA, FlexAttention

## Training Stages

| Stage | Steps | Tokens | Seq Len |
|-------|-------|--------|---------|
| Pretraining | 1,100,000 | 22.9T | 8K |
| Midtrain Stage 1 | 55,000 | 1.1T | 32K |
| Midtrain Stage 2 | 25,000 | 498B | 128K |
| Midtrain Stage 3 | 5,500 | 110B | 512K |
| Midtrain Stage 4 | 10,000 | 199B | 512K |
| RL (Math/Code/Search/Tool) | 2,399 / 601 / 1,499 / 59 / 39 | 29.1B / 6.2B / 12.3B / 8.4B / 1.4B | 64K / — |
| RL Merge (ISO+RAM) | — | — | — |
| SFT Phase 1 | 10,000 | 199B | 512K |
| SFT Phase 2 | 2,500 | 50B | 512K |

**Checkpoint branching:**
- Pretrain: `pretrain_*` → `pretrain_1100000`
- Midtrain: `mid_1_*`…`mid_4_*` → `mid_4_10000`
- RL experts: `rl_math`, `rl_code1`, `rl_code2`, `rl_search`, `rl_tool_use` → `rl_merged`
- SFT: `sft_1_*` → `sft_1_11000`, `sft_2_*` → `sft_2_2500`

## Inference Notes

- **Reasoning:** `reasoning_effort="high"` recommended; thinking in `reasoning_content`, answer in `content`
- **Sampling:** `temperature=1.0`, `top_p=0.95`, `max_tokens >= 32768`
- **Tool formats:** `json`, `xml`, `xml_typed` via `chat_template_kwargs.tool_call_format`
- **Serving:** BF16, TP=1, FlashAttention-3 (SGLang) or vLLM with `--reasoning-parser k2_horizon --tool-call-parser k2_horizon`
- **Uno diffusion adapters** (IFM/K2-Horizon-7B-Uno) provide lossless speedup as LoRA adapters

## References

- Model card: https://huggingface.co/IFM/K2-Horizon-7B
- Blog post: https://ifm.ai/blog/k2/
- Code: https://github.com/ifm-ai/xllm
- W&B logs: https://wandb.ai/llm360/K2-Horizon-7B
