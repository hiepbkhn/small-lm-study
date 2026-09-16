# Small LM Study

Architecture study notes for small-to-mid open LLMs.

## Models

| Model | Directory | Params | Context | Arch Type |
|-------|-----------|--------|---------|-----------|
| [K2-Horizon-7B](K2-Horizon/) | `K2-Horizon/` | 7B dense | 512K | custom `k2_horizon` |
| [MiniCPM5-2B](MiniCPM5/) | `MiniCPM5/` | 2B dense | 128K | standard `Llama` |

## Quick Comparison

| | K2-Horizon-7B | MiniCPM5-2B |
|--|--------------|-------------|
| Layers | 36 | 42 |
| GQA ratio | 4:1 (32/8) | 8:1 (16/2) |
| head_dim | 128 | 128 |
| Context | 512K | 128K |
| RoPE θ | 1e7 | 5e6 |
| Weights (BF16) | ~16.8 GB | ~4.7 GB |
| KV @ max ctx | 72 GB @ 512K | 5.25 GB @ 128K |
| Custom code? | yes (`trust_remote_code`) | no |
| Training recipe | Pretrain → Mid ×4 → RL ×5 → SFT ×2 | Pretrain → Mid → SFT → RL → OPD (16 teachers) |

## Files Per Model

- `arch.md` — architecture: config, components, forward pass, training stages
- `kv-cache.md` — KV-cache analysis (K2-Horizon only so far)
