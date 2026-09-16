# Small LM Study

Architecture study notes for open LLMs, from small to flagship.

## Models

| Model | Directory | Params | Context | Arch Type |
|-------|-----------|--------|---------|-----------|
| [K2-Horizon-7B](K2-Horizon/) | `K2-Horizon/` | 7B dense | 512K | custom `k2_horizon` |
| [MiniCPM5-2B](MiniCPM5/) | `MiniCPM5/` | 2B dense | 128K | standard `Llama` |
| [Nanbeige4.2-3B](Nanbeige-4.2/) | `Nanbeige-4.2/` | 3B non-emb | 256K | looped `Nanbeige` |
| [Spark-X2.5-4B](Spark-X2.5/) | `Spark-X2.5/` | 4B dense | 1M | hybrid SWA+full `Spark2_5` |
| [Agnes-3.0-Flash](Agnes-3.0-Flash/) | `Agnes-3.0-Flash/` | 33B dense | 262K | hybrid delta-rule + global `Agnes` |

## Quick Comparison

| | K2-Horizon-7B | MiniCPM5-2B | Nanbeige4.2-3B | Spark-X2.5-4B | Agnes-3.0-Flash |
|--|--------------|-------------|----------------|---------------|----------------|
| Layers | 36 | 42 | 22 × 2 loops | 36 (27 SWA + 9 full) | 72 (54 delta + 18 global) |
| GQA ratio | 4:1 (32/8) | 8:1 (16/2) | 6:1 (48/8) | 4:1 (16/4) | 6:1 (24/4) |
| head_dim | 128 | 128 | 128 | 256 | 256 |
| Context | 512K | 128K | 256K | 1M | 262K |
| RoPE θ | 1e7 | 5e6 | 7e7 | 5e6 full / 1e4 sliding | 1e7 (3-axis mrope) |
| Weights (BF16) | ~16.8 GB | ~4.7 GB | ~6.5 GB | ~9 GB | ~66 GB |
| KV @ max ctx | 72 GB @ 512K | 5.25 GB @ 128K | 46.9 GB @ 256K | 38.7 GB @ 1M | 18 GB @ 262K |
| Custom code? | yes (`trust_remote_code`) | no | yes (`trust_remote_code`) | yes (`trust_remote_code`) | yes (`trust_remote_code`) |
| Training recipe | Pretrain → Mid ×4 → RL ×5 → SFT ×2 | Pretrain → Mid → SFT → RL → OPD (16 teachers) | Pretrain → SFT (env synthesis) → RL (outcome+process) | Pretrain 20T → SFT → RL → MOPD | — |

## Files Per Model

- `arch.md` — architecture: config, components, forward pass, training stages
- `kv-cache.md` — KV-cache analysis
