# Spark-X2.5-4B Architecture

Source: [XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B)
Base model: [XHToken/Spark-X2.5-4B-Base](https://huggingface.co/XHToken/Spark-X2.5-4B-Base)

Spark-X2.5-4B is a compact general-purpose LLM by the SparkLLM team (XHToken),
designed for on-device deployment. It uses a hybrid attention architecture
(sliding-window + full attention) for efficient long-context inference, and
combines SFT + large-scale RL (MOPD) post-training. Trained on ~20T tokens on
Huawei Ascend clusters.

## config.json

```json
{
  "architectures": ["Spark2_5ForCausalLM"],
  "model_type": "spark2_5",
  "auto_map": {
    "AutoConfig": "configuration_spark.Spark2_5Config",
    "AutoModel": "modeling_spark.Spark2_5Model",
    "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM"
  },
  "vocab_size": 131072,
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "num_hidden_layers": 36,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "head_dim": 256,
  "hidden_act": "gelu",
  "max_position_embeddings": 1048576,
  "rope_parameters": {
    "full_attention": {
      "partial_rotary_factor": 0.25,
      "rope_theta": 5000000
    },
    "sliding_attention": {
      "partial_rotary_factor": 1.0,
      "rope_theta": 10000
    }
  },
  "sliding_window": 512,
  "headwise_attn_output_gate": true,
  "gate_attn_act_mode": "sigmoid",
  "layer_types": [
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
  ],
  "tie_word_embeddings": true,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "mlp_bias": false,
  "rms_norm_eps": 1e-06,
  "initializer_range": 0.01976,
  "pad_token_id": 2,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "torch_dtype": "bfloat16"
}
```

## Key architectural features

### Hybrid attention: sliding-window + full attention
- 36 layers total: **9 full-attention** layers (every 4th: 3, 7, 11, …, 35)
  and **27 sliding-window attention** layers.
- Sliding window size: 512 tokens.
- This reduces KV-cache footprint dramatically: only 9 of 36 layers store
  full KV history; the other 27 store only a 512-token window.
- Each layer type has its own RoPE config:
  - Full attention: `rope_theta=5e6`, `partial_rotary_factor=0.25`
    (only 25% of head_dim uses rotary encoding; the rest is positional-free).
  - Sliding attention: `rope_theta=1e4`, `partial_rotary_factor=1.0`
    (full rotary over head_dim).

### GQA + headwise attention output gate
- GQA: 16 query heads, 4 KV heads (4:1 ratio).
- `head_dim=256` (Q dim = 16×256 = 4096; KV dim = 4×256 = 1024).
- `headwise_attn_output_gate=true`: an extra `g_proj` (hidden→num_heads, no
  bias) produces per-head gate scores, applied as
  `attn_output × sigmoid(gate_score)` before `out_proj`. This is a
  head-wise gating of attention output (analogous to "headwise gates" in
  some hybrid architectures).
- Single fused `q_k_v_proj` for Q/K/V.

### Decoder layer
Pre-norm Transformer block:
- `input_layernorm` (RMSNorm) → attention → residual
- `post_attention_layernorm` (RMSNorm) → MLP → residual
- MLP: SwiGLU-style (gate×up) with **GELU** activation,
  `intermediate_size=10240`.
- `tie_word_embeddings=true`: `lm_head` shares weights with token embedding.

### Inference
- Context length: up to 1,048,576 tokens (1M).
- Recommended sampling: `temperature=1.0`, `top_p=0.95`, `top_k=-1`
  (no top-k cutoff).
- Thinking enabled by default in chat template; disable with
  `enable_thinking=false`.
- Load with `trust_remote_code=True`.
- vLLM: `--trust-remote-code`; Ascend NPUs need the Spark plugin.

## Performance highlights

| Benchmark | Spark-X2.5-4B |
|---|---|
| AIME 2026 | 90.7 |
| IMO-AnswerBench | 74.2 |
| HMMT Feb 2026 | 81.2 |
| SWE-Bench Pro | 44.4 |
| SWE-Bench Multilingual | 53.3 |
| IFEval | 93.0 |
| IFBench | 75.0 |
| τ³-bench | 30.4 |
| MCP-Atlas | 54.6 |
| BrowseComp | 40.9 |

Beats Qwen3.5-9B on several agentic benchmarks (τ³-bench, MCP-Atlas,
BrowseComp) while matching or exceeding it on coding (SWE-Bench Pro/Multilingual).

## Citation
```bibtex
@misc{sparkx2.5,
    title  = {Spark-X2.5 4B&1.7B: Pushing the Limits of Agentic Capabilities in On-Device Models},
    author = {SparkLLM Team},
    year   = {2026}
}
```
