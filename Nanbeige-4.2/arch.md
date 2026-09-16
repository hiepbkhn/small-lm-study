# Nanbeige4.2-3B Architecture

Source: [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
Technical report: [arXiv:2607.22083](https://arxiv.org/abs/2607.22083)
Base model: [Nanbeige4.2-3B-Base](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base)

Nanbeige4.2-3B is a compact agentic LLM built as a Looped Transformer: it
reuses a small decoder layer stack repeatedly (weight-tied loops) to increase
effective depth and capacity without adding parameters. It has ~3B
non-embedding parameters and supports 256K context with strong tool-use /
code-agent behavior.

## config.json

```json
{
  "architectures": ["NanbeigeForCausalLM"],
  "model_type": "nanbeige",
  "auto_map": {
    "AutoConfig": "configuration_nanbeige.NanbeigeConfig",
    "AutoModel": "modeling_nanbeige.NanbeigeModel",
    "AutoModelForCausalLM": "modeling_nanbeige.NanbeigeForCausalLM"
  },
  "vocab_size": 166144,
  "hidden_size": 3072,
  "intermediate_size": 10752,
  "num_hidden_layers": 22,
  "num_attention_heads": 48,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "kv_channels": 128,
  "hidden_act": "silu",
  "max_position_embeddings": 262144,
  "rope_theta": 70000000,
  "rope_scaling": null,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "rms_norm_eps": 1e-05,
  "num_loops": 2,
  "loop_loss_weights": [],
  "skip_loop_final_norm": false,
  "pretraining_tp": 1,
  "tie_word_embeddings": false,
  "pad_token_id": 0,
  "bos_token_id": 166100,
  "eos_token_id": 166101,
  "torch_dtype": "bfloat16"
}
```

## Key architectural features

### Looped Transformer (weight-tied layers)
- `num_hidden_layers = 22` physical layers, `num_loops = 2`.
- The full 22-layer stack is executed twice with shared parameters, giving an
  effective depth of 44 layers at 3B non-embedding params.
- The final RMS norm is applied after each loop unless
  `skip_loop_final_norm=True`.
- `loop_loss_weights`: optional non-empty list used during multi-loop
  training; when set, the model runs `len(loop_loss_weights) + 1` loops.

### Attention
- GQA: `num_attention_heads=48`, `num_key_value_heads=8` (6 groups),
  `head_dim=128` (48*128 = 6144 q/kv dim, projected down via `o_proj`).
- No attention bias; RMSNorm on query/key optional (`qk_layernorm`).
- RoPE with `rope_theta = 70,000,000` for 256K context.
- Per-loop attention dispatch module (`NanbeigeLoopAttention`) for vLLM KV
  cache management.

### Decoder layer
Pre-norm Transformer block:
- `input_layernorm` (RMSNorm) -> self-attention -> residual
- `post_attention_layernorm` (RMSNorm) -> MLP -> residual
- MLP: SwiGLU (`silu` activation), `intermediate_size=10752`.
- Optional hyper-connection module (`NanbeigeHyperConnectionModule`)
  around attention/MLP when `enable_hyper_connection=True`.

### N-gram embeddings (optional, not enabled in this checkpoint)
- Hash-based n-gram embedding tables fused into token embeddings or at
  specific decoder layers.
- Config options: `emb_neighbor_num`, `emb_split_num`,
  `ngram_vocab_size_ratio`, `ngram_fused_mode` ("average" or "concat"),
  `insert_ngram_layer_idx`, `ngram_compressed_tokenizer`.

### Other optional features (disabled by default in this checkpoint)
- **mHC (manifold-constrained hyper-connections)**: residual-stream mixing
  matrices constrained via Sinkhorn normalization
  (`enable_mhc`, `mhc_sinkhorn_iterations`).
- **LoopSplit** (`enable_double_loop_split`): outer layers unlooped, a
  contiguous middle block is repeated; different effective depths for
  different parts of the network.
- **Depth attention** (`enable_depth_attention`): queries mix value states
  from cached anchor depths before normal token-level self-attention.

## Inference
- Context length: up to 262,144 tokens (256K).
- Recommended settings:
  - Agentic / tool-use: `temperature=1.0`, `max_new_tokens=65536`
  - Reasoning / chat: `temperature=0.6`, `max_new_tokens=131072`
- Chat template supports `enable_thinking`, `preserve_thinking`,
  `tool_call_format` ("xml" recommended, "json" supported).
- Load with `trust_remote_code=True` (custom `modeling_nanbeige.py`).
- SGLang / vLLM require the custom branches:
  - SGLang: `git clone -b nbg42 https://github.com/Nanbeige/sglang.git`
  - vLLM: `git clone -b nanbeige42 https://github.com/Nanbeige/vllm.git`
  - llama.cpp: `git clone -b nanbeige42 https://github.com/Nanbeige/llama.cpp.git`

## Performance highlights
- Beats larger models (Qwen3.5-9B, Gemma4-12B) on general-agent,
  code-agent, and reasoning benchmarks.
- SWE-Bench Verified: 63.6; GPQA-Diamond: 87.4; LiveCodeBench-V6: 72.5;
  HLE w/o Search: 17.8; Terminal-Bench 2.0: 44.1.

## Citation
```bibtex
@article{lab2026nanbeige4,
  title={Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model},
  author={Lab, Nanbeige and others},
  journal={arXiv preprint arXiv:2607.22083},
  year={2026}
}
```
