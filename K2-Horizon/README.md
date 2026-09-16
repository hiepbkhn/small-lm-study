# K2-Horizon-7B

Study notes for the **IFM/K2-Horizon-7B** dense 7B-class, 512K-context decoder-only LLM.

## Files

| File | Contents |
|------|----------|
| `arch.md` | Full architecture: config table, component breakdown, forward pass, training stages, inference notes |
| `kv-cache.md` | KV-cache analysis: why the model is not cache-efficient, detailed memory estimates across context lengths, multi-batch costs, quantization savings |

## Key Facts

- **7B dense** decoder-only transformer, 36 layers, GQA (32 q-heads / 8 KV heads, 128 head_dim)
- **512K** native context window, RoPE θ=1e7, SwiGLU MLP (intermediate 12288)
- BF16, ~16.8 GB weights; **72 GB KV cache** at full 512K (4.3× the model)
- Open weights, Apache 2.0; day-0 vLLM / SGLang / Ollama support
- Uno LoRA diffusion adapters available for lossless inference speedup

## Sources

- [Hugging Face — IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B)
- [IFM Blog — Introducing K2 Horizon](https://ifm.ai/blog/k2/)
- [GitHub — xllm (training code)](https://github.com/ifm-ai/xllm)
