# Agnes-3.0-Flash Architecture

Source: [Agnes-AI/Agnes-3.0-Flash on Hugging Face](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash) (Agnes-3.0-Flash **Preview**, open-weights checkpoint).

## Overview

Open-weights multimodal model (text, image, video understanding) built for flagship-class reasoning without flagship-class hardware.

- Parameters: **33B**
- Context window: **262,144 tokens**
- License: Apache-2.0
- Quantized variants available for llama.cpp, Ollama, LM Studio

## Model Architecture

Agnes-3.0-Flash Preview is a **hybrid-attention decoder**: three of every four layers run a gated delta rule (recurrent, with per-layer state independent of sequence length), and the fourth runs standard global attention. Only 18 of the 72 layers therefore hold a KV cache that grows with context.

| Component | Specification |
|---|---|
| Context length | 262,144 tokens |
| Decoder layers | 72 = 54 delta-rule recurrent + 18 global attention, alternating 3:1 |
| Hidden size | 5120 |
| Global attention | 24 query heads / 4 KV heads (6:1 GQA), head dim 256; RMS-norm on q and k, sigmoid-gated output |
| Delta-rule layers | 16 key heads / 48 value heads, head dim 128; causal conv (kernel 4) in front, gated RMS-norm; recurrent state in fp32 |
| Feed-forward | SwiGLU, intermediate size 17,408; plus a parallel SwiGLU 2048 branch in every layer |
| Positions | 3-axis rotary (text / height / width), interleaved mrope sections 11:11:10, base 1e7, applied to the first 25% of each head dim (64 dims) |
| Vocabulary | 248,320 |
| Vision tower | 27 layers, hidden 1152, patch 16, 2x2 spatial merge, projected to 5120 |

## Capabilities

- Advanced reasoning with adjustable effort levels (`high` / `medium` / `low`)
- Coding and debugging
- Long-context analysis (262,144 tokens)
- Image and video understanding
- Tool calling (`tool_call` / `tool_result` tokens)
- Streaming
- OpenAI-compatible APIs via sglang

## Inference

Remote code required — load with `trust_remote_code=True`.

### Transformers

```bash
pip install "transformers>=5.12" torch torchvision accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "Agnes-AI/Agnes-3.0-Flash"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, dtype="bfloat16", device_map="auto", trust_remote_code=True
)
msgs = [{"role": "user", "content": "请用三句话解释什么是人工智能。"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=256)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

### Images and video

```python
from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
msgs = [{"role": "user", "content": [
    {"type": "image", "image": "photo.jpg"},
    {"type": "text", "text": "描述这张图。"},
]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

### Reasoning effort

The chat template exposes three reasoning levels — `high` (default), `medium`, `low` — plus a thinking-off switch:

```python
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                              reasoning_effort="medium")   # or enable_thinking=False
```

### Tool calling

The chat template renders tool definitions. The model emits calls as `tool_call<function=...>