# core/llm — LLM Interfaces

Unified interface for LLM calls used across both pipelines. Models are instantiated by key from `configs/models.yaml`.

## Modules

| File              | Purpose |
|-------------------|---------|
| `base.py`         | `LLMModel` ABC: `generate()`, `generate_with_logprobs()`, batch API hooks |
| `openai_model.py` | `OpenAIModel` — GPT-4o-mini via Chat Completions or Batch API; structured JSON output; logprobs |
| `qwen_model.py`   | `QwenModel` — local HuggingFace or remote vLLM; logprobs |
| `config.py`       | `load_model_from_config(yaml_path, model_key)` — factory function |
| `chat_session.py` | Multi-turn chat session management (used by B5 critique stages) |
| `llm.md`          | Internal notes on prompt formatting and structured output schemas |

## Configuration

Model definitions live in `configs/models.yaml`. Example:
```yaml
gpt4o_mini:
  type: openai
  model_name: gpt-4o-mini
  temperature: 0.0
  max_tokens: 2000

qwen_local_30b:
  type: qwen
  model_name: Qwen/Qwen3-30B-A3B
  mode: local
  device: cuda
  temperature: 0.0
```

## API Keys

Set `OPENAI_API_KEY` (and any HuggingFace tokens) in `.env` at the repository root. See `.env.example`.
