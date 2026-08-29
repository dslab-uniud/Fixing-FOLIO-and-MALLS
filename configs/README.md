# configs — YAML Configuration

| File          | Purpose |
|---------------|---------|
| `models.yaml` | LLM model definitions used by `core.llm.config.load_model_from_config()` |

Each entry specifies `type` (openai / qwen), `model_name`, generation hyperparameters (`temperature`, `max_tokens`), and (for local Qwen) `mode: local`/`api` plus device settings.

API keys are loaded from `.env` at the repository root — never put keys in this YAML.
