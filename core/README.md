# core — Library Code

Reusable modules for dataset handling, FOL formalism, LLM interfacing, prompting strategy implementations, and metric computation. Both `pipeline1/` and `pipeline2/` import from `core/`.

## Subdirectories

| Folder         | Purpose |
|----------------|---------|
| `dataset/`     | Dataset schemas + loaders + the raw data files (FOLIO, MALLS) |
| `formalism/`   | First-Order Logic parser, AST, signature extraction, Z3 solver, ACE translator |
| `llm/`         | Unified LLM interface — OpenAI batch API, Qwen (HuggingFace local + API), config-driven factory |
| `strategies/`  | B1–B4 implementations + prompt templates (system prompts, few-shot examples, four prompt variants) |
| `metrics/`     | `compute_metrics()` (safety/usefulness/refinement), ensemble aggregation, plotting helpers |

Each subdirectory has its own `README.md` describing the modules and the public API.
