# LLM Layer — Usage Guide

The LLM layer (`core/llm/`) provides a backend-agnostic interface for calling
language models.  All code that needs an LLM should depend on the `LLM` ABC —
never on a concrete backend class directly.

---

## Classes

| Class | File | When to use |
| --- | --- | --- |
| `LLM` | `base.py` | ABC — type hint for all LLM arguments |
| `OpenAIModel` | `openai_model.py` | GPT-4o / GPT-4o-mini via the OpenAI API |
| `QwenModel` | `qwen_model.py` | Qwen 2.5, either local (HuggingFace) or via API |
| `ChatSession` | `chat_session.py` | Multi-turn stateful conversations |
| `load_model_from_config` | `config.py` | YAML factory — preferred instantiation path |

---

## Instantiation

Always prefer the YAML factory to avoid hardcoding model names or keys:

```python
from core.llm.config import load_model_from_config

llm = load_model_from_config("configs/models.yaml", model_key="default")
```

Available keys are defined in `configs/models.yaml`.  To list them:

```python
from core.llm.config import list_available_models
print(list_available_models("configs/models.yaml"))
```

---

## API Key Setup

`OpenAIModel` resolves the key in this order (first match wins):

1. `api_key` constructor argument
2. A file at `api_key_path` containing a line `OPENAI_API_KEY=sk-...`
3. The `OPENAI_API_KEY` environment variable

The recommended approach is the key-file method, configured in `configs/models.yaml`:

```yaml
default:
  type: openai
  model_name: gpt-4o
  api_key_path: secrets/api_keys.env
```

---

## `generate()` return values

The return type depends on which optional arguments are set:

| `response_format` | `logprobs` | Return type | Keys present |
| --- | --- | --- | --- |
| not set | `False` | `str` | — |
| set | `False` | `dict` | `"structured"` |
| not set | `True` | `dict` | `"text"`, `"logprobs"` |
| set | `True` | `dict` | `"text"`, `"logprobs"`, `"structured"` |

### Logprobs format

```python
[
    {
        "token": "yes",
        "logprob": -0.123,
        "top_logprobs": [
            {"token": "yes", "logprob": -0.123},
            {"token": "no",  "logprob": -2.456},
        ],
    },
    ...
]
```

---

## Structured output

Pass a Pydantic `BaseModel` subclass as `response_format`.  The parsed model
is returned under the `"structured"` key as a plain dict (`model_dump()`):

```python
from pydantic import BaseModel

class VerdictOutput(BaseModel):
    quality_score: str
    comment: str

result = llm.generate(messages, response_format=VerdictOutput)
parsed = result["structured"]   # {"quality_score": "yes", "comment": "..."}
```

For `OpenAIModel` this uses the native structured-output API
(`type_to_response_format_param`).  For `QwenModel` it falls back to
prompt-based JSON extraction with Pydantic validation.

---

## Multi-turn conversations via ChatSession

`llm.chat()` returns a `ChatSession` bound to the LLM.  Each `send()` call
appends the exchange to the session history:

```python
session = llm.chat(system_prompt="You are a helpful assistant.")
response1 = session.send("First question")
response2 = session.send("Follow-up", response_format=VerdictOutput)
```

`ChatSession` is used internally by B5 (self-critique) to carry multi-round
conversations.

---

## Batch API (OpenAI only)

Submit a batch job and retrieve results asynchronously:

```python
# Submit
batch_result = llm.generate(messages, online=False)
batch_id = batch_result["batch_id"]

# Poll and retrieve (blocks until complete)
results = llm.retrieve_batch(batch_id)
```

Batch mode is useful for large offline evaluations where latency is not
critical (up to 24 h turnaround, 50% cost discount).

---

## Design notes

- All backends default to `temperature=0.0` (greedy / deterministic).
- The `LLM` ABC does not own any rate-limiting logic; that is the runner's
  responsibility (see `experiments/rate_limiter.py`).
- `QwenModel` supports both `mode="local"` (HuggingFace, requires GPU) and
  `mode="api"` (remote endpoint).
- The `online=False` batch path is only implemented for `OpenAIModel`.
