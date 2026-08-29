"""Shared utilities for batch API operations."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from core.dataset.schema import Dataset


def load_dotenv(root: Path) -> None:
    """Load OPENAI_API_KEY from .env file if not already set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    candidate = root / ".env"
    if candidate.is_file():
        with open(candidate) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val and not val.startswith("sk-your"):
                        os.environ["OPENAI_API_KEY"] = val
                        return


def get_openai_client():
    """Get OpenAI client with API key from environment."""
    import openai
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key.startswith("sk-your"):
        raise SystemExit(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "  → Edit the .env file in the project root and set your real key."
        )
    return openai.OpenAI(api_key=key)


def load_model_config(llm_name: str, models_config_path: str) -> tuple[str, float, int]:
    """Load (model_name, temperature, max_tokens) from models.yaml config."""
    import yaml
    with open(models_config_path) as f:
        cfg = yaml.safe_load(f)
    entry = cfg.get(llm_name, {})
    return (
        entry.get("model_name", "gpt-4o-mini"),
        float(entry.get("temperature", 0.0)),
        int(entry.get("max_tokens", entry.get("max_new_tokens", 1024))),
    )


def get_folder_model_name(model_name: str) -> str:
    """Strip org prefix from model name (e.g. 'Qwen/Qwen3-30B-A3B' → 'Qwen3-30B-A3B')."""
    return model_name.split("/")[-1]


def get_results_base_dir(method_name: str, model_name: str) -> str:
    """Get results_base_dir string for (method, judge_model)."""
    folder = get_folder_model_name(model_name)
    return f"results/baselines/{method_name}/{folder}"


def get_majority_label(dataset: Dataset) -> str:
    """Get the majority label from a dataset's working split."""
    counts = Counter(inst.label for inst in dataset.working)
    return counts.most_common(1)[0][0] if counts else "yes"


def submit_batch(requests_path: Path, client) -> str:
    """Submit batch request file to OpenAI and return batch ID."""
    with open(requests_path, "rb") as f:
        batch_obj = client.batches.create(
            input_file=f,
            endpoint="/v1/chat/completions",
            timeout_hours=24,
        )
    return batch_obj.id
