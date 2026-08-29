"""prepare_batch.py — Prepare B1–B4 judge requests for FOLIO dataset curation.

This script takes the raw (NL_sentence, FOL_sentence_old) pairs from the FOLIO
dataset and builds batch request files so that Qwen3-30B-A3B (uni server) and
GPT-4o-mini (OpenAI Batch API) can judge whether each original formalization is
correct — effectively curating the dataset.

The candidate formalization (φ̂) is FOL_sentence_old (the pre-curation formula).
The goal is to see whether the prompting strategies can detect errors that manual
curation found and corrected.

Output layout
-------------
    requests/dataset_curation/
        qwen/
            {method}.jsonl          — vLLM-format requests (uni server)
            index.jsonl             — all metadata entries
        openai/
            {method}.jsonl          — OpenAI Batch API-format requests
            index.jsonl             — all metadata entries

Where method ∈ {B1, B2, B3, B4}.

Custom-ID format
----------------
    {method}|pv{pv}|curation_{id}

    e.g.  B1|pv1|curation_story_380
          B3|pv2|curation_story_12

Workflow
--------
1. Run this script locally (--no-submit by default for OpenAI):
       python pipeline1/prepare_batch.py

2. Inspect the counts and verify a few requests, then call with --submit to
   actually send the OpenAI batches:
       python pipeline1/prepare_batch.py --submit

3. For Qwen, copy request files to the university server and run inference.

4. Copy responses back and process with extract_answers.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.strategies.strategy import (
    B1_ZeroShotCoT,
    B2_FewShot,
    B3_FewShotCoT,
    B4_FeatureBased,
    build_folio_ontology_block,
)
from core.strategies.prompts.common import user_message
from pipeline1.batch.config import (
    get_schema_for_method as _schema_for,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_DATASET: Path = _ROOT / "core" / "dataset" / "datasets" / "folio" / "dataset_rawpairs.jsonl"

_METHOD_CLASSES = {
    "B1": B1_ZeroShotCoT,
    "B2": B2_FewShot,
    "B3": B3_FewShotCoT,
    "B4": B4_FeatureBased,
}

# Qwen (vLLM on uni server)
_QWEN_MODEL_NAME   = "Qwen3-30B-A3B"
_QWEN_MODEL_FULL   = "Qwen/Qwen3-30B-A3B"
_QWEN_LLM_NAME     = "qwen_local_30b"
_QWEN_MAX_TOKENS   = 2000
_QWEN_TEMPERATURE  = 0.0

# GPT-4o-mini (OpenAI Batch API)
_OPENAI_MODEL_NAME  = "gpt-4o-mini"
_OPENAI_LLM_NAME    = "gpt4o_mini"
_OPENAI_MAX_TOKENS  = 1024
_OPENAI_TEMPERATURE = 0.0

_SEED = 0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_raw_pairs(path: Path) -> list[dict]:
    """Load multi-line JSON records from a rawpairs JSONL file."""
    records: list[dict] = []
    decoder = json.JSONDecoder(strict=False)   # allow literal control chars in strings
    content = path.read_text(encoding="utf-8").strip()
    idx = 0
    while idx < len(content):
        try:
            obj, end = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end
            while idx < len(content) and content[idx] in " \t\n\r":
                idx += 1
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error in {path} near pos {idx}: {exc}") from exc
    return records


# ---------------------------------------------------------------------------
# Request building — B1–B4
# ---------------------------------------------------------------------------

def _build_b1_b4_requests(
    records: list[dict],
    model_kind: str,
    model_name: str,
    llm_name: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Build batch requests and index entries for B1–B4.

    Returns dict keyed by method name → (requests, index).
    """
    result: dict[str, tuple[list[dict], list[dict]]] = {}

    for method_name, cls in _METHOD_CLASSES.items():
        requests: list[dict] = []
        index: list[dict] = []
        response_format = _schema_for(method_name)

        for pv in (1, 2, 3, 4):
            method = cls(llm=None, configuration="B", prompt_version=pv)
            system = method._build_system_prompt()
            _fill_requests(
                records, method_name, pv,
                system, response_format, temperature, max_tokens,
                model_name, llm_name, model_kind,
                requests, index,
            )

        result[method_name] = (requests, index)

    return result


def _fill_requests(
    records: list[dict],
    method_name: str,
    pv: int,
    system: str,
    response_format: dict,
    temperature: float,
    max_tokens: int,
    model_name: str,
    llm_name: str,
    model_kind: str,
    requests: list[dict],
    index: list[dict],
    extra_meta: dict | None = None,
) -> None:
    for rec in records:
        instance_id = rec["id"]
        nl          = rec["NL_sentence"]
        phi_hat     = rec["FOL_sentence_old"]
        phi_gold    = rec["FOL_sentence_new"]
        corrected   = rec.get("corrected", "unknown")

        ontology_block = build_folio_ontology_block(instance_id)
        u_msg = user_message(nl, phi_hat, ontology_block=ontology_block, version=pv)
        custom_id = f"{method_name}|pv{pv}|curation_{instance_id}"

        requests.append({
            "custom_id": custom_id,
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":           model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": u_msg},
                ],
                "temperature":     temperature,
                "max_tokens":      max_tokens,
                "response_format": response_format,
                "logprobs":        True,
                "top_logprobs":    5,
            },
        })
        entry = {
            "custom_id":      custom_id,
            "method_name":    method_name,
            "prompt_version": pv,
            "llm_name":       llm_name,
            "model_name":     model_name,
            "model_kind":     model_kind,
            "seed":           _SEED,
            "instance_id":    instance_id,
            "phi_gold":       phi_gold,
            "phi_hat":        phi_hat,
            "corrected":      corrected,
        }
        if extra_meta:
            entry.update(extra_meta)
        index.append(entry)


# ---------------------------------------------------------------------------
# OpenAI submit helpers
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    candidate = _ROOT / ".env"
    if candidate.is_file():
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and not val.startswith("sk-your"):
                    os.environ["OPENAI_API_KEY"] = val
                    return


def _get_openai_client():
    import openai
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key.startswith("sk-your"):
        raise SystemExit(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "  → Edit the .env file in the project root and set your real key."
        )
    return openai.OpenAI(api_key=key)


def _submit_openai_batch(requests_path: Path, client) -> str:
    with open(requests_path, "rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare B1–B4 curation batch requests for Qwen and GPT-4o-mini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually submit the OpenAI batches (default: only write files).",
    )
    args = parser.parse_args()

    out_root   = _ROOT / "pipeline1" / "assessing_original" / "requests"
    qwen_dir   = out_root / "qwen"
    openai_dir = out_root / "openai"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    openai_dir.mkdir(parents=True, exist_ok=True)

    if args.submit:
        _load_dotenv()
        client = _get_openai_client()
    else:
        client = None

    grand_total: dict[str, int] = {"qwen": 0, "openai": 0}
    global_index: dict[str, list[dict]] = {"qwen": [], "openai": []}
    submitted_batches: list[tuple[str, str]] = []  # (key, batch_id)

    print(f"\n{'='*60}")
    print(f"FOLIO  ({_RAW_DATASET.name})")
    records = _load_raw_pairs(_RAW_DATASET)
    print(f"  Loaded {len(records)} records")

    for model_kind, out_dir, model_name, llm_name, temperature, max_tokens in [
        ("qwen",   qwen_dir,   _QWEN_MODEL_FULL,   _QWEN_LLM_NAME,   _QWEN_TEMPERATURE,   _QWEN_MAX_TOKENS),
        ("openai", openai_dir, _OPENAI_MODEL_NAME, _OPENAI_LLM_NAME, _OPENAI_TEMPERATURE, _OPENAI_MAX_TOKENS),
    ]:
        print(f"\n  [{model_kind.upper()}] Building requests …")

        b1b4 = _build_b1_b4_requests(
            records, model_kind, model_name, llm_name, temperature, max_tokens,
        )
        for method_name, (reqs, idx) in b1b4.items():
            out_path = out_dir / f"{method_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for req in reqs:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
            global_index[model_kind].extend(idx)
            grand_total[model_kind] += len(reqs)
            print(f"    {method_name}: {len(reqs):>6} requests → {out_path.relative_to(_ROOT)}")

            if model_kind == "openai" and args.submit:
                print(f"    [{method_name}] Submitting …", end=" ", flush=True)
                batch_id = _submit_openai_batch(out_path, client)
                print(f"batch_id: {batch_id}")
                submitted_batches.append((method_name, batch_id))
                (out_dir / f"{method_name}_batch_id.txt").write_text(
                    batch_id + "\n", encoding="utf-8"
                )
                _append_registry(method_name, out_path, batch_id)

    # Write global indices
    for model_kind, out_dir in [("qwen", qwen_dir), ("openai", openai_dir)]:
        idx_path = out_dir / "index.jsonl"
        with open(idx_path, "w", encoding="utf-8") as f:
            for entry in global_index[model_kind]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n  {model_kind.upper()} global index ({len(global_index[model_kind])} entries) → {idx_path.relative_to(_ROOT)}")

    # Summary
    n_records = len(_load_raw_pairs(_RAW_DATASET))
    n_per_rec = len(_METHOD_CLASSES) * 4   # methods × 4 pvs

    print(f"\n{'='*60}")
    print("REQUEST COUNT SUMMARY")
    print(f"{'='*60}")
    print(f"  Records:                          {n_records}")
    print(f"  B1–B4 configs per record:        {n_per_rec}  (4 methods × 4 pv)")
    print(f"  Qwen total:                      {grand_total['qwen']}")
    print(f"  OpenAI total:                    {grand_total['openai']}")

    if args.submit:
        print(f"\nSubmitted {len(submitted_batches)} OpenAI batches.")
        for method_name, bid in submitted_batches:
            print(f"  [{method_name}] {bid}")
    else:
        print(f"\n--submit not set: all files written, nothing submitted to OpenAI.")
        print("When ready, run:")
        print("  python pipeline1/prepare_batch.py --submit")


def _append_registry(method: str, req_path: Path, batch_id: str) -> None:
    registry_path = _ROOT / "batch_registry.jsonl"
    record = {
        "batch_id":     batch_id,
        "request_file": str(req_path),
        "dataset_name": "folio_rawpairs",
        "method":       method,
        "stage":        "dataset_curation",
        "status":       "submitted",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
