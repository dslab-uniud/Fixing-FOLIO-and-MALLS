"""prepare_malls_batch.py — Prepare B1–B4 judge requests for MALLS dataset
(assessing_original pipeline).

This script reads the ORIGINAL MALLS formula pairs from
    core/dataset/datasets/malls/dataset_rawpairs.jsonl
and builds batch request files so that GPT-4o-mini (OpenAI Batch API) can
judge whether each original formalization (FOL_sentence_old) is correct
against the gold formula (FOL_sentence_new).

Each MALLS instance has its own per-instance ontology (from signature.jsonl),
unlike FOLIO which uses a fixed per-story ontology.

Output layout
-------------
    pipeline1/assessing_original/requests/
        openai/
            malls_{method}.jsonl    — OpenAI Batch API format
            index_malls.jsonl       — metadata index

Where method ∈ {B1, B2, B3, B4}.

Custom-ID format
----------------
    {method}|pv{pv}|curation_malls_{id}

    e.g.  B1|pv1|curation_malls_1
          B3|pv2|curation_malls_42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.strategies.strategy import (
    B1_ZeroShotCoT,
    B2_FewShot,
    B3_FewShotCoT,
    B4_FeatureBased,
    build_malls_ontology_block,
)
from core.strategies.prompts.common import user_message
from pipeline1.batch.config import (
    get_schema_for_method as _schema_for,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MALLS_RAWPAIRS = (
    _ROOT / "core" / "dataset" / "datasets" / "malls" / "dataset_rawpairs.jsonl"
)

_METHOD_CLASSES = {
    "B1": B1_ZeroShotCoT,
    "B2": B2_FewShot,
    "B3": B3_FewShotCoT,
    "B4": B4_FeatureBased,
}

_OPENAI_MODEL_NAME  = "gpt-4o-mini"
_OPENAI_LLM_NAME    = "gpt4o_mini"
_OPENAI_MAX_TOKENS  = 1024
_OPENAI_TEMPERATURE = 0.0

_SEED = 0

_SPLIT = "malls"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rawpairs(path: Path) -> list[dict]:
    """Load MALLS dataset_rawpairs.jsonl (multi-line JSON objects)."""
    decoder = json.JSONDecoder(strict=False)
    content = path.read_text(encoding="utf-8")
    records: list[dict] = []
    idx = 0
    while idx < len(content):
        # Skip whitespace
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1
        if idx >= len(content):
            break
        if content[idx] == "{":
            try:
                obj, end = decoder.raw_decode(content, idx)
                records.append(obj)
                idx = end
            except json.JSONDecodeError:
                idx += 1
        else:
            idx += 1
    return records


def _instance_id(rec: dict) -> str:
    """Stable instance ID: curation_malls_{id}."""
    return f"curation_{_SPLIT}_{rec['id']}"


# ---------------------------------------------------------------------------
# Request building — B1–B4
# ---------------------------------------------------------------------------

def _fill(
    records: list[dict],
    method_name: str,
    pv: int,
    system: str,
    response_format: dict,
    temperature: float,
    max_tokens: int,
    model_name: str,
    llm_name: str,
    requests: list[dict],
    index: list[dict],
    extra_meta: dict | None = None,
) -> None:
    for rec in records:
        instance_id = _instance_id(rec)
        nl          = rec["NL_sentence"]
        phi_hat     = rec.get("FOL_sentence_old", "").strip()
        phi_gold    = rec.get("FOL_sentence_new", "").strip()
        ambiguity   = rec.get("ambiguity", False)
        ontology    = build_malls_ontology_block(int(rec["id"]))

        u_msg     = user_message(nl, phi_hat, ontology_block=ontology, version=pv)
        custom_id = f"{method_name}|pv{pv}|{instance_id}"

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
            "custom_id":   custom_id,
            "method_name": method_name,
            "prompt_version": pv,
            "llm_name":    llm_name,
            "model_name":  model_name,
            "model_kind":  "openai",
            "seed":        _SEED,
            "instance_id": instance_id,
            "malls_id":    rec["id"],
            "phi_gold":    phi_gold,
            "phi_hat":     phi_hat,
            "ambiguity":   ambiguity,
        }
        if extra_meta:
            entry.update(extra_meta)
        index.append(entry)


def _build_b1_b4_requests(
    records: list[dict],
    model_name: str,
    llm_name: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, tuple[list[dict], list[dict]]]:
    result: dict[str, tuple[list[dict], list[dict]]] = {}

    for method_name, cls in _METHOD_CLASSES.items():
        requests: list[dict] = []
        index:    list[dict] = []
        response_format = _schema_for(method_name)

        for pv in (1, 2, 3, 4):
            method = cls(llm=None, configuration="B", prompt_version=pv)
            system = method._build_system_prompt()
            _fill(records, method_name, pv, system, response_format,
                  temperature, max_tokens, model_name, llm_name,
                  requests, index)

        result[method_name] = (requests, index)

    return result


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
        description="Prepare B1–B4 batch requests for MALLS original-formula assessment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually submit the OpenAI batches (default: only write files).",
    )
    args = parser.parse_args()

    if not _MALLS_RAWPAIRS.is_file():
        raise SystemExit(f"ERROR: rawpairs file not found: {_MALLS_RAWPAIRS}")

    out_root   = _ROOT / "pipeline1" / "assessing_original" / "requests"
    openai_dir = out_root / "openai"
    openai_dir.mkdir(parents=True, exist_ok=True)

    if args.submit:
        _load_dotenv()
        client = _get_openai_client()
    else:
        client = None

    records = _load_rawpairs(_MALLS_RAWPAIRS)
    print(f"Loaded {len(records)} MALLS rawpairs records")
    n_ambiguous = sum(1 for r in records if r.get("ambiguity"))
    print(f"  Non-ambiguous: {len(records) - n_ambiguous}  |  Ambiguous: {n_ambiguous}")

    grand_total = 0
    submitted_batches: list[tuple[str, str]] = []
    index_entries: list[dict] = []

    # B1–B4
    b1b4 = _build_b1_b4_requests(
        records, _OPENAI_MODEL_NAME, _OPENAI_LLM_NAME,
        _OPENAI_TEMPERATURE, _OPENAI_MAX_TOKENS,
    )
    for method_name, (reqs, idx) in b1b4.items():
        out_path = openai_dir / f"malls_{method_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for req in reqs:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        index_entries.extend(idx)
        grand_total += len(reqs)
        print(f"  {method_name}: {len(reqs):>5} requests → {out_path.relative_to(_ROOT)}")

        if args.submit:
            print(f"  [{method_name}] Submitting …", end=" ", flush=True)
            batch_id = _submit_openai_batch(out_path, client)
            print(f"batch_id: {batch_id}")
            submitted_batches.append((method_name, batch_id))
            (openai_dir / f"malls_{method_name}_batch_id.txt").write_text(
                batch_id + "\n", encoding="utf-8"
            )

    # Write index
    idx_path = openai_dir / "index_malls.jsonl"
    with open(idx_path, "w", encoding="utf-8") as f:
        for entry in index_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Index ({len(index_entries)} entries) → {idx_path.relative_to(_ROOT)}")

    # Summary
    n       = len(records)
    per_rec = len(_METHOD_CLASSES) * 4   # B1–B4 × pvs

    print(f"\n{'='*60}")
    print("REQUEST COUNT SUMMARY")
    print(f"{'='*60}")
    print(f"  MALLS records:            {n}")
    print(f"  B1–B4 per record:         {per_rec}  (4 methods × 4 pvs)")
    print(f"  Grand total (OpenAI):     {grand_total}")

    if args.submit:
        print(f"\nSubmitted {len(submitted_batches)} OpenAI batches.")
        for method, bid in submitted_batches:
            print(f"  [malls/{method}] {bid}")
    else:
        print(f"\n--submit not set: files written, nothing submitted to OpenAI.")
        print("When ready: python pipeline1/prepare_malls_batch.py --submit")


if __name__ == "__main__":
    main()
