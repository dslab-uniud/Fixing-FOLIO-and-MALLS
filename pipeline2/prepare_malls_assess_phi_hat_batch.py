"""prepare_malls_assess_phi_hat_batch.py — Prepare B1–B4 judge requests for MALLS
phi_hat assessment (pipeline 2) using GPT-4o-mini (OpenAI Batch API).

Reads the already-generated phi_hat results for MALLS from
pipeline2/phi_hat_generation/results/ and builds OpenAI batch requests so
GPT-4o-mini can judge each generated formula.

This covers two dataset configurations:
  - malls_qwen3_P1      (phi_hat source: malls__Qwen3-30B-A3B__P1.jsonl, 100 instances)
  - malls_gpt4omini_P1  (phi_hat source: malls__gpt-4o-mini__P1.jsonl, 100 instances)

For each configuration the script generates B1–B4 batch requests (prompt
versions 1–4).

Output layout
-------------
    pipeline2/assessing_phi_hat/requests/gpt-4o-mini/
        malls_qwen3_P1_{method}.jsonl
        malls_qwen3_P1_index.jsonl
        malls_gpt4omini_P1_{method}.jsonl
        malls_gpt4omini_P1_index.jsonl

Custom-ID format
----------------
    {method}|pv{pv}|{phi_hat_id}

where {phi_hat_id} is the `id` field from the phi_hat result file, e.g.:
    malls__1__Qwen3-30B-A3B__P1
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
from pipeline1.batch.config import get_schema_for_method as _schema_for

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHI_HAT_DIR = _ROOT / "pipeline2" / "phi_hat_generation" / "results"
_OUT_DIR     = _ROOT / "pipeline2" / "assessing_phi_hat" / "requests" / "gpt-4o-mini"

_DATASETS = {
    "malls_qwen3_P1":      "malls__Qwen3-30B-A3B__P1.jsonl",
    "malls_gpt4omini_P1":  "malls__gpt-4o-mini__P1.jsonl",
}

_METHOD_CLASSES = {
    "B1": B1_ZeroShotCoT,
    "B2": B2_FewShot,
    "B3": B3_FewShotCoT,
    "B4": B4_FeatureBased,
}

_MODEL_NAME  = "gpt-4o-mini"
_LLM_NAME    = "gpt4o_mini"
_MAX_TOKENS  = 1024
_TEMPERATURE = 0.0
_SEED        = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_phi_hat_results(filename: str) -> list[dict]:
    path = _PHI_HAT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"phi_hat results not found: {path}")
    instances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def _ontology_block(inst: dict) -> str:
    return build_malls_ontology_block(int(inst["id_source_dataset"]))


def _fill(
    instances: list[dict],
    method_name: str,
    pv: int,
    system: str,
    response_format: dict,
    requests: list[dict],
    index: list[dict],
    extra_meta: dict | None = None,
    dataset_name: str = "",
) -> None:
    for inst in instances:
        inst_id    = inst["id"]
        nl         = inst["p"]
        phi_hat    = inst["phi_hat"]
        phi_gold   = inst["phi_gold"]
        ambiguity  = inst.get("ambiguity_flag", False)
        gt_label   = inst.get("label", None)
        ontology   = _ontology_block(inst)

        u_msg     = user_message(nl, phi_hat, ontology_block=ontology, version=pv)
        custom_id = f"{method_name}|pv{pv}|{inst_id}"

        requests.append({
            "custom_id": custom_id,
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":           _MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": u_msg},
                ],
                "temperature":     _TEMPERATURE,
                "max_tokens":      _MAX_TOKENS,
                "response_format": response_format,
                "logprobs":        True,
                "top_logprobs":    5,
            },
        })
        entry = {
            "custom_id":        custom_id,
            "method_name":      method_name,
            "prompt_version":   pv,
            "llm_name":         _LLM_NAME,
            "model_name":       _MODEL_NAME,
            "model_kind":       "openai",
            "seed":             _SEED,
            "instance_id":      inst_id,
            "id_source_dataset": inst["id_source_dataset"],
            "phi_gold":         phi_gold,
            "phi_hat":          phi_hat,
            "ambiguity":        ambiguity,
            "ground_truth_label": gt_label,
            "dataset_name":     dataset_name,
        }
        if extra_meta:
            entry.update(extra_meta)
        index.append(entry)


def _build_b1_b4(
    instances: list[dict],
    dataset_name: str,
) -> dict[str, tuple[list[dict], list[dict]]]:
    result: dict[str, tuple[list[dict], list[dict]]] = {}

    for method_name, cls in _METHOD_CLASSES.items():
        requests: list[dict] = []
        index:    list[dict] = []
        response_format = _schema_for(method_name)
        for pv in (1, 2, 3, 4):
            method = cls(llm=None, configuration="B", prompt_version=pv)
            system = method._build_system_prompt()
            _fill(instances, method_name, pv, system, response_format,
                  requests, index, dataset_name=dataset_name)
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
        description="Prepare B1–B4 assessing_phi_hat batch requests for MALLS (gpt-4o-mini judge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually submit the OpenAI batches (default: only write files).",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(_DATASETS), default=list(_DATASETS),
        help="Which phi_hat dataset configs to process (default: all).",
    )
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.submit:
        _load_dotenv()
        client = _get_openai_client()
    else:
        client = None

    grand_total = 0

    for dataset_name in args.datasets:
        phi_hat_file = _DATASETS[dataset_name]
        instances    = _load_phi_hat_results(phi_hat_file)
        n            = len(instances)
        print(f"\n[{dataset_name}] Loaded {n} phi_hat instances from {phi_hat_file}")

        all_index: list[dict] = []
        dataset_total = 0

        b1b4 = _build_b1_b4(instances, dataset_name)
        for method_name, (reqs, idx) in b1b4.items():
            out_path = _OUT_DIR / f"{dataset_name}_{method_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for req in reqs:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
            all_index.extend(idx)
            dataset_total += len(reqs)
            grand_total   += len(reqs)
            print(f"  {method_name}: {len(reqs):>5} requests → {out_path.name}")

            if args.submit:
                print(f"  [{method_name}] Submitting …", end=" ", flush=True)
                batch_id = _submit_openai_batch(out_path, client)
                print(f"batch_id: {batch_id}")
                (out_path.parent / f"{dataset_name}_{method_name}_batch_id.txt"
                 ).write_text(batch_id + "\n", encoding="utf-8")

        # Write per-dataset index
        idx_path = _OUT_DIR / f"{dataset_name}_index.jsonl"
        with open(idx_path, "w", encoding="utf-8") as f:
            for entry in all_index:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  Index ({len(all_index)} entries) → {idx_path.name}")

        b1b4_per = len(_METHOD_CLASSES) * 4   # B1–B4 × pvs
        print(f"  Subtotal: {dataset_total}  ({n}×{b1b4_per} B1-B4)")

    print(f"\n{'='*60}")
    print("REQUEST COUNT SUMMARY — assessing_phi_hat MALLS")
    print(f"{'='*60}")
    n_inst     = 100
    b1b4_per   = len(_METHOD_CLASSES) * 4
    n_datasets = len(args.datasets)
    print(f"  Instances per dataset:    {n_inst}")
    print(f"  Dataset configs:          {n_datasets}  {args.datasets}")
    print(f"  B1–B4 per instance:       {b1b4_per}  (4 methods × 4 pvs)")
    print(f"  Per-dataset total:        {n_inst * b1b4_per}")
    print(f"  Grand total:              {grand_total}")

    if not args.submit:
        print("\n--submit not set: files written, nothing submitted to OpenAI.")
        print("When ready: python pipeline2/prepare_malls_assess_phi_hat_batch.py --submit")


if __name__ == "__main__":
    main()
