"""run_qwen_assess.py — Qwen judge for Qwen phi_hat generations (pipeline2).

Runs B1–B4 prompting strategies for:
  - folio_qwen3_P1  (B1–B4)
  - malls_qwen3_P1  (B1–B4)

All results are written directly to:
  pipeline2/assessing_phi_hat/results/{B1..B4}/{dataset}/qwen3/results.jsonl
(all prompt variants and few-shot counts merged into one file, distinguished
by the `hyperparams` row field)

Resume-safe: existing result rows (by id) are skipped.

Prerequisites
-------------
  SSH tunnel to Qwen vLLM server open on port 8002:
      ssh -N -L 8002:127.0.0.1:4242 <remote-server> &

Usage
-----
    python pipeline2/run_qwen_assess.py [--workers 16] [--dry-run]
    python pipeline2/run_qwen_assess.py --datasets folio_qwen3_P1  # subset
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests as _http

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.strategies.strategy import (
    B1_ZeroShotCoT,
    B2_FewShot,
    B3_FewShotCoT,
    B4_FeatureBased,
    build_folio_ontology_block,
    build_malls_ontology_block,
)
from core.strategies.prompts.common import user_message
from pipeline1.batch.config import (
    SCHEMA_B as _SCHEMA_B,
    SCHEMA_B_FEAT as _SCHEMA_B_FEAT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QWEN_BASE_URL   = "http://localhost:8002"
_QWEN_API_KEY    = "mm_vllm_student"
_QWEN_MODEL_FULL = "Qwen/Qwen3-30B-A3B"
_QWEN_MAX_TOKENS = 2000
_QWEN_TEMP       = 0.0

_PVS             = (1, 2, 3, 4)
_DEFAULT_WORKERS = 16

_PHI_HAT_DIR = _ROOT / "pipeline2" / "phi_hat_generation" / "results"
_RESULTS_DIR = _ROOT / "pipeline2" / "assessing_phi_hat" / "results"

# Datasets: name → {phi_hat_file, source, run_b1b4}
_ALL_DATASETS = {
    "folio_qwen3_P1": {
        "phi_hat_file": "folio__Qwen3-30B-A3B__P1.jsonl",
        "source": "folio",
        "run_b1b4": True,
    },
    "malls_qwen3_P1": {
        "phi_hat_file": "malls__Qwen3-30B-A3B__P1.jsonl",
        "source": "malls",
        "run_b1b4": True,
    },
}

_B1B4_CLASSES = {
    "B1": B1_ZeroShotCoT,
    "B2": B2_FewShot,
    "B3": B3_FewShotCoT,
    "B4": B4_FeatureBased,
}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_instances(dataset_cfg: dict) -> list[dict]:
    path = _PHI_HAT_DIR / dataset_cfg["phi_hat_file"]
    instances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def _ontology_block(inst: dict) -> str:
    source = inst.get("source_dataset")
    if source == "malls":
        return build_malls_ontology_block(int(inst["id_source_dataset"]))
    return build_folio_ontology_block(inst["id_source_dataset"])


def _base_id(inst: dict) -> str:
    """Extract base instance id (e.g. 'story_380' or 'ex_1.4.1') from full id."""
    parts = inst["id"].split("__")
    return parts[1] if len(parts) >= 2 else inst["id"]


# ---------------------------------------------------------------------------
# Result file helpers
# ---------------------------------------------------------------------------

def _result_file(method: str, dataset: str) -> Path:
    return _RESULTS_DIR / method / dataset / "qwen3" / "results.jsonl"


def _done_key(row: dict) -> tuple:
    hp = row.get("hyperparams") or {}
    return (row.get("id"), hp.get("prompt_version"), hp.get("n_examples"))


def _load_done_keys(result_file: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not result_file.exists():
        return done
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(_done_key(json.loads(line)))
                except Exception:
                    pass
    return done


def _append_rows(result_file: Path, rows: list[dict], done_keys: set[tuple],
                 lock: threading.Lock, dry_run: bool) -> int:
    """Append rows not already in result_file. Returns count appended."""
    new_rows = [r for r in rows if _done_key(r) not in done_keys]
    if not new_rows:
        return 0
    if dry_run:
        return len(new_rows)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with open(result_file, "a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new_rows)


# ---------------------------------------------------------------------------
# Logprob extraction
# ---------------------------------------------------------------------------

def _extract_logprobs(raw_logprobs: list | None,
                      quality_score: str | None) -> tuple[dict, Optional[bool]]:
    empty = {"yes": None, "no": None, "?": None}
    if not raw_logprobs:
        return empty, None
    quality_idx = None
    for i, entry in enumerate(raw_logprobs):
        tok = entry.get("token", "").strip().strip('"').lower()
        if tok == "quality":
            quality_idx = i
            break
    if quality_idx is None:
        return empty, None
    for offset in (3, 2):
        idx = quality_idx + offset
        if idx >= len(raw_logprobs):
            continue
        entry = raw_logprobs[idx]
        val_tok = entry.get("token", "").strip().strip('"').lower()
        if val_tok in ("yes", "no", "?"):
            top_lps = entry.get("top_logprobs", [])
            lp_map: dict = {"yes": None, "no": None, "?": None}
            best_lp = -math.inf
            best_tok = None
            for lp_entry in top_lps:
                t = lp_entry.get("token", "").strip().strip('"').lower()
                lp = lp_entry.get("logprob")
                if t in lp_map and lp_map[t] is None:
                    lp_map[t] = lp
                if lp is not None and lp > best_lp:
                    best_lp = lp
                    best_tok = t
            consistent = (best_tok == quality_score) if (best_tok and quality_score) else None
            return lp_map, consistent
    return empty, None


def _compute_eq_i(refinement: Optional[str], phi_gold: str) -> Optional[bool]:
    if not refinement:
        return None
    try:
        from core.formalism.fol.fol import FOL
        from core.formalism.fol.solver_z3 import FOLSolver
        return bool(FOLSolver().are_equivalent(FOL(refinement), FOL(phi_gold)))
    except Exception:
        return None


def _parse_response(resp_body: dict) -> tuple[Optional[str], Optional[str],
                                               Optional[str], Optional[str], Optional[list]]:
    """Extract (quality_score, reasoning, comment, refinement, raw_logprobs)."""
    choices = resp_body.get("choices") or []
    if not choices:
        return None, None, None, None, None
    choice = choices[0]
    content_str = choice.get("message", {}).get("content")
    raw_logprobs = choice.get("logprobs", {}).get("content") if choice.get("logprobs") else None
    try:
        content = json.loads(content_str)
        return (
            content.get("quality_score"),
            content.get("reasoning"),
            content.get("comment"),
            content.get("refinement"),
            raw_logprobs,
        )
    except Exception:
        return None, None, None, None, None


def _make_row(inst: dict, method: str,
              hyperparams: dict, quality_score: Optional[str],
              reasoning: Optional[str], comment: Optional[str],
              refinement: Optional[str], raw_logprobs: Optional[list]) -> dict:
    lp, consistent = _extract_logprobs(raw_logprobs, quality_score)
    eq_i = _compute_eq_i(refinement, inst["phi_gold"])
    return {
        "id":                  _base_id(inst),
        "method":              method,
        "task":                "B",
        "hyperparams":         hyperparams,
        "phi_model":           "qwen3",
        "phi_prompt":          "P1",
        "ground_truth_label":  inst["label"],
        "quality_score":       quality_score,
        "reasoning":           reasoning,
        "comment":             comment,
        "refinement":          refinement,
        "eq_i":                eq_i,
        "logprobs":            lp,
        "logprobs_consistent": consistent,
    }


# ---------------------------------------------------------------------------
# HTTP sender (thread-safe)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_session() -> _http.Session:
    if not hasattr(_thread_local, "session"):
        s = _http.Session()
        s.headers.update({
            "Authorization": f"Bearer {_QWEN_API_KEY}",
            "Content-Type": "application/json",
        })
        _thread_local.session = s
    return _thread_local.session


def _send_one(body: dict) -> tuple[dict | None, str | None]:
    """Send one request. Returns (response_body, error_str)."""
    session = _get_session()
    url = _QWEN_BASE_URL.rstrip("/") + "/v1/chat/completions"
    try:
        resp = session.post(url, json=body, timeout=600)
        if resp.ok:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return None, str(exc)


def _send_batch_concurrent(
    tasks: list[tuple],   # list of (key, body)
    workers: int,
    label: str,
    dry_run: bool,
) -> dict:
    """
    Send a batch of requests concurrently.
    Returns {key: (response_body | None, error_str | None)}.
    """
    results: dict = {}
    lock = threading.Lock()
    counter = {"done": 0, "ok": 0, "err": 0}
    n = len(tasks)

    if dry_run:
        print(f"    [DRY-RUN] {label}: would send {n} requests")
        return {key: (None, "dry-run") for key, _ in tasks}

    def process(task):
        key, body = task
        resp_body, err = _send_one(body)
        with lock:
            results[key] = (resp_body, err)
            counter["done"] += 1
            if err:
                counter["err"] += 1
            else:
                counter["ok"] += 1
            if counter["done"] % 100 == 0 or counter["done"] == n:
                print(f"    [{label}] [{counter['done']}/{n}] "
                      f"ok={counter['ok']} err={counter['err']}")

    with ThreadPoolExecutor(max_workers=min(workers, n)) as pool:
        list(pool.map(process, tasks))

    return results


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def _build_b1b4_body(inst: dict, cls, pv: int) -> dict:
    method_obj = cls(llm=None, configuration="B", prompt_version=pv)
    system = method_obj._build_system_prompt()
    u_msg = user_message(inst["p"], inst["phi_hat"],
                         ontology_block=_ontology_block(inst), version=pv)
    return {
        "model":           _QWEN_MODEL_FULL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": u_msg},
        ],
        "temperature":     _QWEN_TEMP,
        "max_tokens":      _QWEN_MAX_TOKENS,
        "response_format": _SCHEMA_B_FEAT if cls is B4_FeatureBased else _SCHEMA_B,
        "logprobs":        True,
        "top_logprobs":    5,
    }


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_b1b4(instances: list[dict], dataset: str, workers: int, dry_run: bool) -> None:
    """Run B1–B4 for one dataset."""
    print(f"\n  --- B1–B4 [{dataset}] ---")
    file_locks: dict[str, threading.Lock] = {}

    for method, cls in _B1B4_CLASSES.items():
        rfile = _result_file(method, dataset)
        lock_key = str(rfile)
        if lock_key not in file_locks:
            file_locks[lock_key] = threading.Lock()

        for pv in _PVS:
            hyperparams = {"prompt_version": pv}
            if method in ("B2", "B3"):
                hyperparams["n_examples"] = 5

            done_keys = _load_done_keys(rfile)
            todo = [inst for inst in instances
                    if (_base_id(inst), pv, hyperparams.get("n_examples")) not in done_keys]
            if not todo:
                print(f"    [{method} pv{pv}] all {len(instances)} done ✓")
                continue

            label = f"{method}/pv{pv}"
            print(f"    [{label}] {len(todo)}/{len(instances)} pending")
            tasks = [(inst["id"], _build_b1b4_body(inst, cls, pv)) for inst in todo]
            resp_map = _send_batch_concurrent(tasks, workers, label, dry_run)

            rows = []
            for inst in todo:
                resp_body, err = resp_map.get(inst["id"], (None, "missing"))
                if err or resp_body is None:
                    continue
                qs, reasoning, comment, refinement, raw_lp = _parse_response(resp_body)
                rows.append(_make_row(inst, method, dict(hyperparams),
                                      qs, reasoning, comment, refinement, raw_lp))
            n = _append_rows(rfile, rows, done_keys, file_locks[lock_key], dry_run)
            print(f"    [{label}] wrote {n} rows → {rfile.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Server check
# ---------------------------------------------------------------------------

def _wait_for_server(timeout_s: int = 300, poll_s: int = 10) -> bool:
    import time
    url = _QWEN_BASE_URL.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {_QWEN_API_KEY}"}
    deadline = time.time() + timeout_s
    print(f"  Checking Qwen server at {_QWEN_BASE_URL} …")
    while time.time() < deadline:
        try:
            r = _http.get(url, headers=headers, timeout=10)
            if r.ok:
                models = [m["id"] for m in r.json().get("data", [])]
                print(f"  ✓ Qwen server ready — models: {models}")
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    print("  ✗ Qwen server not available after timeout")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen judge for Qwen phi_hat generations (pipeline2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS,
                        help=f"Concurrent HTTP workers (default: {_DEFAULT_WORKERS})")
    parser.add_argument("--datasets", nargs="+", default=list(_ALL_DATASETS.keys()),
                        choices=list(_ALL_DATASETS.keys()),
                        help="Datasets to run (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  Pipeline2 Qwen judge")
    print(f"  Workers : {args.workers}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Dry-run : {args.dry_run}")
    print("=" * 65)

    if not args.dry_run:
        if not _wait_for_server():
            raise SystemExit("Qwen server unavailable — ensure SSH tunnel is open on port 8002")

    dataset_instances: dict[str, list[dict]] = {}
    for ds_name in args.datasets:
        cfg = _ALL_DATASETS[ds_name]
        insts = _load_instances(cfg)
        dataset_instances[ds_name] = insts
        print(f"  Loaded {ds_name}: {len(insts)} instances")

    print(f"\n{'='*65}")
    print("PHASE B1–B4")
    print(f"{'='*65}")
    for ds_name in args.datasets:
        cfg = _ALL_DATASETS[ds_name]
        if not cfg["run_b1b4"]:
            print(f"  [{ds_name}] B1–B4 already done — skipping")
            continue
        run_b1b4(dataset_instances[ds_name], cfg["source"], args.workers, args.dry_run)

    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    header = f"  {'dataset':<30} {'method':<10} {'done':>6} {'total':>6}"
    print(header)
    print("  " + "-" * 68)

    for ds_name in args.datasets:
        n_inst = len(dataset_instances[ds_name])
        cfg = _ALL_DATASETS[ds_name]
        if not cfg["run_b1b4"]:
            continue
        for method in _B1B4_CLASSES:
            rfile = _result_file(method, cfg["source"])
            done_keys = _load_done_keys(rfile)
            n_pv = 4
            n_expected = n_inst * n_pv
            done = len(done_keys)
            status = "✓" if done == n_expected else "…"
            print(f"  {ds_name:<30} {method}/qwen3 {done:>6}/{n_expected:<6} {status}")

    print(f"\n{'='*65}")
    print("Done.")


if __name__ == "__main__":
    main()
