"""run_gemma_folio.py — Send FOLIO pipeline1 Gemma4 judge requests to the vLLM server.

Runs B1–B4, Task B only, all 4 prompt variants, for the full FOLIO dataset.

All (method, pv) combos share a thread pool of --workers concurrent HTTP requests.

Prerequisites
-------------
  1. VPN connected.
  2. SSH tunnel open on port 8001:
       ssh -N -L 8001:127.0.0.1:8000 <remote-server> &

Usage
-----
    python pipeline1/run_gemma_folio.py                        # all methods
    python pipeline1/run_gemma_folio.py --workers 4            # more concurrency
    python pipeline1/run_gemma_folio.py --methods B1 B2        # subset of methods
    python pipeline1/run_gemma_folio.py --dry-run

Response files
--------------
    pipeline1/assessing_original/responses/gemma/
        folio_{method}.jsonl    # e.g. folio_B1.jsonl

Resume-safe: already-completed custom_ids are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as _http

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.strategies.strategy import (
    B1_ZeroShotCoT, B2_FewShot, B3_FewShotCoT, B4_FeatureBased,
    build_folio_ontology_block,
)
from core.strategies.prompts.common import user_message
from pipeline1.batch.config import get_schema_for_method as _schema_for

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESP_DIR = _ROOT / "pipeline1" / "assessing_original" / "responses" / "gemma"

_DATASET_PATH: Path = _ROOT / "core" / "dataset" / "datasets" / "folio" / "dataset_rawpairs.jsonl"

_METHOD_CLASSES = {
    "B1": B1_ZeroShotCoT,
    "B2": B2_FewShot,
    "B3": B3_FewShotCoT,
    "B4": B4_FeatureBased,
}
_ALL_METHODS  = ["B1", "B2", "B3", "B4"]
_PVS          = [1, 2, 3, 4]

_DEFAULT_BASE_URL = "http://localhost:8001"
_DEFAULT_API_KEY  = "mm_vllm_student"
_DEFAULT_MODEL    = "google/gemma-4-31B-it"
_DEFAULT_WORKERS  = 4
_MAX_TOKENS       = 8000
_RETRY_ATTEMPTS   = 3

_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Dataset loading (multi-line JSON)
# ---------------------------------------------------------------------------

def _load_raw_pairs(path: Path) -> list[dict]:
    """Load multi-line concatenated JSON records from a FOLIO rawpairs file."""
    records: list[dict] = []
    decoder = json.JSONDecoder(strict=False)
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
# Request building
# ---------------------------------------------------------------------------

def _build_entries(records: list[dict], method_name: str, model: str) -> list[dict]:
    """Build all vLLM request entries for one method combo across all pvs."""
    cls = _METHOD_CLASSES[method_name]
    response_format = _schema_for(method_name)
    entries: list[dict] = []

    for pv in _PVS:
        method = cls(llm=None, configuration="B", prompt_version=pv)
        system = method._build_system_prompt()
        for rec in records:
            instance_id = rec["id"]
            nl          = rec["NL_sentence"]
            phi_hat     = rec.get("FOL_sentence_old", "")
            if not phi_hat:
                continue   # skip records without a candidate formula

            ontology = build_folio_ontology_block(instance_id)
            u_msg    = user_message(nl, phi_hat, ontology_block=ontology, version=pv)
            cid      = f"{method_name}|pv{pv}|curation_{instance_id}"

            entries.append({
                "custom_id": cid,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":           model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": u_msg},
                    ],
                    "temperature":     0.0,
                    "max_tokens":      _MAX_TOKENS,
                    "response_format": response_format,
                    "logprobs":        True,
                    "top_logprobs":    5,
                },
            })

    return entries


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_session(base_url: str, api_key: str) -> _http.Session:
    if not hasattr(_thread_local, "session"):
        s = _http.Session()
        s.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        })
        s.base_url = base_url.rstrip("/")
        _thread_local.session = s
    return _thread_local.session


def _send_one(entry: dict, base_url: str, api_key: str) -> dict:
    import time as _time
    session = _get_session(base_url, api_key)
    url     = session.base_url + entry["url"]
    result  = {"custom_id": entry.get("custom_id"), "response": None, "error": None}
    last_exc = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = session.request(
                entry.get("method", "POST").upper(),
                url, json=entry["body"], timeout=3600,
            )
            try:
                payload = resp.json()
            except Exception:
                payload = {"text": resp.text}
            if resp.ok:
                result["response"] = {"status_code": resp.status_code, "body": payload}
                return result
            else:
                result["error"] = {"status_code": resp.status_code, "body": payload}
                return result
        except Exception as exc:
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                _time.sleep(0.5)
                if hasattr(_thread_local, "session"):
                    del _thread_local.session
    result["error"] = str(last_exc)
    return result


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec     = json.loads(line)
                if rec.get("error"):
                    continue
                choices = rec.get("response", {}).get("body", {}).get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                if content:
                    done.add(rec.get("custom_id", ""))
            except json.JSONDecodeError:
                pass
    return done


# ---------------------------------------------------------------------------
# Parallel sender for one method file
# ---------------------------------------------------------------------------

def _run_combo(method: str, records: list[dict],
               base_url: str, api_key: str, model: str,
               workers: int, dry_run: bool) -> tuple[int, int]:
    """Build and send all requests for one method combo. Returns (ok, err)."""
    resp_path = _RESP_DIR / f"folio_{method}.jsonl"
    label     = method

    all_entries = _build_entries(records, method, model)
    done_ids    = _load_done_ids(resp_path)
    to_send     = [e for e in all_entries if e.get("custom_id") not in done_ids]

    n_total = len(all_entries)
    n_done  = len(done_ids)
    n_todo  = len(to_send)

    if n_todo == 0:
        print(f"  [{label}] all {n_total} done ✓")
        return 0, 0

    print(f"  [{label}] {n_todo}/{n_total} pending  (done: {n_done}, workers: {workers})")

    if dry_run:
        for e in to_send[:2]:
            print(f"    DRY-RUN: {e.get('custom_id', '?')[:90]}")
        if n_todo > 2:
            print(f"    ... and {n_todo - 2} more")
        return 0, 0

    resp_path.parent.mkdir(parents=True, exist_ok=True)
    file_lock = threading.Lock()
    counter   = {"ok": 0, "err": 0, "done": 0}

    def process(entry: dict) -> None:
        result = _send_one(entry, base_url, api_key)
        with file_lock:
            with open(resp_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            counter["done"] += 1
            if result["error"]:
                counter["err"] += 1
                print(f"  [{label}] [{counter['done']}/{n_todo}] "
                      f"ERROR {entry.get('custom_id','?')[:60]}: {result['error']}")
            else:
                counter["ok"] += 1
                if counter["done"] % 100 == 0 or counter["done"] == n_todo:
                    print(f"  [{label}] [{counter['done']}/{n_todo}] "
                          f"ok={counter['ok']} err={counter['err']}")

    with ThreadPoolExecutor(max_workers=min(workers, n_todo)) as pool:
        list(pool.map(process, to_send))

    print(f"  [{label}] finished → {counter['ok']} ok, {counter['err']} errors")
    return counter["ok"], counter["err"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send FOLIO pipeline1 Gemma4 judge requests (B1–B4, Task B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--api-key",  default=_DEFAULT_API_KEY)
    parser.add_argument("--model",    default=_DEFAULT_MODEL)
    parser.add_argument("--workers",  type=int, default=_DEFAULT_WORKERS,
                        help=f"Concurrent HTTP requests per method (default: {_DEFAULT_WORKERS}).")
    parser.add_argument("--methods",  nargs="+", default=_ALL_METHODS,
                        choices=_ALL_METHODS,
                        help="Methods to run (default: B1 B2 B3 B4).")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    print(f"vLLM server : {args.base_url}")
    print(f"Model       : {args.model}")
    print(f"Methods     : {args.methods}")
    print(f"Workers     : {args.workers} concurrent requests per method")
    if args.dry_run:
        print("DRY-RUN mode")
    print()

    # Load dataset
    records = _load_raw_pairs(_DATASET_PATH)
    print(f"  Loaded: {len(records)} records")

    # Calculate total request count
    total_requests = sum(
        len(_build_entries(records, method, args.model))
        for method in args.methods
    )
    print(f"\n  Total requests across all combos: {total_requests}")
    print(f"  (= {len(args.methods)} methods × {len(_PVS)} pvs × instances)\n")

    print(f"{'='*60}")
    print(f"Running {len(args.methods)} combos in parallel…")
    print(f"{'='*60}")

    total_ok = 0
    total_err = 0

    def run_one(method):
        return _run_combo(method, records,
                          args.base_url, args.api_key, args.model,
                          args.workers, args.dry_run)

    with ThreadPoolExecutor(max_workers=len(args.methods)) as outer:
        futures = {outer.submit(run_one, m): m for m in args.methods}
        for fut in as_completed(futures):
            method = futures[fut]
            try:
                ok, err = fut.result()
                total_ok  += ok
                total_err += err
            except Exception as exc:
                print(f"  ERROR in {method}: {exc}")

    print(f"\n{'='*60}")
    print(f"All done. Total: {total_ok} ok, {total_err} errors")
    print(f"Response files: {_RESP_DIR}/folio_*.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
