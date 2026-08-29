"""run_qwen_malls.py — Send MALLS pipeline1 Qwen3 judge requests to the vLLM server.

Handles B1–B4 with full parallelism:
  1. File-level: all methods submitted simultaneously.
  2. Request-level: each method sends up to --workers concurrent HTTP requests.

Requests are built inline from the MALLS dataset and per-instance ontology
(core/dataset/datasets/malls/signature.jsonl).

Prerequisites
-------------
  1. VPN connected.
  2. SSH tunnel open on port 8002:
       ssh -N -L 8002:127.0.0.1:4242 <remote-server> &

Usage
-----
    python pipeline1/run_qwen_malls.py                    # all methods
    python pipeline1/run_qwen_malls.py --workers 16       # more concurrency
    python pipeline1/run_qwen_malls.py --methods B1 B2    # subset
    python pipeline1/run_qwen_malls.py --dry-run

Response files
--------------
    pipeline1/assessing_original/responses/qwen/
        malls_{method}.jsonl          # B1–B4

Custom-ID format
----------------
    {method}|pv{pv}|curation_malls_{instance_id}

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

_RESP_DIR = _ROOT / "pipeline1" / "assessing_original" / "responses" / "qwen"

_DATASET  = _ROOT / "core" / "dataset" / "datasets" / "malls" / "dataset_rawpairs.jsonl"
_SIGNATURE = _ROOT / "core" / "dataset" / "datasets" / "malls" / "signature.jsonl"

_METHOD_CLASSES = {
    "B1": B1_ZeroShotCoT,
    "B2": B2_FewShot,
    "B3": B3_FewShotCoT,
    "B4": B4_FeatureBased,
}
_ALL_METHODS      = ["B1", "B2", "B3", "B4"]
_PVS              = [1, 2, 3, 4]
_SPLIT            = "malls"

_DEFAULT_BASE_URL = "http://localhost:8002"
_DEFAULT_API_KEY  = "mm_vllm_student"
_DEFAULT_MODEL    = "Qwen/Qwen3-30B-A3B"
_DEFAULT_WORKERS  = 16
_MAX_TOKENS       = 2000

_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_multiline_json(path: Path) -> list[dict]:
    """Load a file of concatenated JSON objects (supports both bare-concat and array-style)."""
    decoder = json.JSONDecoder(strict=False)
    content = path.read_text(encoding="utf-8").strip()
    records: list[dict] = []
    idx = 0
    while idx < len(content):
        while idx < len(content) and (content[idx].isspace() or content[idx] in ",[]"):
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error in {path} near pos {idx}: {exc}") from exc
    return records


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
    session = _get_session(base_url, api_key)
    url     = session.base_url + entry["url"]
    result: dict = {"custom_id": entry.get("custom_id"), "response": None, "error": None}
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
        else:
            result["error"] = {"status_code": resp.status_code, "body": payload}
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# File helpers
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
# Core parallel sender
# ---------------------------------------------------------------------------

def _run_entries_parallel(
    entries:   list[dict],
    resp_path: Path,
    label:     str,
    base_url:  str,
    api_key:   str,
    dry_run:   bool,
    workers:   int,
) -> int:
    """Send all entries using a thread pool. Returns number of successful responses."""
    done_ids = _load_done_ids(resp_path)
    to_send  = [e for e in entries if e.get("custom_id") not in done_ids]

    n_total = len(entries)
    n_done  = len(done_ids)
    n_todo  = len(to_send)

    if n_todo == 0:
        print(f"  [{label}] all {n_total} done ✓")
        return 0

    print(f"  [{label}] {n_todo}/{n_total} pending  (done: {n_done}, workers: {workers})")

    if dry_run:
        for e in to_send[:2]:
            print(f"    DRY-RUN: {e.get('custom_id', '?')[:90]}")
        if n_todo > 2:
            print(f"    ... and {n_todo - 2} more")
        return 0

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
    return counter["ok"]


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def _build_b1b4_entries(records: list[dict], method_name: str,
                         model: str) -> list[dict]:
    """Build B1–B4 request entries for one method across all pvs."""
    cls             = _METHOD_CLASSES[method_name]
    response_format = _schema_for(method_name)
    entries: list[dict] = []

    for pv in _PVS:
        method = cls(llm=None, configuration="B", prompt_version=pv)
        system = method._build_system_prompt()
        for rec in records:
            instance_id = rec["id"]
            nl          = rec.get("NL_sentence", "")
            phi_hat     = rec.get("FOL_sentence_old", "").strip()
            if not phi_hat:
                continue
            ontology = build_malls_ontology_block(instance_id)
            u_msg    = user_message(nl, phi_hat, ontology_block=ontology, version=pv)
            cid      = f"{method_name}|pv{pv}|curation_{_SPLIT}_{instance_id}"
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send MALLS pipeline1 Qwen3 judge requests (B1–B4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--api-key",  default=_DEFAULT_API_KEY)
    parser.add_argument("--model",    default=_DEFAULT_MODEL)
    parser.add_argument("--workers",  type=int, default=_DEFAULT_WORKERS,
                        help=f"Concurrent HTTP requests (default: {_DEFAULT_WORKERS}).")
    parser.add_argument("--methods",  nargs="+", default=_ALL_METHODS,
                        choices=_ALL_METHODS,
                        help="Methods to run (default: all B1–B4).")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    base_url = args.base_url
    api_key  = args.api_key
    model    = args.model
    workers  = args.workers

    print(f"vLLM server : {base_url}")
    print(f"Model       : {model}")
    print(f"Methods     : {args.methods}")
    print(f"Workers     : {workers} concurrent requests")
    if args.dry_run:
        print("DRY-RUN mode\n")

    records = _load_multiline_json(_DATASET)
    print(f"\n  Loaded {len(records)} MALLS instances")

    print(f"\n{'='*60}")
    print(f"Methods: {args.methods}")

    def run_single(method: str) -> None:
        entries   = _build_b1b4_entries(records, method, model)
        resp_path = _RESP_DIR / f"malls_{method}.jsonl"
        _run_entries_parallel(entries, resp_path, method,
                              base_url, api_key, args.dry_run, workers)

    with ThreadPoolExecutor(max_workers=len(args.methods)) as outer:
        futures = {outer.submit(run_single, m): m for m in args.methods}
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"  [{m}] EXCEPTION: {exc}")

    print(f"\n{'='*60}")
    print("All done.")
    print(f"Response files: {_RESP_DIR}/malls_*")


if __name__ == "__main__":
    main()
