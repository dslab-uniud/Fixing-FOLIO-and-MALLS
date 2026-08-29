"""run_gemma_assess_phi_hat.py — Send Gemma assessing_phi_hat requests to vLLM server.

Sends all batch request files in pipeline2/assessing_phi_hat/requests/gemma/
to the local vLLM server (Gemma-4-31B-it via SSH tunnel on port 8001).

Resume-safe: skips already-completed custom_ids.
Methods: B1-B4.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as _http

_ROOT = Path(__file__).parent.parent
_REQ_DIR = _ROOT / "pipeline2" / "assessing_phi_hat" / "requests" / "gemma"
_RESP_DIR = _ROOT / "pipeline2" / "assessing_phi_hat" / "responses" / "gemma"
_RESP_DIR.mkdir(parents=True, exist_ok=True)

_BASE_URL = "http://localhost:8001"
_API_KEY = "mm_vllm_student"
_RETRY_ATTEMPTS = 3

_thread_local = threading.local()


def _get_session():
    if not hasattr(_thread_local, "session"):
        s = _http.Session()
        s.headers.update({
            "Authorization": f"Bearer {_API_KEY}",
            "Content-Type": "application/json",
        })
        _thread_local.session = s
    return _thread_local.session


def _send_one(entry: dict) -> dict:
    session = _get_session()
    url = _BASE_URL + entry["url"]
    result = {"custom_id": entry.get("custom_id"), "response": None, "error": None}
    last_exc = None

    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = session.request(
                entry.get("method", "POST").upper(),
                url, json=entry["body"], timeout=600,
            )
            try:
                body = resp.json()
            except Exception:
                body = {"raw_text": resp.text}

            if resp.status_code == 200:
                result["response"] = body
                return result
            else:
                result["error"] = f"HTTP {resp.status_code}: {str(body)[:200]}"
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                return result
        except Exception as e:
            last_exc = str(e)
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
                continue

    result["error"] = f"Exception after retries: {last_exc}"
    return result


def _load_done_ids(resp_file: Path) -> set:
    """Load custom_ids already completed in response file."""
    done = set()
    if resp_file.exists():
        with open(resp_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("response") and not rec.get("error"):
                        done.add(rec["custom_id"])
                except:
                    pass
    return done


def _process_file(req_file: Path, workers: int):
    resp_file = _RESP_DIR / req_file.name

    # Load all requests
    requests_list = []
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line:
                requests_list.append(json.loads(line))

    # Skip already-done
    done_ids = _load_done_ids(resp_file)
    pending = [r for r in requests_list if r["custom_id"] not in done_ids]

    if not pending:
        print(f"  [{req_file.name}] All {len(requests_list)} done ✓")
        return

    print(f"  [{req_file.name}] {len(pending)}/{len(requests_list)} pending (workers={workers})")

    # Open response file in append mode
    write_lock = threading.Lock()
    completed = 0
    errors = 0
    start_time = time.time()

    def _write_result(result):
        nonlocal completed, errors
        with write_lock:
            with open(resp_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            completed += 1
            if result.get("error"):
                errors += 1

            # Progress every 50 requests
            if completed % 50 == 0 or completed == len(pending):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta_remaining = (len(pending) - completed) / rate if rate > 0 else 0
                print(f"  [{req_file.name}] {completed}/{len(pending)} ({errors} err, {rate:.2f} req/s, ETA: {eta_remaining/60:.1f}min)")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_send_one, req) for req in pending]
        for future in as_completed(futures):
            result = future.result()
            _write_result(result)

    elapsed = time.time() - start_time
    print(f"  [{req_file.name}] DONE: {completed} processed ({errors} errors) in {elapsed/60:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pattern", default="*.jsonl", help="File pattern to filter")
    parser.add_argument("--list", action="store_true", help="List files only")
    args = parser.parse_args()

    # Find request files
    req_files = sorted(_REQ_DIR.glob(args.pattern))

    print(f"Gemma Assessing Phi_hat Runner")
    print(f"  vLLM server: {_BASE_URL}")
    print(f"  Workers: {args.workers}")
    print(f"  Request files: {len(req_files)}")

    if args.list:
        for f in req_files:
            with open(f) as file:
                n = sum(1 for _ in file)
            print(f"    {f.name}: {n} requests")
        return

    # Process files sequentially (each file uses its own thread pool)
    print(f"\n{'='*70}")
    overall_start = time.time()

    for req_file in req_files:
        _process_file(req_file, args.workers)

    total_time = (time.time() - overall_start) / 60
    print(f"\n{'='*70}")
    print(f"All done in {total_time:.1f} minutes")


if __name__ == "__main__":
    main()
