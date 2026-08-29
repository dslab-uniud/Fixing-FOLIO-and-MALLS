"""run_gemma_malls.py — Send MALLS pipeline1 Gemma4 judge requests to the vLLM server.

Runs B1–B4, Task B only, all 4 prompt variants, for the MALLS dataset (100 instances).
Total requests: 100 × 4 methods × 4 pvs = 1600.

Each request includes the per-instance ontology block from
core/dataset/datasets/malls/signature.jsonl (same Rel/Const format as FOLIO).

Prerequisites
-------------
  1. VPN connected.
  2. SSH tunnel open on port 8001:
       ssh -N -L 8001:127.0.0.1:8000 <remote-server> &

Usage
-----
    python pipeline1/run_gemma_malls.py                  # all methods
    python pipeline1/run_gemma_malls.py --workers 3      # default: 3 workers
    python pipeline1/run_gemma_malls.py --methods B1 B2  # subset
    python pipeline1/run_gemma_malls.py --dry-run

Response files
--------------
    pipeline1/assessing_original/responses/gemma/malls_B{1-4}.jsonl

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

from core.strategies.strategy import B1_ZeroShotCoT, B2_FewShot, B3_FewShotCoT, B4_FeatureBased
from core.strategies.prompts.common import user_message
from core.dataset.signature_loader import format_folio_vocabulary
from pipeline1.batch.config import get_schema_for_method as _schema_for

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESP_DIR    = _ROOT / "pipeline1" / "assessing_original" / "responses" / "gemma"
_DATASET     = _ROOT / "core" / "dataset" / "datasets" / "malls" / "dataset_rawpairs.jsonl"
_SIGNATURE   = _ROOT / "core" / "dataset" / "datasets" / "malls" / "signature.jsonl"

_METHOD_CLASSES = {
    "B1": B1_ZeroShotCoT,
    "B2": B2_FewShot,
    "B3": B3_FewShotCoT,
    "B4": B4_FeatureBased,
}
_ALL_METHODS  = ["B1", "B2", "B3", "B4"]
_PVS          = [1, 2, 3, 4]
_SPLIT        = "malls"

_DEFAULT_BASE_URL = "http://localhost:8001"
_DEFAULT_API_KEY  = "mm_vllm_student"
_DEFAULT_MODEL    = "google/gemma-4-31B-it"
_DEFAULT_WORKERS  = 3
_MAX_TOKENS       = 8000
_RETRY_ATTEMPTS   = 3

_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Dataset and signature loading
# ---------------------------------------------------------------------------

def _load_multiline_json(path: Path) -> list[dict]:
    """Load a file of concatenated JSON objects (supports both bare-concat and array-style)."""
    decoder = json.JSONDecoder(strict=False)
    content = path.read_text(encoding="utf-8").strip()
    records: list[dict] = []
    idx = 0
    while idx < len(content):
        # Skip whitespace, commas, and array brackets between objects
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


def _build_signature_map(signatures: list[dict]) -> dict[int, dict]:
    """Build {id → signature_dict} map."""
    return {s["id"]: s for s in signatures}


def _build_malls_ontology_block(sig: dict) -> str:
    """Build an ontology block string from a MALLS per-instance signature."""
    rels_str, consts_str = format_folio_vocabulary(sig)
    lines = [
        "### Ontology",
        "",
        "Relations:",
        rels_str,
        "",
        "Constants:",
        consts_str,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _build_entries(
    records: list[dict],
    sig_map: dict[int, dict],
    method_name: str,
    model: str,
) -> list[dict]:
    """Build all vLLM request entries for one method across all pvs."""
    cls = _METHOD_CLASSES[method_name]
    response_format = _schema_for(method_name)
    entries: list[dict] = []

    for pv in _PVS:
        method = cls(llm=None, configuration="B", prompt_version=pv)
        system = method._build_system_prompt()

        for rec in records:
            instance_id = str(rec["id"])
            nl          = rec.get("NL_sentence", "")
            phi_hat     = rec.get("FOL_sentence_old", "").strip()

            if not phi_hat:
                continue  # skip records without a candidate formula

            # Per-instance ontology from signature file
            sig = sig_map.get(rec["id"])
            if sig is None:
                print(f"  WARNING: no signature for id={rec['id']}, skipping")
                continue
            ontology_block = _build_malls_ontology_block(sig)

            u_msg = user_message(nl, phi_hat, ontology_block=ontology_block, version=pv)
            cid   = f"{method_name}|pv{pv}|curation_{_SPLIT}_{instance_id}"

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
    session  = _get_session(base_url, api_key)
    url      = session.base_url + entry["url"]
    result   = {"custom_id": entry.get("custom_id"), "response": None, "error": None}
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
# Parallel sender for one method
# ---------------------------------------------------------------------------

def _run_method(
    method: str,
    records: list[dict],
    sig_map: dict[int, dict],
    base_url: str,
    api_key: str,
    model: str,
    workers: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Build and send all requests for one method. Returns (ok, err)."""
    resp_path = _RESP_DIR / f"malls_{method}.jsonl"
    label     = f"{_SPLIT}/{method}"

    all_entries = _build_entries(records, sig_map, method, model)
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
            # Show ontology snippet
            msgs = e["body"]["messages"]
            u = msgs[1]["content"] if len(msgs) > 1 else ""
            ont_start = u.find("### Ontology")
            if ont_start != -1:
                print(f"    Ontology preview: {u[ont_start:ont_start+120].replace(chr(10), ' ')}")
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
        description="Send MALLS pipeline1 Gemma4 judge requests (B1–B4, Task B).",
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
                        help="Methods to run (default: B1 B2 B3 B4).")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    print(f"vLLM server : {args.base_url}")
    print(f"Model       : {args.model}")
    print(f"Methods     : {args.methods}  pvs: {_PVS}")
    print(f"Workers     : {args.workers} concurrent requests")
    if args.dry_run:
        print("DRY-RUN mode")
    print()

    # Load dataset and signatures
    records    = _load_multiline_json(_DATASET)
    signatures = _load_multiline_json(_SIGNATURE)
    sig_map    = _build_signature_map(signatures)

    print(f"  Loaded {len(records)} MALLS instances")
    print(f"  Loaded {len(signatures)} ontology signatures")

    # Verify all instances have a signature
    missing = [r["id"] for r in records if r["id"] not in sig_map]
    if missing:
        print(f"  WARNING: {len(missing)} instances missing signature: {missing}")

    total_requests = sum(
        len(_build_entries(records, sig_map, m, args.model))
        for m in args.methods
    )
    print(f"\n  Total requests: {total_requests}")
    print(f"  (= {len(args.methods)} methods × {len(_PVS)} pvs × instances)\n")

    # Run all methods in parallel
    combos = [(m, records, sig_map) for m in args.methods]

    print(f"{'='*60}")
    print(f"Running {len(combos)} methods in parallel…")
    print(f"{'='*60}")

    total_ok = total_err = 0

    def run_one(combo):
        m, recs, smap = combo
        return _run_method(m, recs, smap, args.base_url, args.api_key,
                           args.model, args.workers, args.dry_run)

    with ThreadPoolExecutor(max_workers=len(combos)) as outer:
        futures = {outer.submit(run_one, c): c[0] for c in combos}
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
    print(f"Response files: {_RESP_DIR}/malls_*.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
