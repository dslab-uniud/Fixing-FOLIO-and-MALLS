"""run_gemma_folio_train.py — Run B2|B|pv1 Gemma judge on the *new* curated
folio_train set (340 stories + 1001 conclusions = 1341 rawpair records).

Inputs
------
  - core/Dataset_to_curate/folio_train_rawpairs.jsonl
      one-JSON-per-line records with id, NL_sentence, FOL_sentence
      (where id = "story_{sid}" or "concl_{sid}_{example_id}")
  - core/Dataset_to_curate/folio_train_signature.jsonl
      multi-JSON entries {story_id, Rel, Const} (the per-story ontology
      we just built via the LLM signature batch)

Behaviour
---------
For each rawpair record:
  - look up the signature for its story_id
  - build the ontology block (Relations + Constants, same shape as
    `build_folio_ontology_block`)
  - if FOL_sentence contains line breaks, wrap each non-empty line in `(...)`
    and join them with `∧` to produce a single closed FOL formula
  - render B2|B|pv1 system + user message via core/baselines
  - POST to the Gemma vLLM at localhost:8001 with logprobs

Output
------
  pipeline1/assessing_original/responses/gemma/
      folio_train_B2.jsonl   (one JSON line per request, resume-safe)

Resume-safe — already-completed custom_ids are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests as _http

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.strategies.strategy import B2_FewShot
from core.strategies.prompts.common import user_message
from core.dataset.signature_loader import format_folio_vocabulary
from pipeline1.batch.config import get_schema_for_method as _schema_for

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_RAWPAIRS = _ROOT / "core" / "Dataset_to_curate" / "folio_train_rawpairs.jsonl"
_SIG_FILE = _ROOT / "core" / "Dataset_to_curate" / "folio_train_signature.jsonl"
_RESP_DIR = _ROOT / "pipeline1" / "assessing_original" / "responses" / "gemma"

# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

_METHOD_NAME = "B2"
_PV          = 1
_SPLIT_TAG   = "train"   # used only in custom_id and output filename

_DEFAULT_BASE_URL = "http://localhost:8001"
_DEFAULT_API_KEY  = "mm_vllm_student"
_DEFAULT_MODEL    = "google/gemma-4-31B-it"
_DEFAULT_WORKERS  = 4
_MAX_TOKENS       = 13000
_RETRY_ATTEMPTS   = 3

_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Loaders for the new train rawpairs / signature
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load either one-JSON-per-line OR multi-line concatenated JSON. We try
    line-by-line first because both rawpairs and signature files we generated
    use that format separated by blank lines / no commas."""
    out: list[dict] = []
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(strict=False)
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\n\r,":
            idx += 1
        if idx >= n:
            break
        rec, end_idx = decoder.raw_decode(text, idx)
        out.append(rec)
        idx = end_idx
    return out


def _load_signatures() -> dict[int, dict]:
    return {rec["story_id"]: rec for rec in _load_jsonl(_SIG_FILE)}


def _story_id_from_id(rec_id: str) -> int:
    """story_380 → 380; concl_380_1 → 380."""
    m = re.match(r"^(story|concl)_(\d+)", rec_id)
    if not m:
        raise ValueError(f"Unrecognised id: {rec_id}")
    return int(m.group(2))


# ---------------------------------------------------------------------------
# Ontology / formula formatting
# ---------------------------------------------------------------------------

def _ontology_block(sig: dict) -> str:
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


def _conjoin_multiline_fol(fol: str) -> str:
    """If `fol` has multiple non-empty lines, wrap each in `(…)` and join with ` ∧ `.
    Otherwise return it unchanged (stripped)."""
    lines = [ln.strip() for ln in fol.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return lines[0] if lines else ""
    return " ∧ ".join(f"({ln})" for ln in lines)


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _build_entries(records: list[dict], sigs: dict[int, dict], model: str) -> list[dict]:
    method = B2_FewShot(llm=None, configuration="B", prompt_version=_PV)
    system = method._build_system_prompt()
    response_format = _schema_for(_METHOD_NAME)

    entries: list[dict] = []
    skipped = {"no_sig": 0, "no_fol": 0}
    for rec in records:
        rid = rec["id"]
        nl  = rec["NL_sentence"]
        fol_raw = rec.get("FOL_sentence", "") or ""
        if not fol_raw.strip():
            skipped["no_fol"] += 1
            continue
        sid = _story_id_from_id(rid)
        sig = sigs.get(sid)
        if sig is None:
            skipped["no_sig"] += 1
            continue

        phi_hat = _conjoin_multiline_fol(fol_raw)
        ontology = _ontology_block(sig)
        u_msg = user_message(nl, phi_hat, ontology_block=ontology, version=_PV)
        cid = f"{_METHOD_NAME}|pv{_PV}|curation_{_SPLIT_TAG}_{rid}"

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

    if skipped["no_sig"] or skipped["no_fol"]:
        print(f"  skipped: {skipped['no_sig']} no-sig, {skipped['no_fol']} empty-FOL")
    return entries


# ---------------------------------------------------------------------------
# HTTP
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
                rec = json.loads(line)
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Send B2|B|pv1 Gemma judge requests on folio_train rawpairs.")
    ap.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    ap.add_argument("--api-key",  default=_DEFAULT_API_KEY)
    ap.add_argument("--model",    default=_DEFAULT_MODEL)
    ap.add_argument("--workers",  type=int, default=_DEFAULT_WORKERS,
                    help=f"concurrent HTTP requests (default {_DEFAULT_WORKERS})")
    ap.add_argument("--limit",    type=int, default=None,
                    help="only send the first N pending requests (sanity check)")
    ap.add_argument("--inspect",  type=int, default=0,
                    help="print N rendered prompts then exit (no HTTP calls)")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    print(f"vLLM server : {args.base_url}")
    print(f"Model       : {args.model}")
    print(f"Method      : {_METHOD_NAME}|pv{_PV}")
    print(f"Workers     : {args.workers}")

    print("\nLoading dataset and signatures...")
    records = _load_jsonl(_RAWPAIRS)
    sigs    = _load_signatures()
    print(f"  rawpairs:   {len(records)}")
    print(f"  signatures: {len(sigs)} stories")

    entries = _build_entries(records, sigs, args.model)
    print(f"  built {len(entries)} requests\n")

    if args.inspect:
        for e in entries[: args.inspect]:
            print(f"=== {e['custom_id']} ===")
            print("--- system ---")
            print(e["body"]["messages"][0]["content"][:1200])
            print("\n--- user ---")
            print(e["body"]["messages"][1]["content"])
            print()
        return

    resp_path = _RESP_DIR / f"folio_{_SPLIT_TAG}_{_METHOD_NAME}.jsonl"
    resp_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(resp_path)
    to_send  = [e for e in entries if e["custom_id"] not in done_ids]
    if args.limit is not None:
        to_send = to_send[: args.limit]

    n_total, n_done, n_todo = len(entries), len(done_ids), len(to_send)
    print(f"  pending: {n_todo}/{n_total}  (done: {n_done}, workers: {args.workers})")
    if args.dry_run:
        for e in to_send[:3]:
            print(f"    DRY-RUN: {e['custom_id']}")
        return
    if n_todo == 0:
        print("  nothing to send")
        return

    file_lock = threading.Lock()
    counter = {"ok": 0, "err": 0, "done": 0}

    def _process(entry: dict):
        result = _send_one(entry, args.base_url, args.api_key)
        with file_lock:
            with open(resp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            counter["done"] += 1
            if result["error"]:
                counter["err"] += 1
                print(f"  [{counter['done']}/{n_todo}] ERROR {entry['custom_id'][:60]}: "
                      f"{str(result['error'])[:120]}")
            else:
                counter["ok"] += 1
                if counter["done"] % 25 == 0 or counter["done"] == n_todo:
                    print(f"  [{counter['done']}/{n_todo}] ok={counter['ok']} err={counter['err']}")

    with ThreadPoolExecutor(max_workers=min(args.workers, n_todo)) as pool:
        list(pool.map(_process, to_send))

    print(f"\nDone. ok={counter['ok']} err={counter['err']}  →  {resp_path}")


if __name__ == "__main__":
    main()
