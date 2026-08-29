"""run_gemma_phi_hat.py — Generate phi_hat for all datasets using Gemma 4 31B + P1 prompt.

Sends the P1 autoformalization prompt to the Gemma vLLM server on port 8001.
Per-instance ontology (Relations + Constants) is injected from the dataset signature files.

Datasets covered
----------------
  folio   core/dataset/datasets/folio/dataset_rawpairs.jsonl
  malls   core/dataset/datasets/malls/dataset_rawpairs.jsonl

Output files
------------
  pipeline2/phi_hat_generation/results/
      folio__gemma-4-31B-it__P1.jsonl
      malls__gemma-4-31B-it__P1.jsonl

Prerequisites
-------------
  SSH tunnel open on port 8001:
      ssh -N -L 8001:127.0.0.1:8000 <remote-server> &

Usage
-----
    python pipeline2/run_gemma_phi_hat.py                        # all datasets
    python pipeline2/run_gemma_phi_hat.py --datasets folio       # one dataset
    python pipeline2/run_gemma_phi_hat.py --workers 4
    python pipeline2/run_gemma_phi_hat.py --dry-run

Resume-safe: already-written instance ids are skipped per output file.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import requests as _http

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.dataset.signature_loader import (
    get_story_id_from_instance_id,
    load_folio_signature,
    load_malls_signature,
    format_folio_vocabulary,
)

# ---------------------------------------------------------------------------
# P1 system-prompt template (same as run_qwen_malls_phi_hat.py)
# ---------------------------------------------------------------------------

_P1_SYSTEM_TMPL = """\
You are an expert in formal logic. Your task is to translate a natural language \
sentence into a well-formed formula of first-order logic (FOL).

## Syntax Rules
- Logical operators (Unicode only): ∀ (forall), ∃ (exists), ¬ (negation), ∧ (conjunction), \
∨ (disjunction), → (material implication), ↔ (biconditional), ⊕ (exclusive disjunction)
- Quantifier scope must always be made explicit with parentheses, e.g. ∀x(P(x) → Q(x))
- Variables: lowercase single letters, e.g. x, y, z

## Vocabulary — Use ONLY the symbols listed below, with no additions or substitutions
- Relations (predicates): {relations}
- Constants: {constants}

## Output Format
Return a JSON object with exactly two fields:
- "reasoning": a step-by-step explanation of your formalization decisions
- "phi_hat": the final, closed FOL formula as a string

Do not include any text outside the JSON object."""

_FOL_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "FOLOutput",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "phi_hat":   {"title": "Phi Hat",   "type": "string"},
                "reasoning": {"title": "Reasoning", "type": "string"},
            },
            "required": ["phi_hat", "reasoning"],
            "title": "FOLOutput",
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESULTS_DIR   = _ROOT / "pipeline2" / "phi_hat_generation" / "results"
_FOLIO_DIR     = _ROOT / "core" / "dataset" / "datasets" / "folio"
_MALLS_PATH    = _ROOT / "core" / "dataset" / "datasets" / "malls" / "dataset_rawpairs.jsonl"

_GEMMA_BASE_URL  = "http://localhost:8001"
_GEMMA_API_KEY   = "mm_vllm_student"
_GEMMA_MODEL     = "google/gemma-4-31B-it"
_GEMMA_MODEL_KEY = "gemma-4-31B-it"   # used in output filenames
_MAX_TOKENS      = 14000
_DEFAULT_WORKERS = 4
_RETRY_ATTEMPTS  = 3

_thread_local = threading.local()

_ALL_DATASETS = ["folio", "malls"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_multiline_json(path: Path) -> list[dict]:
    decoder = json.JSONDecoder(strict=False)
    content = path.read_text(encoding="utf-8").strip()
    records: list[dict] = []
    idx = 0
    while idx < len(content):
        while idx < len(content) and (content[idx].isspace() or content[idx] in ",[]"):
            idx += 1
        if idx >= len(content):
            break
        obj, end = decoder.raw_decode(content, idx)
        records.append(obj)
        idx = end
    return records


# ---------------------------------------------------------------------------
# Ontology helpers
# ---------------------------------------------------------------------------

def _vocab_folio(instance_id: str) -> tuple[str, str]:
    story_id = get_story_id_from_instance_id(instance_id)
    sig = load_folio_signature(story_id)
    return format_folio_vocabulary(sig)


def _vocab_malls(instance_id: int) -> tuple[str, str]:
    sig = load_malls_signature(instance_id)
    return format_folio_vocabulary(sig)


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def _compute_label(
    phi_hat: str,
    phi_gold_list: list[str],
    ambiguity_flag: bool,
) -> tuple[str, str, Optional[str], Optional[str]]:
    """Return (label, label_method, label_comment, error_type).

    For ambiguous instances: label='?'.
    For normal instances: label='yes' if phi_hat is Z3-equivalent to ANY formula in
    phi_gold_list; otherwise 'no'.
    """
    if ambiguity_flag:
        return "?", "ambiguity", "instance marked as ambiguous", None

    try:
        from core.formalism.fol.fol import FOL
        from core.formalism.fol.solver_z3 import FOLSolver
        solver = FOLSolver()
        hat_fol = FOL(phi_hat)
        _ = hat_fol.tree  # pre-validate

        for gold_str in phi_gold_list:
            gold_fol = FOL(gold_str)
            _ = gold_fol.tree
            if solver.are_equivalent(hat_fol, gold_fol):
                return "yes", "z3", "semantically equivalent to phi_gold", None
        return "no", "z3", "not semantically equivalent to phi_gold", None
    except Exception as exc:
        err = str(exc)
        if "parse" in err.lower() or "syntax" in err.lower():
            return "no", "z3", "formula could not be parsed", "syntactically_incorrect"
        return "no", "z3", f"solver error: {err[:100]}", "solver_error"


# ---------------------------------------------------------------------------
# Gemma response parsing
# ---------------------------------------------------------------------------

def _extract_json_from_gemma_content(content: str) -> dict:
    """Extract JSON from Gemma output, which may include a thinking preamble."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    last_brace = content.rfind("{")
    if last_brace != -1:
        try:
            return json.loads(content[last_brace:])
        except (json.JSONDecodeError, ValueError):
            pass

    for i in range(len(content) - 1, -1, -1):
        if content[i] == "}":
            for j in range(i, -1, -1):
                if content[j] == "{":
                    try:
                        return json.loads(content[j:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        continue

    raise ValueError("Cannot extract JSON from Gemma content")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_session() -> _http.Session:
    if not hasattr(_thread_local, "session"):
        s = _http.Session()
        s.headers.update({
            "Authorization": f"Bearer {_GEMMA_API_KEY}",
            "Content-Type":  "application/json",
        })
        _thread_local.session = s
    return _thread_local.session


def _send_one(body: dict) -> tuple[Optional[dict], Optional[str]]:
    import time
    session = _get_session()
    url = _GEMMA_BASE_URL.rstrip("/") + "/v1/chat/completions"
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = session.post(url, json=body, timeout=3600)
            if resp.ok:
                return resp.json(), None
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(1.0)
                if hasattr(_thread_local, "session"):
                    del _thread_local.session
            else:
                return None, str(exc)
    return None, "max retries exceeded"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def _append_row(path: Path, row: dict, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-instance processor
# ---------------------------------------------------------------------------

def _process_instance(
    instance_id: str,       # unique id within the dataset (used in result row)
    source_dataset: str,    # e.g. "folio", "malls"
    nl: str,
    phi_gold_list: list[str],  # list of valid gold formulas (usually 1; >1 for ambiguous instances)
    ambiguity_flag: bool,
    rels: str,
    consts: str,
    prompt_id: str,
    out_file: Path,
    lock: threading.Lock,
    counter: dict,
    dry_run: bool,
) -> None:
    composite_id = f"{source_dataset}__{instance_id}__{_GEMMA_MODEL_KEY}__{prompt_id}"
    phi_gold = phi_gold_list[0] if phi_gold_list else ""

    system = _P1_SYSTEM_TMPL.format(relations=rels, constants=consts)

    if dry_run:
        print(f"    DRY-RUN: {composite_id}: {nl[:60]}")
        return

    body = {
        "model":           _GEMMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Sentence: {nl}"},
        ],
        "max_tokens":      _MAX_TOKENS,
        "response_format": _FOL_OUTPUT_SCHEMA,
        "temperature":     0.0,
    }

    resp_body, err = _send_one(body)

    phi_hat = ""
    reasoning = ""
    error_type = None

    if err or resp_body is None:
        error_type = f"http_error: {err}"
        label, label_method, label_comment = "no", "error", error_type
    else:
        try:
            content = resp_body["choices"][0]["message"]["content"]
            parsed = _extract_json_from_gemma_content(content)
            phi_hat = parsed.get("phi_hat", "").strip()
            reasoning = parsed.get("reasoning", "")
        except Exception as exc:
            error_type = f"parse_error: {str(exc)[:100]}"

        if phi_hat:
            label, label_method, label_comment, error_type = _compute_label(
                phi_hat, phi_gold_list, ambiguity_flag
            )
        else:
            label, label_method, label_comment = "no", "error", "empty phi_hat"
            if error_type is None:
                error_type = "empty_response"

    row = {
        "id":                  composite_id,
        "id_source_dataset":   instance_id,
        "source_dataset":      source_dataset,
        "generator_model":     _GEMMA_MODEL_KEY,
        "generator_prompt_id": prompt_id,
        "p":                   nl,
        "phi_gold":            phi_gold,
        "phi_hat":             phi_hat,
        "reasoning":           reasoning,
        "label":               label,
        "label_method":        label_method,
        "label_comment":       label_comment,
        "error_type":          error_type,
        "ambiguity_flag":      ambiguity_flag,
        "comment":             None,
    }

    _append_row(out_file, row, lock)

    with lock:
        counter["done"] += 1
        if err or not phi_hat:
            counter["err"] += 1
        else:
            counter["ok"] += 1
        total = counter["total"]
        done  = counter["done"]
        if done % 10 == 0 or done == total:
            print(f"  [{done}/{total}] ok={counter['ok']} err={counter['err']}")


# ---------------------------------------------------------------------------
# Dataset-specific runners
# ---------------------------------------------------------------------------

def _run_folio(workers: int, dry_run: bool) -> None:
    path      = _FOLIO_DIR / "dataset_rawpairs.jsonl"
    out_file  = _RESULTS_DIR / f"folio__{_GEMMA_MODEL_KEY}__P1.jsonl"
    done_ids  = _load_done_ids(out_file)
    records   = _load_multiline_json(path)

    # Build todo list, check ontology for each
    todo = []
    for r in records:
        cid = f"folio__{r['id']}__{_GEMMA_MODEL_KEY}__P1"
        if cid in done_ids:
            continue
        story_id = get_story_id_from_instance_id(r["id"])
        sig      = load_folio_signature(story_id)
        rels, consts = format_folio_vocabulary(sig)
        if not rels:
            print(f"  WARNING: no relations for {r['id']}, skipping")
            continue
        todo.append({
            "instance_id": r["id"],
            "nl":          r.get("NL_sentence", ""),
            "phi_gold_list": [r.get("FOL_sentence_new", "")],
            "ambiguity":   bool(r.get("ambiguity", False)),
            "rels":        rels,
            "consts":      consts,
        })

    print(f"\n  folio: {len(todo)}/{len(records)} pending → {out_file.name}")
    if not todo:
        return

    lock    = threading.Lock()
    counter = {"done": 0, "ok": 0, "err": 0, "total": len(todo)}

    def _task(item):
        _process_instance(
            instance_id=item["instance_id"], source_dataset="folio",
            nl=item["nl"], phi_gold_list=item["phi_gold_list"],
            ambiguity_flag=item["ambiguity"],
            rels=item["rels"], consts=item["consts"],
            prompt_id="P1", out_file=out_file,
            lock=lock, counter=counter, dry_run=dry_run,
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
        list(pool.map(_task, todo))

    print(f"  folio: finished → ok={counter['ok']} err={counter['err']}")


def _run_malls(workers: int, dry_run: bool) -> None:
    out_file = _RESULTS_DIR / f"malls__{_GEMMA_MODEL_KEY}__P1.jsonl"
    done_ids = _load_done_ids(out_file)
    records  = _load_multiline_json(_MALLS_PATH)

    todo = []
    for r in records:
        cid = f"malls__{r['id']}__{_GEMMA_MODEL_KEY}__P1"
        if cid in done_ids:
            continue
        sig = load_malls_signature(r["id"])
        rels, consts = format_folio_vocabulary(sig)
        if not rels:
            print(f"  WARNING: no relations for MALLS id={r['id']}, skipping")
            continue
        todo.append({
            "instance_id": str(r["id"]),
            "nl":          r.get("NL_sentence", ""),
            "phi_gold_list": [r.get("FOL_sentence_new", "")],
            "ambiguity":   False,
            "rels":        rels,
            "consts":      consts,
        })

    print(f"\n  malls: {len(todo)}/{len(records)} pending → {out_file.name}")
    if not todo:
        return

    lock    = threading.Lock()
    counter = {"done": 0, "ok": 0, "err": 0, "total": len(todo)}

    def _task(item):
        _process_instance(
            instance_id=item["instance_id"], source_dataset="malls",
            nl=item["nl"], phi_gold_list=item["phi_gold_list"],
            ambiguity_flag=item["ambiguity"],
            rels=item["rels"], consts=item["consts"],
            prompt_id="P1", out_file=out_file,
            lock=lock, counter=counter, dry_run=dry_run,
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
        list(pool.map(_task, todo))

    print(f"  malls: finished → ok={counter['ok']} err={counter['err']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate phi_hat for all datasets using Gemma 4 31B + P1 prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=_ALL_DATASETS,
        choices=_ALL_DATASETS,
        help=f"Datasets to run (default: all). Choices: {_ALL_DATASETS}",
    )
    parser.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS,
        help=f"Concurrent HTTP workers per dataset (default: {_DEFAULT_WORKERS}).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Ontology sanity check before sending any requests
    print("Verifying ontology coverage...")
    _verify_ontologies(args.datasets)
    print("  All ontologies OK.\n")

    print("=" * 60)
    print(f"  Gemma phi_hat generation — {_GEMMA_MODEL_KEY}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Prompt   : P1")
    print(f"  Workers  : {args.workers}")
    print(f"  Dry-run  : {args.dry_run}")
    print("=" * 60)

    for ds in args.datasets:
        if ds == "folio":
            _run_folio(args.workers, args.dry_run)
        elif ds == "malls":
            _run_malls(args.workers, args.dry_run)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"Output directory: {_RESULTS_DIR}")


def _verify_ontologies(datasets: list[str]) -> None:
    """Check every instance in the requested datasets has a non-empty ontology."""
    import re

    def _load(path):
        import re as _re
        with open(path) as f:
            content = f.read()
        content = _re.sub(
            r'("|\btrue\b|\bfalse\b|\bnull\b|[0-9])\s*\n(\s*")',
            lambda m: m.group(1) + ",\n" + m.group(2), content
        )
        d = json.JSONDecoder(strict=False); pos, records = 0, []
        while pos < len(content.strip()):
            try:
                obj, end = d.raw_decode(content, pos); records.append(obj); pos = end
                while pos < len(content) and content[pos] in " \t\n\r,": pos += 1
            except: break
        return records

    errors = []
    for ds in datasets:
        if ds == "folio":
            records = _load(_FOLIO_DIR / "dataset_rawpairs.jsonl")
            for r in records:
                sid = get_story_id_from_instance_id(r["id"])
                try:
                    sig = load_folio_signature(sid)
                    rels, _ = format_folio_vocabulary(sig)
                    if not rels:
                        errors.append(f"  {ds}: {r['id']} — empty relations")
                except KeyError:
                    errors.append(f"  {ds}: {r['id']} — signature missing")

        elif ds == "malls":
            records = _load(_MALLS_PATH)
            for r in records:
                try:
                    sig = load_malls_signature(r["id"])
                    rels, _ = format_folio_vocabulary(sig)
                    if not rels:
                        errors.append(f"  malls: id={r['id']} — empty relations")
                except KeyError:
                    errors.append(f"  malls: id={r['id']} — signature missing")


    if errors:
        raise RuntimeError(
            "Ontology verification failed:\n" + "\n".join(errors)
        )


if __name__ == "__main__":
    main()
