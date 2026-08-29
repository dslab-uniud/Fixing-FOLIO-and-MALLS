"""run_qwen_malls_phi_hat.py — Generate phi_hat for MALLS using Qwen3-30B-A3B.

Sends P1 and P2 generation prompts (one per MALLS instance) to the Qwen vLLM
server on port 8002.  Per-instance ontology is injected into the system prompt
from core/dataset/datasets/malls/signature.jsonl.

For each instance, the script:
  1. Builds the P1 / P2 system prompt with instance-specific Relations/Constants.
  2. Sends to Qwen: "Sentence: {NL_sentence}".
  3. Parses the JSON response → phi_hat + reasoning.
  4. Computes the Z3 label against phi_gold = FOL_sentence_new:
       - ambiguity=True  → label='?'
       - Z3 equivalent   → label='yes'
       - not equivalent  → label='no'
       - parse/solver error → label='no', error_type set
  5. Appends the result row to the output JSONL.

Output files
------------
    pipeline2/phi_hat_generation/results/
        malls__Qwen3-30B-A3B__P1.jsonl
        malls__Qwen3-30B-A3B__P2.jsonl

Result row fields (matching FOLIO/MALLS result format):
    id, id_source_dataset, source_dataset, generator_model, generator_prompt_id,
    p, phi_gold, phi_hat, reasoning, label, label_method, label_comment,
    error_type, ambiguity_flag, comment

Prerequisites
-------------
  SSH tunnel to Qwen vLLM server open on port 8002:
      ssh -N -L 8002:127.0.0.1:4242 <remote-server> &

Usage
-----
    python pipeline2/run_qwen_malls_phi_hat.py               # both P1 and P2
    python pipeline2/run_qwen_malls_phi_hat.py --prompts P1  # P1 only
    python pipeline2/run_qwen_malls_phi_hat.py --workers 8   # concurrency
    python pipeline2/run_qwen_malls_phi_hat.py --dry-run

Resume-safe: already-written instance ids are skipped.
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

from core.dataset.signature_loader import load_malls_signature, format_folio_vocabulary

# ---------------------------------------------------------------------------
# P1 / P2 system-prompt templates
# (mirroring Assess_quality/core/dataset/gen_prompts.py)
# ---------------------------------------------------------------------------

_P2_FEW_SHOT_EXAMPLES = """\
Note: the examples below use illustrative predicates only. \
Your answer MUST use exclusively the vocabulary declared above.

--- Example 1 ---
"Relations (predicates): - Student(x): x is a student\n- GradStudent(x): x is a graduate \
student\n- Professor(x): x is a professor\n- Tenured(x): x has tenure\n- Graduates(x): x \
graduates\n- Passes(x, y): x passes y\n- Teaches(x, y): x teaches y\nConstants: \
- ScienceExam: the science exam \n- neighbors: the group of neighbors \n- Alice: Alice\n\
- Bob: Bob\n Sentence: Every student who passes the science exam will graduate.\n\n\
OUTPUT:\n{"reasoning": "The sentence has the universal structure \\"Every A that satisfies B \
also satisfies C\\", which in FOL corresponds to \\"For every x, if x is A and x satisfies B, \
then x satisfies C\\". Instantiating A = Student, B = Passes(x, ScienceExam), C = Graduates(x) \
yields '∀x (Student(x) ∧ Passes(x, ScienceExam) → Graduates(x))' that use the symbols given \
by the user.", "phi_hat": "∀x (Student(x) ∧ Passes(x, ScienceExam) → Graduates(x))"}

--- Example 2 ---
"Relations (predicates): - Tourist(x): x is a tourist\n- City(x): x is a city\n\
- Animal(x): x is an animal\n- WakesUp(x, y): x wakes up y\n- Visits(x, y): x visits \
location y\nConstants: - John : John\n - James : James\n Sentence: There exists a city that \
every tourist visits. To avoid delay and visit a city, John has to wake James up. If James \
visits a city, then also John visits it. \n\nOUTPUT:\n{"reasoning": "...", \
"phi_hat": "(∃x (City(x) ∧ ∀y (Tourist(y) → Visits(y, x)))) ∧ \
((∃x (City(x) ∧ Visits(John, x))) → WakesUp(John, James)) ∧ \
(∀x (City(x) ∧ Visits(James, x) → Visits(John, x)))"}

--- Example 3 ---
"Relations (predicates):\n- Researcher(x): x is a researcher\n- Paper(x): x is a paper\n\
- Conference(x): x is a conference\n- Prestigious(x): x is prestigious\n\
- SubmitsToYYY(x, y): x submits y to YYY\n- Reviews(x, y): x reviews y\n\
- AcceptedAt(x, y): x is accepted at y\nConstants:\n- Alice: Alice\n- Bob: Bob\n\n\
Sentence: Every researcher submits a paper to the YYY. Alice reviews every paper that is \
submitted to YYY.\n\nOUTPUT:\n{"reasoning": "...", \
"phi_hat": "∀x (Researcher(x) → ∃y (Paper(y) ∧ SubmitsToYYY(x, y))) ∧ \
(∀y ((Paper(y) ∧ ∃z (Researcher(z) ∧ SubmitsToYYY(z, y))) → Reviews(Alice, y)))"}
"""

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

_P2_SYSTEM_TMPL = """\
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


## Constraints
- Every variable must be bound by a quantifier — no free variables in the final formula
- Do not introduce any predicate, constant, or function symbol not listed in the vocabulary above
- Do not add information not present in the original sentence
- If the sentence cannot be faithfully formalized using only the provided vocabulary, \
still produce the closest possible formula and flag the limitation in the reasoning field

## Examples:
{examples}

## Output Format
Return your answer as a JSON object with:
- "reasoning": your step-by-step thinking process
- "phi_hat": the final FOL formula"""

_FOL_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "FOLOutput",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "phi_hat":    {"title": "Phi Hat",    "type": "string"},
                "reasoning":  {"title": "Reasoning",  "type": "string"},
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

_RESULTS_DIR = _ROOT / "pipeline2" / "phi_hat_generation" / "results"
_DATASET     = _ROOT / "core" / "dataset" / "datasets" / "malls" / "dataset_rawpairs.jsonl"

_QWEN_BASE_URL  = "http://localhost:8002"
_QWEN_API_KEY   = "mm_vllm_student"
_QWEN_MODEL     = "Qwen/Qwen3-30B-A3B"
_QWEN_MAX_TOKENS = 2500
_SEED            = 3
_DEFAULT_WORKERS = 16
_ALL_PROMPTS     = ["P1", "P2"]

_thread_local = threading.local()


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
        try:
            obj, end = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error in {path} near pos {idx}: {exc}") from exc
    return records


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def _compute_label(phi_hat: str, phi_gold: str,
                   ambiguity_flag: bool) -> tuple[str, str, Optional[str], Optional[str]]:
    """Return (label, label_method, label_comment, error_type)."""
    if ambiguity_flag:
        return "?", "ambiguity", "instance marked as ambiguous", None

    # Try Z3
    try:
        from core.formalism.fol.fol import FOL
        from core.formalism.fol.solver_z3 import FOLSolver
        equivalent = FOLSolver().are_equivalent(FOL(phi_hat), FOL(phi_gold))
        if equivalent:
            return "yes", "z3", "semantically equivalent to phi_gold", None
        else:
            return "no",  "z3", "not semantically equivalent to phi_gold", None
    except Exception as exc:
        err_str = str(exc)
        if "parse" in err_str.lower() or "syntax" in err_str.lower():
            return "no", "z3", "formula could not be parsed", "syntactically_incorrect"
        return "no", "z3", f"solver error: {err_str[:100]}", "solver_error"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_vocab(instance_id: int) -> tuple[str, str]:
    """Return (relations_str, constants_str) for the given MALLS instance."""
    sig = load_malls_signature(instance_id)
    return format_folio_vocabulary(sig)


def _build_system_prompt(prompt_id: str, relations: str, constants: str) -> str:
    if prompt_id == "P1":
        return _P1_SYSTEM_TMPL.format(relations=relations, constants=constants)
    else:  # P2
        return _P2_SYSTEM_TMPL.format(
            relations=relations,
            constants=constants,
            examples=_P2_FEW_SHOT_EXAMPLES,
        )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_session() -> _http.Session:
    if not hasattr(_thread_local, "session"):
        s = _http.Session()
        s.headers.update({
            "Authorization": f"Bearer {_QWEN_API_KEY}",
            "Content-Type":  "application/json",
        })
        _thread_local.session = s
    return _thread_local.session


def _send_one(body: dict) -> tuple[Optional[dict], Optional[str]]:
    session = _get_session()
    url = _QWEN_BASE_URL.rstrip("/") + "/v1/chat/completions"
    try:
        resp = session.post(url, json=body, timeout=600)
        if resp.ok:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Result file helpers
# ---------------------------------------------------------------------------

def _load_done_ids(result_file: Path) -> set[str]:
    done: set[str] = set()
    if not result_file.exists():
        return done
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def _append_row(result_file: Path, row: dict, lock: threading.Lock) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Runner for one prompt variant
# ---------------------------------------------------------------------------

def run_prompt(
    records:    list[dict],
    prompt_id:  str,
    workers:    int,
    dry_run:    bool,
) -> None:
    out_file  = _RESULTS_DIR / f"malls__Qwen3-30B-A3B__{prompt_id}.jsonl"
    done_ids  = _load_done_ids(out_file)
    lock      = threading.Lock()

    todo = [r for r in records
            if f"malls__{r['id']}__Qwen3-30B-A3B__{prompt_id}" not in done_ids]

    print(f"\n  [{prompt_id}] {len(todo)}/{len(records)} pending")

    if not todo:
        return

    if dry_run:
        for rec in todo[:2]:
            print(f"    DRY-RUN id={rec['id']}: {rec.get('NL_sentence','')[:60]}")
        return

    counter = {"ok": 0, "err": 0, "done": 0}

    def process(rec: dict) -> None:
        instance_id  = rec["id"]
        nl           = rec.get("NL_sentence", "")
        phi_gold     = rec.get("FOL_sentence_new", "").strip()
        ambiguity    = bool(rec.get("ambiguity", False))
        composite_id = f"malls__{instance_id}__Qwen3-30B-A3B__{prompt_id}"

        rels, consts = _build_vocab(instance_id)
        system       = _build_system_prompt(prompt_id, rels, consts)

        body = {
            "model": _QWEN_MODEL,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Sentence: {nl}"},
            ],
            "max_tokens":      _QWEN_MAX_TOKENS,
            "response_format": _FOL_OUTPUT_SCHEMA,
            "seed":            _SEED,
        }

        resp_body, err = _send_one(body)

        phi_hat    = None
        reasoning  = None
        error_type = None

        if err or resp_body is None:
            phi_hat    = ""
            reasoning  = ""
            error_type = f"http_error: {err}"
            label, label_method, label_comment, _ = "no", "error", error_type, None
        else:
            try:
                content = resp_body["choices"][0]["message"]["content"]
                parsed  = json.loads(content)
                phi_hat   = parsed.get("phi_hat", "").strip()
                reasoning = parsed.get("reasoning", "")
            except Exception as exc:
                phi_hat    = ""
                reasoning  = ""
                error_type = f"parse_error: {str(exc)[:100]}"

            if phi_hat:
                label, label_method, label_comment, error_type = _compute_label(
                    phi_hat, phi_gold, ambiguity
                )
            else:
                label, label_method, label_comment = "no", "error", "empty phi_hat"
                if error_type is None:
                    error_type = "empty_response"

        row = {
            "id":                composite_id,
            "id_source_dataset": str(instance_id),
            "source_dataset":    "malls",
            "generator_model":   "Qwen3-30B-A3B",
            "generator_prompt_id": prompt_id,
            "p":                 nl,
            "phi_gold":          phi_gold,
            "phi_hat":           phi_hat,
            "reasoning":         reasoning,
            "label":             label,
            "label_method":      label_method,
            "label_comment":     label_comment,
            "error_type":        error_type,
            "ambiguity_flag":    ambiguity,
            "comment":           None,
        }

        _append_row(out_file, row, lock)

        with lock:
            counter["done"] += 1
            if err or not phi_hat:
                counter["err"] += 1
            else:
                counter["ok"] += 1
            if counter["done"] % 20 == 0 or counter["done"] == len(todo):
                print(f"  [{prompt_id}] [{counter['done']}/{len(todo)}] "
                      f"ok={counter['ok']} err={counter['err']}")

    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
        list(pool.map(process, todo))

    print(f"  [{prompt_id}] finished → {counter['ok']} ok, {counter['err']} errors")
    print(f"  [{prompt_id}] output: {out_file.relative_to(_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MALLS phi_hat using Qwen3-30B-A3B (P1 and P2 prompts).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompts",  nargs="+", default=_ALL_PROMPTS,
                        choices=_ALL_PROMPTS,
                        help="Prompt variants to run (default: P1 P2).")
    parser.add_argument("--workers",  type=int, default=_DEFAULT_WORKERS,
                        help=f"Concurrent HTTP workers (default: {_DEFAULT_WORKERS}).")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MALLS phi_hat generation — Qwen3-30B-A3B")
    print(f"  Prompts : {args.prompts}")
    print(f"  Workers : {args.workers}")
    print(f"  Dry-run : {args.dry_run}")
    print("=" * 60)

    records = _load_multiline_json(_DATASET)
    print(f"\n  Loaded {len(records)} MALLS instances")

    for prompt_id in args.prompts:
        run_prompt(records, prompt_id, args.workers, args.dry_run)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"Output directory: {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
