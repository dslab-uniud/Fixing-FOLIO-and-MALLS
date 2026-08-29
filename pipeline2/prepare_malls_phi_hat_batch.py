"""prepare_malls_phi_hat_batch.py — Prepare phi_hat generation batch requests
for the MALLS dataset using gpt-4o-mini (OpenAI Batch API).

Generates P1 (zero-shot) and P2 (few-shot + constraints) prompts for each of
the 100 MALLS instances.  Each instance has a per-instance ontology loaded
from core/dataset/datasets/malls/signature.jsonl.

Output layout
-------------
    pipeline2/phi_hat_generation/requests/
        malls__gpt-4o-mini__P1.jsonl
        malls__gpt-4o-mini__P1__index.jsonl
        malls__gpt-4o-mini__P2.jsonl
        malls__gpt-4o-mini__P2__index.jsonl

Custom-ID format
----------------
    main|malls_{id}|gpt-4o-mini|{prompt_id}

    e.g.  main|malls_1|gpt-4o-mini|P1
          main|malls_42|gpt-4o-mini|P2
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

from core.dataset.signature_loader import load_malls_signature, format_folio_vocabulary

# ---------------------------------------------------------------------------
# System-prompt templates (mirroring run_qwen_malls_phi_hat.py)
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
                "phi_hat":   {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["phi_hat", "reasoning"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MALLS_RAWPAIRS = (
    _ROOT / "core" / "dataset" / "datasets" / "malls" / "dataset_rawpairs.jsonl"
)
_OUT_DIR = _ROOT / "pipeline2" / "phi_hat_generation" / "requests"

_MODEL_NAME = "gpt-4o-mini"
_MAX_TOKENS = 1024
_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rawpairs(path: Path) -> list[dict]:
    decoder = json.JSONDecoder(strict=False)
    content = path.read_text(encoding="utf-8")
    records: list[dict] = []
    idx = 0
    while idx < len(content):
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


def _build_system_prompt(rec: dict, prompt_id: str) -> str:
    sig = load_malls_signature(int(rec["id"]))
    rels_str, consts_str = format_folio_vocabulary(sig)
    if prompt_id == "P1":
        return _P1_SYSTEM_TMPL.format(relations=rels_str, constants=consts_str)
    else:  # P2
        return _P2_SYSTEM_TMPL.format(
            relations=rels_str,
            constants=consts_str,
            examples=_P2_FEW_SHOT_EXAMPLES,
        )


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
        description="Prepare phi_hat generation batch requests for MALLS (gpt-4o-mini).",
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

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.submit:
        _load_dotenv()
        client = _get_openai_client()
    else:
        client = None

    records = _load_rawpairs(_MALLS_RAWPAIRS)
    print(f"Loaded {len(records)} MALLS instances")

    submitted_batches: list[tuple[str, str]] = []

    for prompt_id in ("P1", "P2"):
        requests:      list[dict] = []
        index_entries: list[dict] = []

        for rec in records:
            malls_id   = rec["id"]
            custom_id  = f"main|malls_{malls_id}|{_MODEL_NAME}|{prompt_id}"
            nl         = rec["NL_sentence"]
            phi_gold   = rec.get("FOL_sentence_new", "").strip()
            ambiguity  = rec.get("ambiguity", False)
            system     = _build_system_prompt(rec, prompt_id)

            requests.append({
                "custom_id": custom_id,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":           _MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": f"Sentence: {nl}"},
                    ],
                    "response_format": _FOL_OUTPUT_SCHEMA,
                    "temperature":     _TEMPERATURE,
                    "max_tokens":      _MAX_TOKENS,
                },
            })
            index_entries.append({
                "custom_id":      custom_id,
                "instance_id":    f"malls_{malls_id}",
                "p":              nl,
                "phi_gold":       phi_gold,
                "source_dataset": "malls",
                "malls_id":       malls_id,
                "ambiguity":      ambiguity,
                "split_name":     "main",
                "model_name":     _MODEL_NAME,
                "prompt_id":      prompt_id,
            })

        req_path = _OUT_DIR / f"malls__{_MODEL_NAME}__{prompt_id}.jsonl"
        idx_path = _OUT_DIR / f"malls__{_MODEL_NAME}__{prompt_id}__index.jsonl"

        with open(req_path, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        with open(idx_path, "w", encoding="utf-8") as f:
            for entry in index_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  {prompt_id}: {len(requests):>4} requests → {req_path.relative_to(_ROOT)}")

        if args.submit:
            print(f"  [{prompt_id}] Submitting …", end=" ", flush=True)
            batch_id = _submit_openai_batch(req_path, client)
            print(f"batch_id: {batch_id}")
            submitted_batches.append((prompt_id, batch_id))
            (req_path.parent / f"malls__{_MODEL_NAME}__{prompt_id}__batch_id.txt").write_text(
                batch_id + "\n", encoding="utf-8"
            )

    print(f"\n{'='*60}")
    print("REQUEST COUNT SUMMARY")
    print(f"{'='*60}")
    print(f"  MALLS instances:          {len(records)}")
    print(f"  Prompt variants:          2 (P1, P2)")
    print(f"  Total requests:           {len(records) * 2}")

    if args.submit:
        print(f"\nSubmitted {len(submitted_batches)} OpenAI batches.")
        for pid, bid in submitted_batches:
            print(f"  [malls/{_MODEL_NAME}/{pid}] {bid}")
    else:
        print(f"\n--submit not set: files written, nothing submitted to OpenAI.")
        print("When ready: python pipeline2/prepare_malls_phi_hat_batch.py --submit")


if __name__ == "__main__":
    main()
