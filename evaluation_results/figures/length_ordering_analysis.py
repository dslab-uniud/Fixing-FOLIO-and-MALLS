"""length_ordering_analysis.py — Length-based review-ordering strategies.

For gemma, compares two new "review order" strategies against Pipeline 1 (P1,
B1-B3 x pv1,pv3 band) and Pipeline 2 (P2, best single variant):

  1. "Original dataset, longest-first" — human reviews FOL_sentence_old
     instances in DESCENDING length order, fixing each one reviewed.
  2. "Green baseline, longest-first" — human reviews phi_hat (Stage 1
     direct-translation) instances in DESCENDING length order.

Each of the above is computed for two length metrics:
  (a) FOL formula length  (len(FOL_sentence_old) or len(phi_hat))
  (b) NL sentence length  (len(NL_sentence))

Unlike Black/Green (straight lines, order-independent), these are genuine
empirical step curves: x = k/N instances reviewed (longest-first), y =
cumulative fraction of the dataset with a correct final formula, where the
k reviewed instances are fixed (become correct) and the N-k unreviewed
instances keep their original correctness.

Scoped to FOLIO and MALLS only — GGC/Stanford is a private dataset and is
excluded from this public release.

Writes:
  evaluation_results/figures/length_ordering_results/_records_cache.pkl
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.formalism.fol.fol import FOL
from core.formalism.fol.solver_z3 import FOLSolver
from core.formalism.fol.ast import QuantifierNode, BooleanNode, RelationNode, BoolConstNode

Z3_TIMEOUT = 20_000
solver = FOLSolver()
_cache = {}


import re as _re

# Same spurious-symbol map used elsewhere in this project (∇/∪/∘/∗/∊ are
# non-standard unicode look-alikes gemma sometimes emits instead of
# ¬/∨/∨/∧/∀). Used HERE only as a parsing aid to recover an element count for
# otherwise-unparseable phi_hat strings — it does not change the correctness
# verdict, which is computed separately via are_equiv on the raw text.
_SYMBOL_MAP = {"∇": "¬", "−": "¬", "∪": "∨", "∘": "∨", "∗": "∧", "∊": "∀"}
_NEG_EXISTS_RE = _re.compile(r"∄")


def _apply_symbol_map(formula: str) -> str:
    out = _NEG_EXISTS_RE.sub("¬∃", formula)
    for bad, good in _SYMBOL_MAP.items():
        out = out.replace(bad, good)
    return out


def _count_tree_elements(tree) -> int:
    if isinstance(tree, QuantifierNode):
        return 1 + _count_tree_elements(tree.body)
    if isinstance(tree, BooleanNode):
        return 1 + sum(_count_tree_elements(c) for c in tree.children)
    if isinstance(tree, RelationNode):
        return 1 + len(tree.arguments)
    if isinstance(tree, BoolConstNode):
        return 1
    return 0


def count_fol_elements(formula_str: str) -> int | None:
    """Count FOL 'elements': each quantifier, logical operator, predicate/relation,
    and variable/constant argument counts as one. Returns None if the formula
    cannot be parsed even after mapping gemma's spurious symbols to standard ones."""
    try:
        return _count_tree_elements(FOL(formula_str).tree)
    except Exception:
        pass
    mapped = _apply_symbol_map(formula_str)
    if mapped == formula_str:
        return None
    try:
        return _count_tree_elements(FOL(mapped).tree)
    except Exception:
        return None


def count_nl_words(text: str) -> int:
    """Word count for an NL sentence (whitespace-separated tokens)."""
    return len(text.split())


def are_equiv(a, b, theory=None):
    if isinstance(b, list):
        results = [are_equiv(a, bi, theory) for bi in b if bi]
        if any(r is True for r in results):
            return True
        if all(r is False for r in results):
            return False
        return None
    if not a or not b:
        return None
    if a == b:
        return True
    key = ((a, b) if a < b else (b, a)) + (bool(theory),)
    if key in _cache:
        return _cache[key]
    try:
        r = solver.are_equivalent(FOL(a), FOL(b), theory=theory or None, timeout=Z3_TIMEOUT)
    except Exception:
        r = None
    _cache[key] = r
    return r


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_input_fol(fol_sentence_old: str) -> str:
    lines = [l.strip() for l in fol_sentence_old.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    return " ∧ ".join(f"({l})" for l in lines)


GEMMA_PHI_HAT_FILES = {
    "FOLIO": [ROOT / "pipeline2/phi_hat_generation/results/folio__gemma-4-31B-it__P1.jsonl"],
    "MALLS": [ROOT / "pipeline2/phi_hat_generation/results/malls__gemma-4-31B-it__P1.jsonl"],
}


def load_dataset_instances(dataset: str):
    """Return list of dicts: {id, nl, gold (str or list), input_fol, ambiguous}."""
    out = []
    if dataset == "FOLIO":
        for r in load_jsonl(ROOT / "core/dataset/datasets/folio/dataset_rawpairs.jsonl"):
            out.append({
                "id": str(r["id"]),
                "nl": r.get("NL_sentence", ""),
                "gold": r["FOL_sentence_new"],
                "input_fol": build_input_fol(r.get("FOL_sentence_old", "")),
                "ambiguous": bool(r.get("ambiguity", False)),
                "is_original_correct": r.get("is_original_correct"),
            })
    elif dataset == "MALLS":
        for r in load_jsonl(ROOT / "core/dataset/datasets/malls/dataset_rawpairs.jsonl"):
            out.append({
                "id": str(r["id"]),
                "nl": r.get("NL_sentence", ""),
                "gold": r["FOL_sentence_new"],
                "input_fol": r.get("FOL_sentence_old", ""),
                "ambiguous": bool(r.get("ambiguity", False)),
                "is_original_correct": r.get("is_original_correct"),
            })
    return out


def load_gemma_phi_hat(dataset: str):
    """Return {instance_id: phi_hat_str}."""
    out = {}
    for path in GEMMA_PHI_HAT_FILES[dataset]:
        if not path.exists():
            continue
        for r in load_jsonl(path):
            iid = str(r.get("id_source_dataset", ""))
            phi_hat = r.get("phi_hat", "")
            if isinstance(phi_hat, list):
                phi_hat = phi_hat[0] if phi_hat else ""
            out[iid] = phi_hat
    return out


def compute_correctness(dataset: str):
    """Return list of dicts per instance:
    {id, nl_len, input_len, phi_hat_len, orig_correct (bool|None), green_correct (bool|None)}.
    """
    instances = load_dataset_instances(dataset)
    phi_hat_map = load_gemma_phi_hat(dataset)

    records = []
    for inst in instances:
        iid = inst["id"]
        gold = inst["gold"]
        nl_len = count_nl_words(inst["nl"])

        # original-dataset correctness.
        # Uses the dataset's own is_original_correct label directly (ground truth
        # as curated), NOT a freshly Z3-recomputed equivalence check — these can
        # disagree (e.g. on ambiguous instances), and the curated label is what
        # the Black baseline / dataset-composition numbers are defined against.
        input_fol = inst["input_fol"]
        input_len = count_fol_elements(input_fol) if input_fol else None
        orig_correct = None
        ioc = inst.get("is_original_correct")
        if ioc is not None:
            orig_correct = (ioc == "yes")

        # green-baseline (phi_hat) correctness
        phi_hat = phi_hat_map.get(iid, "")
        phi_hat_len = count_fol_elements(phi_hat) if phi_hat else None
        green_correct = None
        if phi_hat and gold:
            green_correct = are_equiv(phi_hat, gold) is True

        records.append({
            "id": iid,
            "nl_len": nl_len,
            "input_len": input_len,
            "phi_hat_len": phi_hat_len,
            "orig_correct": orig_correct,
            "green_correct": green_correct,
        })
    return records


def build_length_curve(records, length_key, correct_key):
    """Sort by length_key DESC (longest reviewed first). Reviewed instances become
    correct; unreviewed keep their original correctness.
    Returns (xs, ys) arrays of length N+1 (x=0..1)."""
    valid = [r for r in records if r.get(length_key) is not None and r.get(correct_key) is not None]
    valid.sort(key=lambda r: r[length_key], reverse=True)
    n = len(valid)
    if n == 0:
        return None
    correct_flags = [r[correct_key] for r in valid]
    xs = [0.0]
    ys = [sum(correct_flags) / n]
    n_correct_reviewed_so_far = 0
    for k in range(1, n + 1):
        n_correct_reviewed_so_far += 1  # instance k-1 just got fixed
        n_correct_remaining = sum(correct_flags[k:])  # unreviewed, still at original correctness
        y = (n_correct_reviewed_so_far + n_correct_remaining) / n
        xs.append(k / n)
        ys.append(y)
    return np.array(xs), np.array(ys)


def auc_step(xs, ys):
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(ys, xs))


def find_threshold(xs, ys, thresh):
    idx = np.searchsorted(ys, thresh)
    if idx == 0:
        return 0.0
    if idx >= len(ys):
        return 1.0
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    if y1 == y0:
        return float(x0)
    t = (thresh - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


if __name__ == "__main__":
    OUT_DIR = Path(__file__).parent / "length_ordering_results"
    OUT_DIR.mkdir(exist_ok=True)

    all_records = {}
    for ds in ["FOLIO", "MALLS"]:
        print(f"Computing correctness for {ds}...")
        all_records[ds] = compute_correctness(ds)
        print(f"  {len(all_records[ds])} instances")

    import pickle
    with open(OUT_DIR / "_records_cache.pkl", "wb") as f:
        pickle.dump(all_records, f)
    print("Saved records cache.")
