"""evaluate.py — Evaluate Pipeline 1 curation predictions against ground-truth labels.

Steps:
  A. Compute ground-truth labels (y) from raw dataset files using Z3 equivalence.
  B. Aggregate B6 samples (majority / all_agree / logit_mean).
  C. Build per-baseline metrics table (safety_yes, safety_3, usefulness).
  D. Ensemble analysis (pairs and triples of variants).
  E. Hard-instance analysis for top-10 baselines.

Usage:
    python pipeline1/evaluate.py                     # full FOLIO dataset
    python pipeline1/evaluate.py --recompute-labels  # force Z3 rerun
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RAW_DIR  = _ROOT / "core" / "dataset" / "datasets" / "folio"
_PRED_DIR = _ROOT / "pipeline1" / "assessing_original" / "results" / "predictions"
_GT_DIR   = _ROOT / "pipeline1" / "assessing_original" / "results" / "ground_truth"
_RES_DIR  = _ROOT / "pipeline1" / "assessing_original" / "results"
_TAB_DIR  = _RES_DIR / "tables" / "folio"
_HARD_DIR = _RES_DIR / "hard_instances"

_LLMS     = ("gpt4o_mini", "qwen_local_30b")
_LLM_DIRS = {"gpt4o_mini": "openai", "qwen_local_30b": "qwen"}
_VALID_Q  = ("yes", "no", "?")
_B6_AGG_MODES = ("majority", "all_agree", "logit_mean")

# Z3 timeout per formula (ms) — used for GT label computation
_Z3_TIMEOUT = 30_000
# Z3 timeout for refinement equivalence checks (ms) — can be looser since it's not GT
_Z3_REF_TIMEOUT = 30_000

# ---------------------------------------------------------------------------
# Lazy Z3 equivalence cache for refinement strings.
# Key: canonical (min, max) pair of refinement strings.
# Value: True (equivalent) | False (not equivalent) | None (timeout/error)
# ---------------------------------------------------------------------------
_ref_equiv_cache: Dict[Tuple[str, str], Optional[bool]] = {}
_ref_equiv_stats: Dict[str, int] = {"hits": 0, "z3_ok": 0, "z3_timeout": 0, "z3_error": 0}


def _refs_equivalent(r1: str, r2: str) -> Optional[bool]:
    """Return True/False if r1 ≡ r2 by Z3, None on timeout/error. Results are cached."""
    if r1 == r2:
        return True
    key = (r1, r2) if r1 < r2 else (r2, r1)
    if key in _ref_equiv_cache:
        _ref_equiv_stats["hits"] += 1
        return _ref_equiv_cache[key]

    try:
        from core.formalism.fol.fol import FOL
        from core.formalism.fol.solver_z3 import FOLSolver, SolverTimeoutError
        result = FOLSolver().are_equivalent(FOL(r1), FOL(r2), timeout=_Z3_REF_TIMEOUT)
        _ref_equiv_cache[key] = result
        _ref_equiv_stats["z3_ok"] += 1
        return result
    except Exception as e:
        _ref_equiv_cache[key] = None
        if "timeout" in str(e).lower() or "SolverTimeout" in type(e).__name__:
            _ref_equiv_stats["z3_timeout"] += 1
        else:
            _ref_equiv_stats["z3_error"] += 1
        return None


def _all_refs_equivalent(refs: list[str]) -> Optional[bool]:
    """Check if all non-empty strings in refs are logically equivalent.

    Returns True if all agree, False if any pair provably differs, None if
    any equivalence check timed out (so we cannot confirm agreement).
    """
    non_null = [r for r in refs if r]
    if not non_null:
        return None  # no refinements to compare
    unique = list(dict.fromkeys(non_null))  # deduplicate while preserving order
    if len(unique) == 1:
        return True  # all strings identical
    # Pairwise check (for k=2 or k=3, at most 3 pairs)
    had_timeout = False
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            result = _refs_equivalent(unique[i], unique[j])
            if result is False:
                return False
            if result is None:
                had_timeout = True
    return None if had_timeout else True


# ===========================================================================
# A. Ground-Truth Label Computation
# ===========================================================================

import re as _re

def _load_raw_pairs(path: Path) -> list[dict]:
    """Load multi-line JSON records from a rawpairs file.

    Uses strict=False decoder and auto-repairs a known class of dataset bugs:
    missing commas between adjacent JSON keys inside an object
    (e.g. `"key1": val1\\n  "key2": val2` → `"key1": val1,\\n  "key2": val2`).
    Any unfixable records are skipped with a warning.
    """
    records: list[dict] = []
    decoder = json.JSONDecoder(strict=False)
    content = path.read_text(encoding="utf-8").strip()

    # Pre-repair: insert missing commas between consecutive key-value pairs
    # Pattern: value-end (", true, false, number, or }) followed by whitespace + "
    content = _re.sub(
        r'("|\btrue\b|\bfalse\b|\bnull\b|[0-9])\s*\n(\s*")',
        lambda m: m.group(1) + ",\n" + m.group(2),
        content,
    )

    idx = 0
    n_skipped = 0
    while idx < len(content):
        try:
            obj, end = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end
        except json.JSONDecodeError as e:
            # Skip to the next top-level '{' and try again
            n_skipped += 1
            next_brace = content.find("\n{", idx)
            if next_brace == -1:
                break
            idx = next_brace + 1
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1

    if n_skipped:
        print(f"  WARNING: {n_skipped} malformed record(s) skipped in {path.name} "
              f"(dataset file has JSON syntax errors).")
    return records


def _build_phi_hat_conj(fol_sentence_old: str) -> str:
    """Join multi-line FOL_sentence_old into a single conjunction.

    Each non-empty line is wrapped in parentheses and joined with ∧.
    """
    lines = [ln.strip() for ln in fol_sentence_old.strip().splitlines() if ln.strip()]
    if len(lines) == 1:
        return lines[0]
    return " ∧ ".join(f"({ln})" for ln in lines)


def _compute_y(record: dict) -> str:
    """Compute ground-truth label for one dataset record.

    Returns: 'yes' | 'no' | '?' | 'error'
    """
    if record.get("ambiguity"):
        return "?"

    phi_gold = record.get("FOL_sentence_new", "").strip()
    phi_old  = record.get("FOL_sentence_old", "").strip()

    if not phi_gold or not phi_old:
        return "error"

    phi_hat_conj = _build_phi_hat_conj(phi_old)

    try:
        from core.formalism.fol.fol import FOL
        from core.formalism.fol.solver_z3 import FOLSolver, SolverTimeoutError

        solver = FOLSolver()
        equiv  = solver.are_equivalent(
            FOL(phi_gold), FOL(phi_hat_conj), timeout=_Z3_TIMEOUT
        )
        return "yes" if equiv else "no"
    except Exception as e:
        return "error"


def compute_ground_truth(
    recompute: bool = False
) -> tuple[dict[str, str], dict[str, str]]:
    """Load or compute GT labels for the full FOLIO dataset.

    Returns:
        labels:    {instance_id: y}  — 'yes' / 'no' / '?' / 'error'
        gold_fols: {instance_id: gold_fol_string}
    """
    cache_path = _GT_DIR / "folio.jsonl"

    if cache_path.exists() and not recompute:
        labels, gold_fols = {}, {}
        for line in open(cache_path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            labels[r["id"]]    = r["y"]
            gold_fols[r["id"]] = r.get("FOL_sentence", "")
        return labels, gold_fols

    # Compute from scratch
    raw_path = _RAW_DIR / "dataset_rawpairs.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    records = _load_raw_pairs(raw_path)
    labels, gold_fols = {}, {}
    _GT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Computing Z3 labels for {len(records)} instances ...")
    with open(cache_path, "w") as fout:
        for i, rec in enumerate(records):
            y = _compute_y(rec)
            labels[rec["id"]]    = y
            gold_fols[rec["id"]] = rec.get("FOL_sentence_new", "")
            row = {
                "id":               rec["id"],
                "y":                y,
                "ambiguity":        rec.get("ambiguity", False),
                "corrected":        rec.get("corrected"),
                "FOL_sentence":     rec.get("FOL_sentence_new", ""),
                "FOL_sentence_old": rec.get("FOL_sentence_old", ""),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0 or (i + 1) == len(records):
                print(f"    {i+1}/{len(records)} done", flush=True)

    dist = Counter(labels.values())
    print(f"  Labels: yes={dist['yes']} no={dist['no']} ?={dist['?']} error={dist.get('error', 0)}")
    return labels, gold_fols


# ===========================================================================
# B. Load Extracted Answers
# ===========================================================================

def load_answers(llm_dir: str) -> list[dict]:
    """Load all extracted answer records for an llm (full FOLIO dataset)."""
    records = []
    for method_dir in sorted(_PRED_DIR.iterdir()):
        path = method_dir / "folio" / llm_dir / "results.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ===========================================================================
# B2. B6 Aggregation
# ===========================================================================

def _softmax_logits(logits: dict) -> dict:
    """Convert logprob dict {'yes':lp, 'no':lp, '?':lp} → probabilities."""
    vals = {k: v for k, v in logits.items() if v is not None and k in _VALID_Q}
    if not vals:
        return {}
    max_v = max(vals.values())
    exps = {k: math.exp(v - max_v) for k, v in vals.items()}
    total = sum(exps.values())
    return {k: e / total for k, e in exps.items()}


def aggregate_b6(samples: list[dict], mode: str) -> dict:
    """Aggregate B6 samples for one (instance, pv, task) group.

    Returns a prediction record with quality_score and logits.
    """
    scores = [s["quality_score"] for s in samples if s.get("quality_score") in _VALID_Q]

    # Collect non-null refinements
    refinements = [s.get("refinement") for s in samples if s.get("refinement")]

    if mode == "majority":
        if not scores:
            return {"quality_score": None, "logits": {"yes": None, "no": None, "?": None}, "refinement": None}
        counts = Counter(scores)
        best_count = max(counts.values())
        winners = [k for k, v in counts.items() if v == best_count]
        q = winners[0] if len(winners) == 1 else "?"
        ref = Counter(refinements).most_common(1)[0][0] if refinements and q == "no" else None
        return {"quality_score": q, "logits": {"yes": None, "no": None, "?": None}, "refinement": ref}

    elif mode == "all_agree":
        if not scores:
            return {"quality_score": None, "logits": {"yes": None, "no": None, "?": None}, "refinement": None}
        unique = set(scores)
        q = unique.pop() if len(unique) == 1 else "?"
        ref = refinements[0] if refinements and len(set(refinements)) == 1 and q == "no" else None
        return {"quality_score": q, "logits": {"yes": None, "no": None, "?": None}, "refinement": ref}

    elif mode == "logit_mean":
        # Average probabilities across samples (in probability space after softmax)
        sum_probs: dict[str, float] = {"yes": 0.0, "no": 0.0, "?": 0.0}
        n_valid = 0
        for s in samples:
            logits = s.get("logits") or {}
            probs = _softmax_logits(logits)
            if not probs:
                continue
            for k in _VALID_Q:
                sum_probs[k] += probs.get(k, 0.0)
            n_valid += 1

        if n_valid == 0:
            # Fall back to majority
            return aggregate_b6(samples, "majority")

        avg_probs = {k: v / n_valid for k, v in sum_probs.items()}
        q = max(avg_probs, key=avg_probs.get)
        avg_logits = {k: math.log(max(v, 1e-12)) for k, v in avg_probs.items()}
        ref = Counter(refinements).most_common(1)[0][0] if refinements and q == "no" else None
        return {"quality_score": q, "logits": avg_logits, "refinement": ref}

    raise ValueError(f"Unknown B6 agg mode: {mode}")


def build_variant_predictions(
    answers: list[dict],
    llm_name: str,
) -> Dict[str, Dict[str, dict]]:
    """Build a dict: variant_key → {instance_id: prediction}.

    variant_key format:
        B1-B5: "{method}|{task}|pv{pv}"
        B6:    "B6_{agg_mode}|{task}|pv{pv}"

    Prediction: {"quality_score": str, "logits": dict}
    """
    # Separate B6 samples from single-call methods
    b6_samples: Dict[Tuple, list] = defaultdict(list)  # (task, pv, instance_id) → [samples]
    single: Dict[str, Dict[str, dict]] = defaultdict(dict)  # variant_key → {iid: pred}

    for r in answers:
        method     = r.get("method", "")
        task       = r.get("task", "")
        pv         = r.get("pv", 1)
        iid        = r.get("instance_id", "")
        qs         = r.get("quality_score")
        logits     = r.get("logits") or {"yes": None, "no": None, "?": None}
        refinement = r.get("refinement") or None

        if method == "B6":
            b6_samples[(task, pv, iid)].append({
                "quality_score": qs,
                "logits":        logits,
                "refinement":    refinement,
            })
        else:
            vkey = f"{method}|{task}|pv{pv}"
            single[vkey][iid] = {"quality_score": qs, "logits": logits,
                                  "refinement": refinement}

    result = dict(single)

    # Aggregate B6 with each mode
    for agg_mode in _B6_AGG_MODES:
        variant_preds: Dict[str, Dict[str, dict]] = defaultdict(dict)
        for (task, pv, iid), samples in b6_samples.items():
            vkey = f"B6_{agg_mode}|{task}|pv{pv}"
            variant_preds[vkey][iid] = aggregate_b6(samples, agg_mode)
        for vkey, preds in variant_preds.items():
            result[vkey] = preds

    return result


# ===========================================================================
# C. Metrics
# ===========================================================================

def compute_metrics(
    preds:     Dict[str, dict],          # instance_id → {"quality_score": ..., "refinement": ..., ...}
    gt:        Dict[str, str],           # instance_id → y
    gold_fols: Optional[Dict[str, str]] = None,  # instance_id → gold FOL string
) -> dict:
    """Compute per-variant metrics over the full dataset.

    Prediction counts (all_yes / all_no / all_qm) cover ALL instances (GT valid),
    so their sum equals N_total (60 or 215).

    n_committed = all_yes + all_no  (model committed, regardless of GT label).
    n_correct_yes = model said 'yes' AND GT='yes'.
    n_correct_no  = model said 'no'  AND GT='no'.
    n_correct_refinement = model said 'no' AND GT='no' AND refinement ψ ≡ gold
                           (requires refinement data; set to None when unavailable).
    safety_yes  = n_correct_yes / all_yes
    safety_no   = n_correct_no  / all_no
    safety_3    = (n_correct_yes + n_correct_no) / n_committed
    usefulness1 = n_committed / N_total
    """
    # Collect instances present in both (exclude GT='error')
    common_ids = [iid for iid in preds if iid in gt]
    all_recs   = [(iid, preds[iid].get("quality_score"), gt[iid]) for iid in common_ids
                  if gt[iid] in _VALID_Q]

    n_total = len(all_recs)

    # ---- Ground-truth label distribution ----
    gt_yes = sum(1 for _, _, y in all_recs if y == "yes")
    gt_no  = sum(1 for _, _, y in all_recs if y == "no")
    gt_qm  = sum(1 for _, _, y in all_recs if y == "?")

    # ---- Full-dataset prediction counts ----
    all_yes = sum(1 for _, q, _ in all_recs if q == "yes")
    all_no  = sum(1 for _, q, _ in all_recs if q == "no")
    all_qm  = sum(1 for _, q, _ in all_recs if q == "?")

    # ---- n_committed = pred_yes + pred_no (all instances, including ambiguous) ----
    n_committed = all_yes + all_no

    if n_committed == 0 and n_total == 0:
        return _empty_metrics(n_total=0, gt_yes=0, gt_no=0, gt_qm=0,
                              all_yes=0, all_no=0, all_qm=0)

    # ---- Correct predictions ----
    n_correct_yes = sum(1 for _, q, y in all_recs if q == "yes" and y == "yes")
    n_correct_no  = sum(1 for _, q, y in all_recs if q == "no"  and y == "no")
    n_correct_qm  = sum(1 for _, q, y in all_recs if q == "?"   and y == "?")

    # ---- Correct refinements: pred='no', refinement ψ ≡ gold FOL ----
    # n_correct_refinement: pred='no' AND eq_i=True (any gt)
    # n_correct_ref_no:     pred='no' AND gt='no' AND eq_i=True (correctly identified negative + correct fix)
    if gold_fols is not None:
        n_correct_refinement = 0
        n_correct_ref_no = 0
        for iid in common_ids:
            if gt.get(iid) not in _VALID_Q:
                continue
            pred = preds[iid]
            if pred.get("quality_score") != "no":
                continue
            refinement = pred.get("refinement")
            if not refinement:
                continue
            gold = gold_fols.get(iid, "")
            if not gold:
                continue
            equiv = _refs_equivalent(refinement, gold)
            if equiv is True:
                n_correct_refinement += 1
                if gt.get(iid) == "no":
                    n_correct_ref_no += 1
    else:
        n_correct_refinement = None
        n_correct_ref_no = None

    # ---- Refinement count (instances where a refinement was proposed) ----
    n_refinement = sum(
        1 for iid in common_ids
        if gt.get(iid) in _VALID_Q and preds[iid].get("refinement")
    )

    # ---- Safety metrics ----
    safety_yes = n_correct_yes / all_yes if all_yes > 0 else None
    safety_no  = n_correct_no  / all_no  if all_no  > 0 else None
    safety_3   = (n_correct_yes + n_correct_no) / n_committed if n_committed > 0 else None

    # ---- Usefulness: fraction of all instances where model committed ----
    usefulness1 = n_committed / n_total if n_total > 0 else None

    # ---- Ambiguous-instance abstain rate (diagnostic) ----
    amb = [(iid, q, y) for iid, q, y in all_recs if y == "?"]
    n_amb = len(amb)
    abstain_rate = (
        sum(1 for _, q, _ in amb if q == "?") / n_amb if n_amb > 0 else None
    )

    return {
        # Ground-truth distribution (fixed for the full dataset)
        "gt_yes":                gt_yes,
        "gt_no":                 gt_no,
        "gt_qm":                 gt_qm,
        # Full-dataset prediction counts (sum = n_total)
        "n_total":               n_total,
        "all_yes":               all_yes,
        "all_no":                all_no,
        "all_qm":                all_qm,
        # Committed predictions and breakdown of correct ones
        "n_committed":           n_committed,
        "n_refinement":          n_refinement,
        "n_correct_yes":         n_correct_yes,
        "n_correct_no":          n_correct_no,
        "n_correct_qm":          n_correct_qm,
        "n_correct_refinement":  n_correct_refinement,
        "n_correct_ref_no":      n_correct_ref_no,
        # Safety/usefulness
        "usefulness1":           usefulness1,
        "safety_yes":            safety_yes,
        "safety_no":             safety_no,
        "safety_3":              safety_3,
        # Ambiguous diagnostic
        "n_amb":                 n_amb,
        "abstain_rate":          abstain_rate,
    }


def _empty_metrics(n_total=0, gt_yes=0, gt_no=0, gt_qm=0,
                   all_yes=0, all_no=0, all_qm=0) -> dict:
    return {
        "gt_yes": gt_yes, "gt_no": gt_no, "gt_qm": gt_qm,
        "n_total": n_total, "all_yes": all_yes, "all_no": all_no, "all_qm": all_qm,
        "n_committed": 0, "n_refinement": 0,
        "n_correct_yes": 0, "n_correct_no": 0, "n_correct_qm": 0,
        "n_correct_refinement": None, "n_correct_ref_no": None,
        "usefulness1": None, "safety_yes": None, "safety_no": None,
        "safety_3": None, "n_amb": 0, "abstain_rate": None,
    }


def _pct(v: Optional[float]) -> str:
    return f"{100*v:.1f}%" if v is not None else "  N/A"


# ===========================================================================
# D. Ensemble Analysis
# ===========================================================================

def _agg_verdict_safe(verdicts: list[str]) -> str:
    """All must agree on a non-'?' verdict; otherwise '?'."""
    unique = set(verdicts)
    if len(unique) == 1:
        return unique.pop()
    # If all agree it's '?' that's also fine; but if there's any disagreement → '?'
    return "?"


def _agg_verdict_useful_pair(v1: str, v2: str) -> str:
    """Useful pair rule for 2 variants.

    - Both same → that verdict.
    - One is 'yes'/'no', other is '?' → take the committed verdict.
    - One 'yes' + one 'no', or both '?' → '?'.
    """
    if v1 == v2:
        return v1
    committed = [v for v in (v1, v2) if v != "?"]
    abstained = [v for v in (v1, v2) if v == "?"]
    if len(committed) == 1:
        # One committed, one abstained → take the committed verdict
        return committed[0]
    # Both committed with different verdicts → '?'
    return "?"


def _agg_verdict_majority(verdicts: list[str]) -> str:
    """Strict majority ignoring abstentions; tie → '?'."""
    committed = [v for v in verdicts if v != "?"]
    if not committed:
        return "?"
    counts = Counter(committed)
    best = counts.most_common(1)[0][1]
    winners = [k for k, v in counts.items() if v == best]
    return winners[0] if len(winners) == 1 else "?"


def _apply_agg_verdict(verdicts: list[str], rule: str) -> str:
    if rule in ("safe", "pair_safe"):
        return _agg_verdict_safe(verdicts)
    elif rule in ("useful", "pair_useful"):
        assert len(verdicts) == 2, "useful rule is for pairs only"
        return _agg_verdict_useful_pair(verdicts[0], verdicts[1])
    elif rule == "majority":
        return _agg_verdict_majority(verdicts)
    raise ValueError(f"Unknown rule: {rule}")


def _agg_refinement_safe(refinements: list[Optional[str]]) -> Optional[str]:
    """SAFE: propose refinement only if all non-null refinements are logically equivalent.

    Uses cached Z3 checks.  Returns the first non-null refinement if all agree,
    None if they disagree, or None if any equivalence check timed out/errored
    (cannot confirm agreement → conservative fallback).
    """
    non_null = [r for r in refinements if r]
    if not non_null:
        return None
    result = _all_refs_equivalent(non_null)
    return non_null[0] if result is True else None


def _agg_refinement_useful(refinements: list[Optional[str]]) -> Optional[str]:
    """USEFUL: propose the most commonly proposed refinement (by string equality)."""
    non_null = [r for r in refinements if r]
    if not non_null:
        return None
    return Counter(non_null).most_common(1)[0][0]


def compute_ensemble(
    variant_preds: Dict[str, Dict[str, dict]],
    gt: Dict[str, str],
    rule: str,
    n_components: int,
    top_k: int = 20,
    gold_fols: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Compute metrics for all combinations of n_components variants.

    rule values: 'pair_safe', 'pair_useful' (n=2); 'safe', 'majority' (n=3).
    Returns up to top_k rows; caller may re-sort as needed.
    """
    is_safe_rule    = rule in ("pair_safe", "safe")
    agg_refinement  = _agg_refinement_safe if is_safe_rule else _agg_refinement_useful

    keys = sorted(variant_preds.keys())
    all_rows = []

    for combo in itertools.combinations(keys, n_components):
        all_iids: set = set()
        for k in combo:
            all_iids.update(variant_preds[k].keys())

        ensemble_preds: Dict[str, dict] = {}
        for iid in all_iids:
            verdicts    = []
            refinements = []
            for k in combo:
                p = variant_preds[k].get(iid)
                q = p["quality_score"] if p and p.get("quality_score") in _VALID_Q else "?"
                verdicts.append(q)
                refinements.append(p.get("refinement") if p else None)
            agg_q   = _apply_agg_verdict(verdicts, rule)
            agg_ref = agg_refinement(refinements) if agg_q == "no" else None
            ensemble_preds[iid] = {"quality_score": agg_q, "logits": {}, "refinement": agg_ref}

        m = compute_metrics(ensemble_preds, gt, gold_fols=gold_fols)
        n_correct_ref = m["n_correct_refinement"]
        all_rows.append({
            "variants":              " + ".join(combo),
            "rule":                  rule,
            "gt_yes":                m["gt_yes"],
            "gt_no":                 m["gt_no"],
            "gt_?":                  m["gt_qm"],
            "pred_yes":              m["all_yes"],
            "pred_no":               m["all_no"],
            "pred_?":                m["all_qm"],
            "n_correct_yes":         m["n_correct_yes"],
            "n_correct_no":          m["n_correct_no"],
            "n_correct_?":           m["n_correct_qm"],
            "n_correct_refinement":  n_correct_ref,
            "n_correct_ref_no":      m["n_correct_ref_no"],
            "safety_yes":            m["safety_yes"],
            "n_fully_correct":       m["n_correct_yes"] + (n_correct_ref or 0),
        })

    all_rows.sort(
        key=lambda r: (
            r["safety_yes"] if r["safety_yes"] is not None else -1.0,
            r["n_fully_correct"],
        ),
        reverse=True,
    )
    return all_rows[:top_k]


# ===========================================================================
# E. Hard Instance Analysis
# ===========================================================================

def hard_instance_analysis(
    variant_preds: Dict[str, Dict[str, dict]],
    gt: Dict[str, str],
    raw_records: list[dict],
    gold_fols: Optional[Dict[str, str]] = None,
    top_baselines_k: int = 10,
    top_instances_k: int = 10,
) -> List[dict]:
    """Find the hardest instances across the top-k baselines by safety_3."""
    # Rank variants by safety_3
    ranked = []
    for vkey, preds in variant_preds.items():
        m = compute_metrics(preds, gt, gold_fols=gold_fols)
        ranked.append((vkey, m))
    ranked.sort(
        key=lambda x: (
            x[1]["safety_3"] if x[1]["safety_3"] is not None else -1.0,
        ),
        reverse=True,
    )
    top_keys = [vkey for vkey, _ in ranked[:top_baselines_k]]

    # For each instance, count how many top baselines committed and were wrong
    fail_counts: Counter = Counter()
    fail_details: Dict[str, dict] = {}

    for vkey in top_keys:
        preds = variant_preds[vkey]
        for iid, p in preds.items():
            y = gt.get(iid)
            if y not in ("yes", "no"):
                continue
            q = p.get("quality_score")
            if q in ("yes", "no") and q != y:
                fail_counts[iid] += 1
                if iid not in fail_details:
                    fail_details[iid] = {}
                fail_details[iid][vkey] = q

    # Build raw record lookup
    raw_by_id = {r["id"]: r for r in raw_records}

    # Top instances by fail count
    result = []
    for iid, count in fail_counts.most_common(top_instances_k):
        raw = raw_by_id.get(iid, {})
        row = {
            "instance_id":     iid,
            "fail_count":      count,
            "y":               gt.get(iid, "?"),
            "NL_sentence":     (raw.get("NL_sentence") or "")[:250],
            "FOL_sentence_old": (raw.get("FOL_sentence_old") or "")[:300],
            "FOL_sentence":    (raw.get("FOL_sentence_new") or "")[:300],
        }
        for vkey in top_keys:
            q = fail_details.get(iid, {}).get(vkey, "—")
            row[f"q_{vkey}"] = q
        result.append(row)

    return result


# ===========================================================================
# Main
# ===========================================================================

def run_evaluation(
    gt: Dict[str, str],
    raw_records: list[dict],
    gold_fols: Optional[Dict[str, str]] = None,
) -> None:
    """Full evaluation pipeline for the FOLIO dataset."""
    print(f"\n{'='*60}")
    print(f"FOLIO  (N={len(gt)} instances)")
    print(f"{'='*60}")

    # Label distribution
    dist = Counter(gt.values())
    print(f"  GT labels: yes={dist['yes']} no={dist['no']} ?={dist.get('?', 0)} error={dist.get('error', 0)}")

    _TAB_DIR.mkdir(parents=True, exist_ok=True)
    _HARD_DIR.mkdir(parents=True, exist_ok=True)

    all_variant_preds: Dict[str, Dict[str, dict]] = {}  # for cross-llm if needed

    for llm_name in _LLMS:
        llm_dir = _LLM_DIRS[llm_name]
        ans_files = [
            method_dir / "folio" / llm_dir / "results.jsonl"
            for method_dir in sorted(_PRED_DIR.iterdir())
            if (method_dir / "folio" / llm_dir / "results.jsonl").exists()
        ]
        if not ans_files:
            print(f"  [{llm_name}] No answer files found, skipping.")
            continue

        print(f"\n  [{llm_name}] Loading answers ...")
        answers = load_answers(llm_dir)
        print(f"  [{llm_name}] {len(answers)} answer records loaded.")

        variant_preds = build_variant_predictions(answers, llm_name)
        print(f"  [{llm_name}] {len(variant_preds)} variants found.")

        # Store with llm prefix for ensemble
        for vkey, preds in variant_preds.items():
            all_variant_preds[f"{llm_name}/{vkey}"] = preds

        # ---- C. Per-baseline table ----
        rows = []
        for vkey, preds in sorted(variant_preds.items()):
            m = compute_metrics(preds, gt, gold_fols=gold_fols)
            rows.append({
                "variant":               vkey,
                "pred_yes":              m["all_yes"],
                "pred_no":               m["all_no"],
                "pred_?":                m["all_qm"],
                "gt_yes":                m["gt_yes"],
                "gt_no":                 m["gt_no"],
                "gt_?":                  m["gt_qm"],
                "safety_yes":            m["safety_yes"],
                "usefulness1":           m["usefulness1"],
                "safety_3":              m["safety_3"],
                "n_committed":           m["n_committed"],
                "n_refinement":          m["n_refinement"],
                "n_correct_yes":         m["n_correct_yes"],
                "n_correct_no":          m["n_correct_no"],
                "n_correct_?":           m["n_correct_qm"],
                "n_correct_refinement":  m["n_correct_refinement"],
                "n_correct_ref_no":      m["n_correct_ref_no"],
                "n_fully_correct":       m["n_correct_yes"] + (m["n_correct_refinement"] or 0),
            })

        df = pd.DataFrame(rows)
        out_txt = _TAB_DIR / f"{llm_name}_baselines.txt"
        _write_txt_dual(df, out_txt, title=f"Baselines — llm={llm_name}")
        print(f"\n  [{llm_name}] Per-baseline table ({len(df)} rows) → {out_txt.relative_to(_ROOT)}")
        _print_table(df.sort_values(["safety_yes", "n_fully_correct"], ascending=False), top=15)

        # ---- E. Hard instances ----
        hard = hard_instance_analysis(variant_preds, gt, raw_records, gold_fols=gold_fols)
        if hard:
            hard_df = pd.DataFrame(hard)
            out_hard = _HARD_DIR / f"{llm_name}_hard_instances.txt"
            _write_txt(hard_df, out_hard, title=f"Hard Instances — llm={llm_name}")
            print(f"\n  [{llm_name}] Hard instances → {out_hard.relative_to(_ROOT)}")
            _print_hard_table(hard_df)

    # ---- D. Ensemble Analysis (all LLMs combined) ----
    if len(all_variant_preds) < 2:
        print("\n  Skipping ensemble: fewer than 2 variants available.")
        return

    print(f"\n  Ensemble analysis ({len(all_variant_preds)} total variants) ...")

    ens_specs = [
        ("pair_safe",    2, "pairs_safe"),
        ("pair_useful",  2, "pairs_useful"),
        ("safe",         3, "triples_safe"),
        ("majority",     3, "triples_majority"),
    ]
    for rule, n_comp, label in ens_specs:
        print(f"  Computing {label} ensembles ...")
        rows = compute_ensemble(all_variant_preds, gt, rule, n_comp, top_k=20, gold_fols=gold_fols)
        df = pd.DataFrame(rows)
        out = _TAB_DIR / f"ensemble_{label}_top.txt"
        _write_txt_dual(df, out, title=f"Ensemble ({label})")
        print(f"  → {out.relative_to(_ROOT)}")
        _print_table(df.sort_values(["safety_yes", "n_fully_correct"], ascending=False), top=10)

    # Report refinement Z3 cache stats
    s = _ref_equiv_stats
    print(
        f"\n  Refinement Z3 cache: {s['z3_ok']} checks, {s['hits']} cache hits, "
        f"{s['z3_timeout']} timeouts, {s['z3_error']} errors  "
        f"(total unique pairs cached: {len(_ref_equiv_cache)})"
    )
    if s["z3_timeout"] > 0 or s["z3_error"] > 0:
        print(
            f"  WARNING: {s['z3_timeout']} refinement equivalence timeouts and "
            f"{s['z3_error']} errors — safe refinements conservatively set to null "
            "for those cases."
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "  N/A"
    return f"{100*v:5.1f}%" if pct else f"{v:.4f}"


_GT_DISPLAY_COLS = {"gt_yes", "gt_no", "gt_?", "gt_qm"}   # excluded from table rows


def _gt_header_from_df(df: pd.DataFrame) -> str:
    """Extract GT distribution from first row (constant across the dataset) and format as header line."""
    if df.empty:
        return "GT distribution: (no data)"
    row = df.iloc[0]
    gy = int(row.get("gt_yes", row.get("gt_yes", 0)) or 0)
    gn = int(row.get("gt_no",  row.get("gt_no",  0)) or 0)
    gq = int(row.get("gt_?",   row.get("gt_qm",  0)) or 0)
    return f"GT distribution (constant):  yes={gy}   no={gn}   ?={gq}   (N={gy+gn+gq})"


def _fmt_df_for_txt(df: pd.DataFrame) -> str:
    """Format a DataFrame as a fixed-width string.

    Percentage columns are formatted; gt_* columns are dropped (shown in header).
    """
    pct_cols = {"safety_yes", "usefulness1", "safety_3"}
    drop_cols = [c for c in df.columns if c in _GT_DISPLAY_COLS]
    display = df.drop(columns=drop_cols, errors="ignore").copy()
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda v: f"{100*v:.1f}%" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"
            )
    return display.to_string(index=False)


def _write_txt(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Write a DataFrame to a human-readable fixed-width .txt file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, "=" * len(title), "", _gt_header_from_df(df), "", _fmt_df_for_txt(df), ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_txt_dual(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Write a DataFrame twice — sorted by safety_yes, then by n_fully_correct.

    GT distribution is shown once as a header; gt_* columns are excluded from table rows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _sort(df_: pd.DataFrame, primary: str, secondary: str) -> pd.DataFrame:
        return df_.sort_values(
            [primary, secondary],
            ascending=False,
            na_position="last",
        )

    df_sy  = _sort(df, "safety_yes",      "n_fully_correct")
    df_nfc = _sort(df, "n_fully_correct",  "safety_yes")

    sep = "-" * 80
    lines = [
        title,
        "=" * len(title),
        "",
        _gt_header_from_df(df),          # GT counts shown once here
        "",
        "[Sorted by safety_yes  ↓  (then n_fully_correct  ↓)]",
        sep,
        _fmt_df_for_txt(df_sy),
        "",
        "[Sorted by n_fully_correct  ↓  (then safety_yes  ↓)]",
        sep,
        _fmt_df_for_txt(df_nfc),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _print_table(df: pd.DataFrame, top: int = 15) -> None:
    """Print a compact table to stdout."""
    sub = df.head(top)
    first_col = "variants" if "variants" in sub.columns else "variant"
    num_cols = [
        "gt_yes", "gt_no", "gt_?",
        "pred_yes", "pred_no", "pred_?",
        "n_correct_yes", "n_correct_no", "n_correct_?", "n_correct_refinement",
        "n_fully_correct",
    ]
    pct_cols = ["safety_yes"]

    avail_num = [c for c in num_cols if c in sub.columns]
    avail_pct = [c for c in pct_cols if c in sub.columns]

    short = {
        "gt_yes": "gY", "gt_no": "gN", "gt_?": "g?",
        "pred_yes": "pY", "pred_no": "pN", "pred_?": "p?",
        "n_correct_yes": "cY", "n_correct_no": "cN",
        "n_correct_?": "c?", "n_correct_refinement": "cRef",
        "n_fully_correct": "nFC",
        "safety_yes": "safe_Y",
    }

    name_w = 55 if first_col == "variants" else 35
    hdr = f"{first_col.upper():<{name_w}}"
    for c in avail_num:
        hdr += f" {short.get(c, c[:5]):>5}"
    for c in avail_pct:
        hdr += f" {short.get(c, c[:7]):>7}"
    print(hdr)
    print("-" * len(hdr))

    for _, r in sub.iterrows():
        name = str(r.get(first_col, ""))
        name = name.replace("gpt4o_mini/", "G/").replace("qwen_local_30b/", "Q/")
        line = f"{name[:name_w]:<{name_w}}"
        for c in avail_num:
            val = r.get(c)
            line += f" {str(val) if val is not None else 'N/A':>5}"
        for c in avail_pct:
            v = r.get(c)
            s = f"{100*v:.1f}%" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"
            line += f" {s:>7}"
        print(line)
    print()


def _print_baseline_table(df: pd.DataFrame, top: int = 15) -> None:
    """Print a readable per-baseline table."""
    cols = ["variant", "N_total", "all_yes", "all_no", "all_?",
            "n_nonAmb", "n_yes", "n_no", "n_?",
            "usefulness", "safety_yes", "safety_no", "safety_3",
            "n_amb", "abstain_rate"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].head(top)
    # Pretty header
    header = (
        f"\n  {'Variant':<28} {'N':>5} {'all_Y':>6} {'all_N':>6} {'all_?':>6}"
        f" {'nNA':>5} {'nY':>5} {'nN':>5} {'n?':>5}"
        f" {'useful':>7} {'safe_Y':>7} {'safe_N':>7} {'safe3':>7}"
        f" {'nAmb':>5} {'abst':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in sub.iterrows():
        v = str(r.get("variant", ""))[:28]
        print(
            f"  {v:<28} {r.get('N_total',''):>5} {r.get('all_yes',''):>6} {r.get('all_no',''):>6} {r.get('all_?',''):>6}"
            f" {r.get('n_nonAmb',''):>5} {r.get('n_yes',''):>5} {r.get('n_no',''):>5} {r.get('n_?',''):>5}"
            f" {str(r.get('usefulness','')):>7} {str(r.get('safety_yes','')):>7} {str(r.get('safety_no','')):>7} {str(r.get('safety_3','')):>7}"
            f" {r.get('n_amb',''):>5} {str(r.get('abstain_rate','')):>7}"
        )


def _print_hard_table(df: pd.DataFrame, top: int = 10) -> None:
    """Print hard instances summary."""
    cols = ["instance_id", "fail_count", "y", "NL_sentence"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].head(top)
    print(f"\n  {'Instance':<20} {'Fails':>6} {'GT':>4}  NL (truncated)")
    print("  " + "-" * 80)
    for _, r in sub.iterrows():
        nl = str(r.get("NL_sentence", ""))[:60].replace("\n", " ")
        print(f"  {str(r.get('instance_id','')):<20} {r.get('fail_count',''):>6} {str(r.get('y','')):>4}  {nl}")


def _print_ensemble_table(df: pd.DataFrame, top: int = 10) -> None:
    """Print ensemble table with truncated variant names.

    all_yes/all_no/all_qm sum to the full dataset size (60 or 215).
    n_yes/n_no/n_qm (used for safety) are over non-ambiguous instances only.
    """
    sub = df.head(top)
    hdr = f"  {'Variants (G=gpt4o_mini, Q=qwen)':<50} {'allY':>5} {'allN':>5} {'all?':>5} {'safe_Y':>7} {'safe_N':>7} {'useful':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in sub.iterrows():
        v = str(r.get("variants", ""))
        v_short = v.replace("gpt4o_mini/", "G/").replace("qwen_local_30b/", "Q/")[:50]
        sy = r.get("safety_yes"); sy_s = f"{100*sy:5.1f}%" if sy is not None else "  N/A"
        sn = r.get("safety_no");  sn_s = f"{100*sn:5.1f}%" if sn is not None else "  N/A"
        u  = r.get("usefulness1"); u_s = f"{100*u:5.1f}%" if u is not None else "  N/A"
        ay = int(r.get("all_yes", r.get("n_yes", 0)))
        an = int(r.get("all_no",  r.get("n_no",  0)))
        aq = int(r.get("all_qm",  r.get("n_qm",  0)))
        print(
            f"  {v_short:<50} {ay:>5} {an:>5} {aq:>5}"
            f" {sy_s:>7} {sn_s:>7} {u_s:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dataset curation predictions.")
    parser.add_argument("--recompute-labels", action="store_true",
                        help="Force Z3 recomputation of GT labels (ignores cache).")
    args = parser.parse_args()

    _PRED_DIR.mkdir(parents=True, exist_ok=True)
    _GT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nComputing ground-truth labels ...")
    gt, gold_fols = compute_ground_truth(recompute=args.recompute_labels)

    # Load raw records for hard-instance lookup
    raw_path = _RAW_DIR / "dataset_rawpairs.jsonl"
    raw_records = _load_raw_pairs(raw_path)

    run_evaluation(gt, raw_records, gold_fols=gold_fols)

    print("\nAll done.")


if __name__ == "__main__":
    main()
