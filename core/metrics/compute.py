"""
Metric computation for E1 (verdict quality), E2 (refinement quality),
and E3 (combined quality), plus contingency tables and cost statistics.

All functions operate on a :class:`~loader.ResultSet` and return plain dicts
so they are easy to serialise and print.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loader import ResultSet


# ---------------------------------------------------------------------------
# E1 — Verdict Quality
# ---------------------------------------------------------------------------

def compute_e1(rs: ResultSet) -> Dict[str, Optional[float]]:
    """
    Compute E1 metrics for one result-set.

    Returns:
        usefulness1   – fraction of committed instances
        safety1       – precision of committed verdicts (aggregate)
        safety1_yes   – precision of 'yes' verdicts
        safety1_no    – precision of 'no' verdicts
        n             – total instances
        n_qm          – instances with q = '?'
        n_yes         – instances with q = 'yes'
        n_no          – instances with q = 'no'
    """
    recs = [r for r in rs.records if r.y in ("yes", "no")]
    N = len(recs)
    if N == 0:
        return _empty_e1()

    qm_pred = [r for r in recs if r.q == "?"]
    yes_pred   = [r for r in recs if r.q == "yes"]
    no_pred    = [r for r in recs if r.q == "no"]
    committed = [r for r in recs if r.q != "?"]

    n_qm = len(qm_pred)
    n_yes = len(yes_pred)
    n_no  = len(no_pred)
    n_committed = len(committed)

    # usefulness1: fraction of non-ambiguous instances where the model committed.
    # recs already contains only y∈{yes,no} (ambiguous instances filtered by the
    # caller), so N is exactly n_non_ambiguous.
    usefulness1 = 1 - n_qm / N

    safety1 = (
        sum(1 for r in committed if r.q == r.y) / n_committed
        if n_committed > 0 else None
    )
    safety1_yes = (
        sum(1 for r in yes_pred if r.y == "yes") / n_yes
        if n_yes > 0 else None
    )
    safety1_no = (
        sum(1 for r in no_pred if r.y == "no") / n_no
        if n_no > 0 else None
    )

    return {
        "usefulness1": usefulness1,
        "safety1":     safety1,
        "safety1_yes": safety1_yes,
        "safety1_no":  safety1_no,
        "n":           N,
        "n_committed": n_committed,
        "n_yes":       n_yes,
        "n_no":        n_no,
        "n_qm":        n_qm,
    }


def _empty_e1() -> dict:
    return {
        "usefulness1": None, "safety1": None,
        "safety1_yes": None, "safety1_no": None,
        "n": 0, "n_committed": 0, "n_yes": 0, "n_no": 0, "n_qm": 0,
    }


# ---------------------------------------------------------------------------
# E2 — Refinement Quality
# ---------------------------------------------------------------------------

def compute_e2(rs: ResultSet) -> Dict[str, Optional[float]]:
    """
    Compute E2 metrics for one result-set.

    Applicable only to methods that produce a refinement ψ (task 'B').
    Returns None for all metrics when no records have a refinement.

    Returns:
        usefulness2  – fraction of all instances where q = 'no' (ψ was produced)
        safety2      – fraction of q='no' instances where BOTH verdict AND
                       refinement are correct
        n            – total instances
        n_no_pred    – instances with q = 'no'
    """
    recs = [r for r in rs.records if r.y in ("yes", "no")]
    N = len(recs)
    if N == 0:
        return _empty_e2()

    no_pred = [r for r in recs if r.q == "no"]
    n_no = len(no_pred)

    usefulness2 = n_no / N

    # safety2: q='no' AND y='no' AND eq=1
    n_both_correct = sum(
        1 for r in no_pred
        if r.y == "no" and r.eq_i is True
    )
    safety2 = n_both_correct / n_no if n_no > 0 else None

    return {
        "usefulness2": usefulness2,
        "safety2":     safety2,
        "n":           N,
        "n_no_pred":   n_no,
    }


def _empty_e2() -> dict:
    return {"usefulness2": None, "safety2": None, "n": 0, "n_no_pred": 0}


# ---------------------------------------------------------------------------
# E3 — Combined Quality
# ---------------------------------------------------------------------------

def compute_e3(rs: ResultSet) -> Dict[str, Optional[float]]:
    """
    Compute E3 metrics for one result-set.

    For methods without ψ (task 'A'), E3 = E1 exactly.

    Returns:
        usefulness3  – identical to usefulness1
        safety3      – fraction of committed instances that are *fully correct*
        n            – total instances
        n_committed  – instances with q ≠ '?'
        n_fully_correct – instances that are fully correct and committed
    """
    recs = [r for r in rs.records if r.y in ("yes", "no")]
    N = len(recs)
    if N == 0:
        return _empty_e3()

    committed = [r for r in recs if r.q != "?"]
    n_committed = len(committed)

    usefulness3 = n_committed / N

    n_fully_correct = sum(1 for r in committed if _fully_correct(r))
    safety3 = n_fully_correct / n_committed if n_committed > 0 else None

    return {
        "usefulness3":     usefulness3,
        "safety3":         safety3,
        "n":               N,
        "n_committed":     n_committed,
        "n_fully_correct": n_fully_correct,
    }


def _fully_correct(rec) -> bool:
    """Instance-level predicate for E3."""
    if rec.q == "yes" and rec.y == "yes":
        return True
    if rec.q == "no" and rec.y == "no" and rec.eq_i is True:
        return True
    return False


def _empty_e3() -> dict:
    return {
        "usefulness3": None, "safety3": None,
        "n": 0, "n_committed": 0, "n_fully_correct": 0,
    }


# ---------------------------------------------------------------------------
# Contingency table
# ---------------------------------------------------------------------------

def contingency_table(rs: ResultSet) -> pd.DataFrame:
    """
    Build the full contingency table over (q_i, y_i, eq_i).

    Every combination is present (zero-count cells included).
    Columns: q, y, eq, count, pct_of_N
    """
    recs = [r for r in rs.records if r.y in ("yes", "no")]
    N = len(recs)

    # All possible combinations
    q_vals  = ["yes", "?", "no"]
    y_vals  = ["yes", "no"]
    eq_vals = [1, 0, None]  # 1=True, 0=False, None=N/A

    counts: dict = {}
    for q in q_vals:
        for y in y_vals:
            for eq in eq_vals:
                counts[(q, y, eq)] = 0

    for r in recs:
        eq_key = (
            None if r.eq_i is None
            else (1 if r.eq_i else 0)
        )
        key = (r.q, r.y, eq_key)
        if key in counts:
            counts[key] += 1

    rows = []
    for (q, y, eq), cnt in counts.items():
        eq_label = "N/A" if eq is None else str(eq)
        rows.append(
            {
                "q": q,
                "y": y,
                "eq": eq_label,
                "count": cnt,
                "pct_of_N": round(100 * cnt / N, 2) if N > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    # Sort for readability
    q_order  = {"yes": 0, "?": 1, "no": 2}
    y_order  = {"yes": 0, "no": 1}
    eq_order = {"N/A": 0, "1": 1, "0": 2}
    df["_qo"] = df["q"].map(q_order)
    df["_yo"] = df["y"].map(y_order)
    df["_eo"] = df["eq"].map(eq_order)
    df = df.sort_values(["_qo", "_yo", "_eo"]).drop(
        columns=["_qo", "_yo", "_eo"]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Cost statistics
# ---------------------------------------------------------------------------

def cost_stats(rs: ResultSet) -> Dict[str, float]:
    """
    Compute mean ± std for token counts and n_calls across instances.

    Returns dict with keys:
        tokens_in_mean, tokens_in_std,
        tokens_out_mean, tokens_out_std,
        tokens_total_mean, tokens_total_std,
        tokens_total_eff_mean, tokens_total_eff_std,
        n_calls_mean, n_calls_std,
        n_instances
    """
    tokens_in, tokens_out, tokens_total, tokens_eff, n_calls = [], [], [], [], []

    for r in rs.records:
        cost = r.cost
        if not cost:
            continue
        calls = cost.get("calls") or []
        nc = cost.get("n_calls", len(calls))
        tin  = sum(c.get("tokens_in", 0) for c in calls)
        tout = sum(c.get("tokens_out", 0) for c in calls)
        ttot = sum(c.get("tokens_total", tin + tout) for c in calls)
        teff = cost.get("total_tokens_eff") or ttot

        tokens_in.append(tin)
        tokens_out.append(tout)
        tokens_total.append(ttot)
        tokens_eff.append(teff)
        n_calls.append(nc)

    def _ms(arr):
        if not arr:
            return None, None
        a = np.array(arr, dtype=float)
        return float(np.mean(a)), float(np.std(a))

    ti_m, ti_s   = _ms(tokens_in)
    to_m, to_s   = _ms(tokens_out)
    tt_m, tt_s   = _ms(tokens_total)
    te_m, te_s   = _ms(tokens_eff)
    nc_m, nc_s   = _ms(n_calls)

    return {
        "tokens_in_mean":         ti_m,
        "tokens_in_std":          ti_s,
        "tokens_out_mean":        to_m,
        "tokens_out_std":         to_s,
        "tokens_total_mean":      tt_m,
        "tokens_total_std":       tt_s,
        "tokens_total_eff_mean":  te_m,
        "tokens_total_eff_std":   te_s,
        "n_calls_mean":           nc_m,
        "n_calls_std":            nc_s,
        "n_instances":            len(rs.records),
    }


# ---------------------------------------------------------------------------
# Curve helpers (for plotting)
# ---------------------------------------------------------------------------

def curve_e1(result_sets: List[ResultSet]):
    """
    Return three lists of (usefulness, safety) pairs — one per safety variant.
    Points with None safety are excluded from the curve but the usefulness
    is still valid (all-abstain corner point at safety=NaN).
    """
    agg, pos, neg = [], [], []
    for rs in result_sets:
        m = compute_e1(rs)
        u = m["usefulness1"]
        if u is None:
            continue
        agg.append((u, m["safety1"]))
        pos.append((u, m["safety1_yes"]))
        neg.append((u, m["safety1_no"]))
    return agg, pos, neg


def curve_e2(result_sets: List[ResultSet]):
    """Return list of (usefulness2, safety2) pairs."""
    pts = []
    for rs in result_sets:
        m = compute_e2(rs)
        u = m["usefulness2"]
        if u is None:
            continue
        pts.append((u, m["safety2"]))
    return pts


def curve_e3(result_sets: List[ResultSet]):
    """Return list of (usefulness3, safety3) pairs."""
    pts = []
    for rs in result_sets:
        m = compute_e3(rs)
        u = m["usefulness3"]
        if u is None:
            continue
        pts.append((u, m["safety3"]))
    return pts


# ---------------------------------------------------------------------------
# Abstention quality (for ambiguous instances with y='?')
# ---------------------------------------------------------------------------

def compute_abstention_quality(rs: ResultSet) -> Dict[str, Optional[float]]:
    """
    For instances where ground truth is '?' (ambiguous), assess whether
    the model correctly abstained.

    Returns:
        n_ambiguous        – instances with y = '?'
        n_correct_abstain  – instances with y = '?' and q = '?'
        n_overconfident    – instances with y = '?' and q ≠ '?'
        abstain_rate       – n_correct_abstain / n_ambiguous
    """
    amb = [r for r in rs.records if r.y == "?"]
    n_amb = len(amb)
    if n_amb == 0:
        return {
            "n_ambiguous": 0,
            "n_correct_abstain": 0,
            "n_overconfident": 0,
            "abstain_rate": None,
        }
    n_correct = sum(1 for r in amb if r.q == "?")
    return {
        "n_ambiguous":       n_amb,
        "n_correct_abstain": n_correct,
        "n_overconfident":   n_amb - n_correct,
        "abstain_rate":      n_correct / n_amb,
    }
