"""
Aggregator: combine verdicts from multiple ResultSets and evaluate the ensemble.

Rules
-----
  pair (n=2):          both agree  → that verdict; any disagreement → '?'
  triplet_safe (n=3):  all 3 agree → that verdict; any disagreement → '?'
  majority (n≥2):      strict majority (>n/2 votes, ignoring '?') of one class
                       → that class; tie or no majority → '?'

For n=3, both 'triplet_safe' and 'majority' ResultSets are produced by
enumerate_aggregations (two operating points per triple).
For n>3, only 'majority' is generated.

Missing instances: if an instance_id is absent from a constituent run,
its verdict is treated as '?' (unknown, not committed).

Cost reporting: each aggregated RunRecord carries a merged cost dict that
sums token usage from all constituent methods, so cost_stats() works
unchanged on an AggregatedResultSet.

Usage
-----
    from core.metrics.aggregator import evaluate_aggregations, wrongly_labelled

    report = evaluate_aggregations(pool, output_dir="eval/aggregations")
    for row in report["top_composite"]:
        print(row["run_id"], row["composite"])
        bad = wrongly_labelled(report["agg_result_sets"][row["run_id"]])
        for b in bad:
            print("  WRONG:", b)
"""
from __future__ import annotations

import dataclasses as _dc
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .loader import ResultSet, RunRecord
from .compute import compute_e1, cost_stats


# ---------------------------------------------------------------------------
# Verdict aggregation rules
# ---------------------------------------------------------------------------

def _agg_pair(q1: str, q2: str) -> str:
    """Both agree → that verdict; any disagreement (including '?') → '?'."""
    return q1 if q1 == q2 else "?"


def _agg_safe(verdicts: list[str]) -> str:
    """Absolute concordance across all members — all same → verdict; else '?'."""
    unique = set(verdicts)
    return unique.pop() if len(unique) == 1 else "?"


def _agg_majority(verdicts: list[str]) -> str:
    """
    Strict majority: a class must receive strictly more than n/2 votes.
    '?' votes are excluded from counting (they do not count for any class).
    If no class reaches the threshold → '?'.
    """
    n = len(verdicts)
    committed = [v for v in verdicts if v != "?"]
    counts = Counter(committed)
    for label in ("yes", "no"):
        if counts[label] > n / 2:
            return label
    return "?"


# ---------------------------------------------------------------------------
# Cost merging
# ---------------------------------------------------------------------------

def _merge_costs(costs: list[dict | None]) -> dict:
    """
    Sum token costs from all constituent methods for one instance.

    The resulting dict is structurally identical to a single-method cost dict,
    so cost_stats() works unchanged on the aggregated ResultSet.

    Critically, when no constituent has actual cost data (all inputs are falsy
    empty dicts or None), an empty dict {} is returned — preserving the falsy
    signal that cost_stats() uses to skip records with no cost information.
    Returning a non-empty dict with zero values would cause cost_stats() to
    report 0.0 means instead of None.
    """
    all_calls: list = []
    total_n_calls = 0
    total_eff = 0
    any_data = False          # tracks whether at least one constituent had cost data
    for cost in costs:
        if not cost:
            continue
        any_data = True
        calls = cost.get("calls") or []
        all_calls.extend(calls)
        total_n_calls += cost.get("n_calls", len(calls))
        teff = cost.get("total_tokens_eff") or sum(
            c.get("tokens_total", c.get("tokens_in", 0) + c.get("tokens_out", 0))
            for c in calls
        )
        total_eff += teff
    if not any_data:
        return {}             # falsy → cost_stats skips this record → means are None
    return {
        "calls":            all_calls,
        "n_calls":          total_n_calls,
        "total_tokens_eff": total_eff,
    }


# ---------------------------------------------------------------------------
# Build one aggregated ResultSet
# ---------------------------------------------------------------------------

def build_aggregated_result_set(
    result_sets: list[ResultSet],
    mode: str,
) -> ResultSet:
    """
    Aggregate verdicts from *result_sets* per instance_id.

    Parameters
    ----------
    result_sets : pool to combine, length ≥ 2
    mode        : 'pair'          — same-or-'?' (n must be 2)
                  'triplet_safe'  — all-same-or-'?' (n must be 3)
                  'majority'      — strict majority, any n ≥ 2

    Returns
    -------
    A new ResultSet whose records carry aggregated verdicts and merged costs.
    instance_ids absent from any constituent run are treated as '?'.
    """
    n = len(result_sets)
    if n < 2:
        raise ValueError("Need at least 2 ResultSets to aggregate.")
    if mode == "pair" and n != 2:
        raise ValueError("'pair' mode requires exactly 2 ResultSets.")
    if mode == "triplet_safe" and n != 3:
        raise ValueError("'triplet_safe' mode requires exactly 3 ResultSets.")

    # Index each ResultSet by instance_id for O(1) lookup
    indices: list[dict[str, RunRecord]] = [
        {r.instance_id: r for r in rs.records}
        for rs in result_sets
    ]

    # Union of all instance_ids (instances missing from some runs are included
    # with their verdict treated as '?')
    all_ids = sorted(set().union(*(idx.keys() for idx in indices)))

    r0 = result_sets[0]
    members = [rs.run_id for rs in result_sets]
    agg_run_id = f"agg_{mode}_" + "__".join(members)

    agg_records: list[RunRecord] = []
    for iid in all_ids:
        recs: list[RunRecord | None] = [idx.get(iid) for idx in indices]
        verdicts = [r.q if r is not None else "?" for r in recs]
        costs    = [r.cost if r is not None else None for r in recs]

        if mode == "pair":
            q_agg = _agg_pair(verdicts[0], verdicts[1])
        elif mode == "triplet_safe":
            q_agg = _agg_safe(verdicts)
        else:  # majority (n=3 or n>3)
            q_agg = _agg_majority(verdicts)

        # Ground truth and eq_i: take from the first non-None record
        # (all constituent runs share the same dataset, so y is the same)
        y    = next((r.y    for r in recs if r is not None), "?")
        eq_i = next((r.eq_i for r in recs if r is not None), None)
        base = next( r      for r in recs if r is not None)

        agg_records.append(
            _dc.replace(
                base,
                run_id=agg_run_id,
                q=q_agg,
                y=y,
                eq_i=eq_i,
                cost=_merge_costs(costs),
            )
        )

    return ResultSet(
        run_id=agg_run_id,
        method=f"AGG[{mode}]",
        task=r0.task,
        hyperparams={"mode": mode, "members": members},
        llm_name=None,
        records=agg_records,
    )


# ---------------------------------------------------------------------------
# Enumerate all pairs + triplets (+ arbitrary n via majority)
# ---------------------------------------------------------------------------

def enumerate_aggregations(
    pool: list[ResultSet],
) -> list[tuple[ResultSet, str]]:
    """
    Generate all pair and triplet aggregations from *pool*.

    - Pairs  (n=2): one ResultSet per pair using the 'pair' rule.
    - Triplets (n=3): two ResultSets per triple — 'triplet_safe' and 'majority'.
    - Larger groups are not enumerated here; use build_aggregated_result_set
      directly with mode='majority' for n>3.

    Returns
    -------
    List of (AggregatedResultSet, mode_label) tuples.
    """
    results: list[tuple[ResultSet, str]] = []

    for rs1, rs2 in itertools.combinations(pool, 2):
        agg = build_aggregated_result_set([rs1, rs2], mode="pair")
        results.append((agg, "pair"))

    for rs1, rs2, rs3 in itertools.combinations(pool, 3):
        for mode in ("triplet_safe", "majority"):
            agg = build_aggregated_result_set([rs1, rs2, rs3], mode=mode)
            results.append((agg, mode))

    return results


# ---------------------------------------------------------------------------
# Ranking table
# ---------------------------------------------------------------------------

def rank_aggregations(
    aggregations: list[tuple[ResultSet, str]],
) -> list[dict]:
    """
    Compute E1 metrics and cost statistics for every aggregation and return
    a table sorted by composite score (descending).

    Metrics
    -------
    safety_score = (S1_yes + S1_no) / 2   — equal-weight class safety.
                   None values are replaced with 0.0 for ranking purposes.
    composite    = (safety_score + usefulness1) / 2

    Cost
    ----
    cost_tokens_total_eff_mean — mean total tokens (all methods combined) per instance
    cost_n_calls_mean          — mean number of LLM calls per instance
    """
    rows = []
    for agg_rs, mode in aggregations:
        m1 = compute_e1(agg_rs)
        cs = cost_stats(agg_rs)

        s_yes = m1["safety1_yes"] if m1["safety1_yes"] is not None else 0.0
        s_no  = m1["safety1_no"]  if m1["safety1_no"]  is not None else 0.0
        safety_score = (s_yes + s_no) / 2
        u1        = m1["usefulness1"] if m1["usefulness1"] is not None else 0.0
        composite = (safety_score + u1) / 2

        rows.append({
            "run_id":                      agg_rs.run_id,
            "mode":                        mode,
            "members":                     agg_rs.hyperparams.get("members", []),
            "n":                           m1["n"],
            "n_committed":                 m1["n_committed"],
            "n_yes":                       m1["n_yes"],
            "n_no":                        m1["n_no"],
            "n_qm":                        m1["n_qm"],
            "usefulness1":                 u1,
            "safety1":                     m1["safety1"],
            "safety1_yes":                 m1["safety1_yes"],
            "safety1_no":                  m1["safety1_no"],
            "safety_score":                safety_score,
            "composite":                   composite,
            "cost_tokens_total_eff_mean":  cs.get("tokens_total_eff_mean"),
            "cost_tokens_total_eff_std":   cs.get("tokens_total_eff_std"),
            "cost_n_calls_mean":           cs.get("n_calls_mean"),
            "cost_n_calls_std":            cs.get("n_calls_std"),
        })

    return sorted(rows, key=lambda r: r["composite"], reverse=True)


# ---------------------------------------------------------------------------
# Wrongly labelled instances
# ---------------------------------------------------------------------------

def wrongly_labelled(rs: ResultSet) -> dict:
    """
    Return committed instances where the verdict was wrong, split into two
    disjoint categories that mirror the scoping used by compute_e1.

    Parameters
    ----------
    rs : a ResultSet (original or aggregated).

    Returns
    -------
    dict with keys:
        errors        — y ∈ {yes, no}, q ≠ '?', q ≠ y.
                        These are the actual metric errors counted against
                        safety in E1.  n_errors is consistent with S1.
        overconfident — y = '?', q ≠ '?'.
                        The model committed on an ambiguous instance.
                        These are NOT counted by compute_e1 (which filters
                        to y ∈ {yes, no}), so they never appear in safety
                        metrics — reporting them separately avoids confusion.
        members       — constituent run_ids (for aggregated ResultSets).
    """
    members = rs.hyperparams.get("members", [])

    errors: list[dict] = []
    overconfident: list[dict] = []

    for r in rs.records:
        if r.q == "?":
            continue
        if r.y in ("yes", "no"):
            if r.q != r.y:
                errors.append({"instance_id": r.instance_id, "q": r.q, "y": r.y})
        elif r.y == "?":
            overconfident.append({"instance_id": r.instance_id, "q": r.q, "y": r.y})

    return {
        "members":       members,
        "n_errors":      len(errors),
        "errors":        errors,
        "n_overconfident": len(overconfident),
        "overconfident": overconfident,
    }


# ---------------------------------------------------------------------------
# Short label helper for plot annotations
# ---------------------------------------------------------------------------

def _short_label(run_id: str) -> str:
    """
    Produce a concise plot label from a long aggregation run_id.

    Input:  'agg_pair_B1_v1_B_gpt4o_mini_s0__B2_v3_fs5_B_gpt4o_mini_s0'
    Output: 'B1v1B+B2v3fs5'

    Takes the first 3 underscore-separated segments of each member run_id
    (method, version/hp1, task/hp2) and joins them without separators.
    """
    try:
        # Strip mode prefix: 'agg_pair_', 'agg_triplet_safe_', 'agg_majority_'
        rest = run_id
        for prefix in ("agg_triplet_safe_", "agg_majority_", "agg_pair_"):
            if rest.startswith(prefix):
                rest = rest[len(prefix):]
                break

        members = rest.split("__")
        abbrevs = []
        for m in members:
            segs = m.split("_")
            # Take up to 3 segments: method + version/hp + task/hp2
            abbrevs.append("".join(segs[:3]))
        return "+".join(abbrevs)
    except Exception:
        return run_id[:20]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_CAT_META = {
    "safety":     {"color": "#d62728", "marker": "^", "label": "Top safety"},
    "usefulness": {"color": "#1f77b4", "marker": "s", "label": "Top usefulness"},
    "composite":  {"color": "#2ca02c", "marker": "D", "label": "Top composite"},
}


def plot_aggregations(
    table: list[dict],
    output_path: str | Path,
    top_n: int = 5,
    title_prefix: str = "",
) -> Path:
    """
    Scatter plot of all aggregations in the Safety–Usefulness plane.

    X-axis : Usefulness₁
    Y-axis : Safety score  (mean of S₁_yes and S₁_no)

    Rendering
    ---------
    - All aggregations : light grey dots (background).
    - Top *top_n* by safety_score  → red triangles, labeled.
    - Top *top_n* by usefulness1   → blue squares, labeled.
    - Top *top_n* by composite     → green diamonds, labeled.
    - A point in multiple top lists is drawn with all matching markers
      stacked (slightly offset) so its membership is visible.
    - Each highlighted point is annotated with a short label.
      Points in multiple categories are annotated once.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not table:
        return output_path

    xs_all  = [r["usefulness1"]  for r in table]
    ys_all  = [r["safety_score"] for r in table]
    ids_all = [r["run_id"]       for r in table]

    # Identify top sets per category
    top_ids: dict[str, set[str]] = {
        "safety":     {r["run_id"] for r in sorted(table, key=lambda r: r["safety_score"],  reverse=True)[:top_n]},
        "usefulness": {r["run_id"] for r in sorted(table, key=lambda r: r["usefulness1"],   reverse=True)[:top_n]},
        "composite":  {r["run_id"] for r in sorted(table, key=lambda r: r["composite"],     reverse=True)[:top_n]},
    }
    highlighted = set().union(*top_ids.values())

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(
        f"{title_prefix}Aggregation ensembles — Safety vs Usefulness".strip(),
        fontsize=13,
    )

    # --- Background: non-highlighted points ----------------------------------
    bg_xs = [x for x, rid in zip(xs_all, ids_all) if rid not in highlighted]
    bg_ys = [y for y, rid in zip(ys_all, ids_all) if rid not in highlighted]
    ax.scatter(bg_xs, bg_ys, color="#cccccc", s=20, zorder=2, label="_nolegend_")

    # --- Highlighted points: one scatter per category for legend -------------
    # For points in multiple categories, draw stacked markers with small offsets
    offset_map = {
        "safety":     ( 0.000,  0.000),
        "usefulness": ( 0.004, -0.004),
        "composite":  (-0.004,  0.004),
    }

    for cat, meta in _CAT_META.items():
        cat_xs, cat_ys = [], []
        dx, dy = offset_map[cat]
        for x, y, rid in zip(xs_all, ys_all, ids_all):
            if rid in top_ids[cat]:
                cat_xs.append(x + dx)
                cat_ys.append(y + dy)
        if cat_xs:
            ax.scatter(
                cat_xs, cat_ys,
                color=meta["color"],
                marker=meta["marker"],
                s=130, zorder=5,
                label=f"{meta['label']} (n={top_n})",
                edgecolors="black", linewidths=0.6,
            )

    # --- Annotations: one label per highlighted point ------------------------
    labeled: set[str] = set()
    for x, y, rid in zip(xs_all, ys_all, ids_all):
        if rid not in highlighted or rid in labeled:
            continue
        lbl = _short_label(rid)
        ax.annotate(
            lbl,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6.0,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0),
        )
        labeled.add(rid)

    # --- Axes and formatting -------------------------------------------------
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Usefulness₁", fontsize=11)
    ax.set_ylabel("Safety score  (mean of S₁_yes, S₁_no)", fontsize=11)
    ax.set_title(
        f"All aggregations (N={len(table)})  ·  highlighted = top {top_n} per criterion",
        fontsize=9,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0, zorder=1)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_aggregations(
    pool: list[ResultSet],
    output_dir: str | Path | None = None,
    top_n: int = 5,
    title_prefix: str = "",
    verbose: bool = True,
) -> dict:
    """
    Enumerate all pair + triplet aggregations from *pool*, rank them,
    generate the scatter plot, and save JSON reports.

    Parameters
    ----------
    pool       : list of ResultSets to combine (curated by the caller).
    output_dir : directory for artefacts; skipped if None.
    top_n      : number of top aggregations to highlight per criterion.
    title_prefix : prepended to plot title (e.g. dataset name).
    verbose    : print progress messages.

    Returns
    -------
    dict with keys:
        table           – all aggregations ranked by composite (list of dicts)
        top_safety      – top *top_n* by safety_score
        top_usefulness  – top *top_n* by usefulness1
        top_composite   – top *top_n* by composite
        agg_result_sets – dict run_id → ResultSet (for wrongly_labelled lookup)
        output_dir      – str path where artefacts were saved (or None)
    """
    if verbose:
        print(f"Enumerating pair + triplet aggregations from pool of "
              f"{len(pool)} ResultSets …")

    aggregations = enumerate_aggregations(pool)

    if verbose:
        n_pairs    = sum(1 for _, m in aggregations if m == "pair")
        n_t_safe   = sum(1 for _, m in aggregations if m == "triplet_safe")
        n_majority = sum(1 for _, m in aggregations if m == "majority")
        print(f"  {len(aggregations)} aggregations: "
              f"{n_pairs} pairs, {n_t_safe} triplet-safe, {n_majority} majority")

    table = rank_aggregations(aggregations)

    # Lookup map for wrongly_labelled queries
    agg_rs_map: dict[str, ResultSet] = {
        agg_rs.run_id: agg_rs for agg_rs, _ in aggregations
    }

    top_safety     = sorted(table, key=lambda r: r["safety_score"],  reverse=True)[:top_n]
    top_usefulness = sorted(table, key=lambda r: r["usefulness1"],   reverse=True)[:top_n]
    top_composite  = sorted(table, key=lambda r: r["composite"],     reverse=True)[:top_n]

    if verbose and top_composite:
        best = top_composite[0]
        print(f"  Best composite: {_short_label(best['run_id'])}  "
              f"composite={best['composite']:.3f}  "
              f"safety={best['safety_score']:.3f}  "
              f"usefulness={best['usefulness1']:.3f}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full ranking table as JSON
        table_path = output_dir / "aggregation_ranking.json"
        with table_path.open("w") as fh:
            json.dump(table, fh, indent=2, default=str)
        if verbose:
            print(f"  Ranking table → {table_path}")

        # Wrongly labelled for every distinct top aggregation
        seen_ids: set[str] = set()
        wrong_report: list[dict] = []
        for row in top_safety + top_usefulness + top_composite:
            rid = row["run_id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rs    = agg_rs_map[rid]
            wrong = wrongly_labelled(rs)
            wrong_report.append({
                "run_id":          rid,
                "mode":            row["mode"],
                "members":         row["members"],
                "safety_score":    row["safety_score"],
                "usefulness1":     row["usefulness1"],
                "composite":       row["composite"],
                # errors: y∈{yes,no}, q wrong — consistent with E1 safety
                "n_errors":        wrong["n_errors"],
                "errors":          wrong["errors"],
                # overconfident: y='?', model committed — separate from E1
                "n_overconfident": wrong["n_overconfident"],
                "overconfident":   wrong["overconfident"],
            })
        wrong_path = output_dir / "wrongly_labelled_top.json"
        with wrong_path.open("w") as fh:
            json.dump(wrong_report, fh, indent=2, default=str)
        if verbose:
            print(f"  Wrongly labelled → {wrong_path}")

        # Scatter plot
        plot_path = output_dir / "plot_aggregations.png"
        plot_aggregations(
            table,
            output_path=plot_path,
            top_n=top_n,
            title_prefix=title_prefix,
        )
        if verbose:
            print(f"  Plot → {plot_path}")

    return {
        "table":           table,
        "top_safety":      top_safety,
        "top_usefulness":  top_usefulness,
        "top_composite":   top_composite,
        "agg_result_sets": agg_rs_map,
        "output_dir":      str(output_dir) if output_dir else None,
    }
