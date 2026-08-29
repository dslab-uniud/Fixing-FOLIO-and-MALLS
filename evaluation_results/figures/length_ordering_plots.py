"""length_ordering_plots.py — Plots + AUC/T90/T95 table for length-based orderings.

Uses the records cache built by length_ordering_analysis.py, plus the existing
pipeline1/pipeline2 human_oversight CSVs, to compare (for gemma, full dataset):

  - P1 band: B1-B3 x pv1,pv3 (6 variants)
  - P2: best single variant (by AUC)
  - Original-dataset ordering, longest-first, by FOL_sentence_old length
  - Original-dataset ordering, longest-first, by NL_sentence length
  - Green-baseline (phi_hat) ordering, longest-first, by phi_hat length
  - Green-baseline (phi_hat) ordering, longest-first, by NL_sentence length

Scoped to FOLIO and MALLS only — GGC/Stanford is a private dataset and is
excluded from this public release.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from length_ordering_analysis import build_length_curve, auc_step, find_threshold

OUT_DIR = Path(__file__).parent / "length_ordering_results"

P1_COLOR = "#1f77b4"
P2_COLOR = "#d62728"
BLACK_COLOR = "#222222"
ORIG_FOL_COLOR = BLACK_COLOR
ORIG_NL_COLOR = BLACK_COLOR
GREEN_FOL_COLOR = "#2ca02c"
GREEN_NL_COLOR = "#98df8a"

GRID = np.linspace(0.0, 1.0, 201)

DS_MAP_P1 = {"FOLIO": "folio_validation", "MALLS": "malls"}
DS_MAP_P2 = {"FOLIO": "folio", "MALLS": "malls"}


def _curve_xy(row, mode="full"):
    xs = [0.0, row.get(f"{mode}_x1"), row.get(f"{mode}_x2"), row.get(f"{mode}_x3"), 1.0]
    ys = [row.get(f"{mode}_y0"), row.get(f"{mode}_y1"), row.get(f"{mode}_y2"),
          row.get(f"{mode}_y3"), 1.0]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in xs + ys):
        return None
    xs = list(xs)
    for i in range(1, len(xs)):
        if xs[i] < xs[i - 1]:
            xs[i] = xs[i - 1]
    return np.array(xs), np.array(ys)


def load_p1_band(ds_label):
    """Return matrix of interpolated curves for gemma, B1-B3 x pv1,pv3, given dataset."""
    p1 = pd.read_csv(ROOT / "evaluation_results/pipeline1/human_oversight/all_datasets_oversight.csv")
    p1 = p1[p1["method"].isin(["B1", "B2", "B3", "B4"])].copy()
    p1 = p1[~p1["variant"].str.contains("fs3|fs5", na=False)]
    sub = p1[(p1["model"] == "gemma_4_31b") & (p1["dataset"] == DS_MAP_P1[ds_label])]
    sub = sub[sub["method"].isin(["B1", "B2", "B3"]) & sub["pv"].isin(["pv1", "pv3"])]
    mats = []
    for _, r in sub.iterrows():
        c = _curve_xy(r)
        if c is not None:
            mats.append(np.interp(GRID, c[0], c[1]))
    return np.vstack(mats) if mats else np.zeros((0, len(GRID)))


def load_p2_best(ds_label):
    """Return (xs, ys) of the single best-AUC P2 variant for gemma, given dataset."""
    p2 = pd.read_csv(ROOT / "evaluation_results/pipeline2/human_oversight/all_datasets_oversight.csv")
    p2 = p2[(p2["phi_model"] == p2["judge_model"]) & (p2["phi_prompt"] == "P1")
            & p2["baseline"].isin(["B1", "B2", "B3", "B4"])].copy()
    b14 = p2["baseline"].isin(["B1", "B4"]) & (p2["task_tag"] == "B")
    b23 = p2["baseline"].isin(["B2", "B3"]) & (p2["task_tag"] == "fs5_B")
    p2 = p2[b14 | b23].copy()
    sub = p2[(p2["judge_model"] == "gemma") & (p2["source_dataset"] == DS_MAP_P2[ds_label])]
    if sub.empty:
        return None
    best_row = sub.loc[sub["full_area"].idxmax()]
    return _curve_xy(best_row), best_row


def build_all_curves():
    with open(OUT_DIR / "_records_cache.pkl", "rb") as f:
        all_records = pickle.load(f)

    curves = {}  # (dataset, curve_name) -> (xs, ys)
    for ds in ["FOLIO", "MALLS"]:
        recs = all_records[ds]
        c = build_length_curve(recs, "input_len", "orig_correct")
        if c:
            curves[(ds, "orig_fol_len")] = c
        c = build_length_curve(recs, "nl_len", "orig_correct")
        if c:
            curves[(ds, "orig_nl_len")] = c
        c = build_length_curve(recs, "phi_hat_len", "green_correct")
        if c:
            curves[(ds, "green_fol_len")] = c
        c = build_length_curve(recs, "nl_len", "green_correct")
        if c:
            curves[(ds, "green_nl_len")] = c
    return curves


def main():
    curves = build_all_curves()

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("LENGTH-ORDERING STRATEGIES vs PIPELINE 1 / PIPELINE 2 — gemma, full dataset")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("P1 = band over B1-B3 x pv1,pv3 (6 variants), mean curve used for AUC/T90/T95.")
    report_lines.append("P2 = single best-AUC variant.")
    report_lines.append("Orig-*-len = review FOL_sentence_old longest-first.")
    report_lines.append("Green-*-len = review phi_hat (Stage-1 direct translation) longest-first.")
    report_lines.append("")

    header = f"{'Dataset':<8}{'Curve':<16}{'AUC':>8}{'T_90':>8}{'T_95':>8}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    ds_list = ["FOLIO", "MALLS"]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(20 * len(ds_list) / 3, 6.5))
    if len(ds_list) == 1:
        axes = [axes]

    for ds in ds_list:
        ax = axes[ds_list.index(ds)]

        mat_p1 = load_p1_band(ds)
        if mat_p1.shape[0] > 0:
            mu = mat_p1.mean(0)
            sd = mat_p1.std(0)
            ax.fill_between(GRID, mu - sd, mu + sd, color=P1_COLOR, alpha=0.15, linewidth=0)
            ax.plot(GRID, mu, color=P1_COLOR, lw=2.3, label=f"P1 (B1-B3 x pv1,pv3, n={mat_p1.shape[0]})")
            auc = auc_step(GRID, mu)
            t90 = find_threshold(GRID, mu, 0.90)
            t95 = find_threshold(GRID, mu, 0.95)
            report_lines.append(f"{ds:<8}{'P1':<16}{auc:>8.4f}{t90:>8.4f}{t95:>8.4f}")

        p2_result = load_p2_best(ds)
        if p2_result is not None and p2_result[0] is not None:
            (xs2, ys2), best_row = p2_result
            ax.plot(xs2, ys2, color=P2_COLOR, lw=2.0, ls="-",
                     label=f"P2 best ({best_row['baseline']}_{best_row['pv']})")
            mu2 = np.interp(GRID, xs2, ys2)
            auc = auc_step(GRID, mu2)
            t90 = find_threshold(GRID, mu2, 0.90)
            t95 = find_threshold(GRID, mu2, 0.95)
            report_lines.append(f"{ds:<8}{'P2':<16}{auc:>8.4f}{t90:>8.4f}{t95:>8.4f}")

        curve_specs = [
            ("orig_fol_len", ORIG_FOL_COLOR, "-", "Orig-dataset, FOL length"),
            ("orig_nl_len", ORIG_NL_COLOR, "--", "Orig-dataset, NL length"),
            ("green_fol_len", GREEN_FOL_COLOR, "-", "Green (phi_hat), FOL length"),
            ("green_nl_len", GREEN_NL_COLOR, "--", "Green (phi_hat), NL length"),
        ]
        for key, color, ls, label in curve_specs:
            c = curves.get((ds, key))
            if c is None:
                continue
            xs, ys = c
            ax.plot(xs, ys, color=color, lw=1.8, ls=ls, label=label, alpha=0.9)
            auc = auc_step(xs, ys)
            t90 = find_threshold(xs, ys, 0.90)
            t95 = find_threshold(xs, ys, 0.95)
            report_lines.append(f"{ds:<8}{key:<16}{auc:>8.4f}{t90:>8.4f}{t95:>8.4f}")

        report_lines.append("")

        for thresh in [0.90, 0.95]:
            ax.axhline(thresh, ls="--", lw=0.7, color="gray", alpha=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0.4, 1.01)
        ax.set_title(f"Gemma — {ds}", fontsize=12)
        ax.set_xlabel("Human oversight (fraction reviewed)", fontsize=10)
        if ds == ds_list[0]:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(
        "Length-based review ordering vs Pipeline 1 / Pipeline 2 — Gemma, full dataset\n"
        "Longest-first review order; step curves from per-instance correctness",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / "fig_length_ordering_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    report_path = OUT_DIR / "length_ordering_auc_table.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"Saved {report_path}")
    print()
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
