"""generate_paper_plots.py — Produce paper figures for Claim 1 and Claim 3.

Claim 1: P1 > P2 > Green  (area metric, statistical significance shown)
  → fig_claim1_area_bars.png : 2×2 grid (model × dataset), 3 bars (Green/P2/P1),
      stars for P1>Green, P2>Green, P1>P2

Claim 3: Gemma P1 — band plot restricted to best variants; P2 best single curve
  → fig_claim3_gemma_bands_best_p2.png : oversight band for P1, best-AUC curve
      for P2, FOLIO and MALLS

Appendix:
  → fig_appendix_aag_bars.png, fig_appendix_t90_bars.png, fig_appendix_t95_bars.png

Scoped to FOLIO and MALLS only — GGC/Stanford is a private dataset and is
excluded from this public release (data, intermediate NL sentences, and LLM
responses are not redistributed).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]   # DatasetCuration_public/
_BASE = Path(__file__).parent.parent          # evaluation_results/
OUT   = Path(__file__).parent                 # evaluation_results/figures/

# ── constants ──────────────────────────────────────────────────────────────────
# Canonical green-line y0 = direct LLM translation accuracy (Stage 1: phi_hat_generation).
GREEN_Y0 = {
    ("gemma_4_31b",    "folio_validation", "full"):  0.7418,
    ("gemma_4_31b",    "folio_validation", "noamb"): 0.8274,
    ("gemma_4_31b",    "malls",            "full"):  0.8600,
    ("gemma_4_31b",    "malls",            "noamb"): 0.9388,
    ("gpt4o_mini",     "folio_validation", "full"):  0.6145,
    ("gpt4o_mini",     "folio_validation", "noamb"): 0.6903,
    ("gpt4o_mini",     "malls",            "full"):  0.5700,
    ("gpt4o_mini",     "malls",            "noamb"): 0.5510,
    ("qwen_local_30b", "folio_validation", "full"):  0.5855,
    ("qwen_local_30b", "folio_validation", "noamb"): 0.6504,
    ("qwen_local_30b", "malls",            "full"):  0.6800,
    ("qwen_local_30b", "malls",            "noamb"): 0.6531,
}

MODEL_MAP_P1 = {"gpt4o_mini": "GPT-4o-mini", "qwen_local_30b": "Qwen", "gemma_4_31b": "Gemma"}
MODEL_MAP_P2 = {"gemma": "Gemma", "gpt4omini": "GPT-4o-mini", "qwen3": "Qwen"}
P2_TO_P1_TOK = {"gemma": "gemma_4_31b", "gpt4omini": "gpt4o_mini", "qwen3": "qwen_local_30b"}
P2_DS_TO_P1  = {"folio": "folio_validation", "malls": "malls"}

MODEL_KEY_P1 = {"Gemma": "gemma_4_31b", "GPT-4o-mini": "gpt4o_mini", "Qwen": "qwen_local_30b"}

MODELS   = ["Gemma", "GPT-4o-mini", "Qwen"]
DATASETS = ["folio_validation", "malls"]
DS_LABEL = {"folio_validation": "FOLIO", "malls": "MALLS"}
BASELINES = ["B1", "B2", "B3", "B4"]
PVS       = ["pv1", "pv2", "pv3", "pv4"]
GRID      = np.linspace(0.0, 1.0, 201)

P1_COLOR    = "#1f77b4"
P2_COLOR    = "#d62728"
GREEN_COLOR = "#2ca02c"
BLACK_COLOR = "#222222"


def _load_is_original_correct_counts(paths):
    X = Y = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                X += 1
                if o.get("is_original_correct") == "no":
                    Y += 1
    return X, Y


_FOLIO_RAW = [_ROOT / "core/dataset/datasets/folio/dataset_rawpairs.jsonl"]
_MALLS_RAW = [_ROOT / "core/dataset/datasets/malls/dataset_rawpairs.jsonl"]

_folio_X, _folio_Y = _load_is_original_correct_counts(_FOLIO_RAW)
_malls_X, _malls_Y = _load_is_original_correct_counts(_MALLS_RAW)

# Black-line y0: human annotator reviewing original dataset from scratch.
BLACK_Y0 = {
    ("folio_validation", "full"): round(1.0 - _folio_Y / _folio_X, 4),
    ("malls",            "full"): round(1.0 - _malls_Y / _malls_X, 4),
}

# ── load data ──────────────────────────────────────────────────────────────────
p1 = pd.read_csv(_BASE / "pipeline1/human_oversight/all_datasets_oversight.csv")
p1 = p1[p1["method"].isin(["B1", "B2", "B3", "B4"])].copy()
p1 = p1[~p1["variant"].str.contains("fs3|fs5", na=False)]
p1["_model"]     = p1["model"].map(MODEL_MAP_P1)
p1["_ds_raw"]    = p1["dataset"]
p1["_model_tok"] = p1["model"]   # P1 token used in GREEN_Y0

p2 = pd.read_csv(_BASE / "pipeline2/human_oversight/all_datasets_oversight.csv")
p2 = p2[(p2["phi_model"] == p2["judge_model"]) & (p2["phi_prompt"] == "P1")
        & p2["baseline"].isin(["B1", "B2", "B3", "B4"])].copy()
# B2/B3 use 5-shot examples (fs5_B); exclude fs3_B to keep 16 variants per cell
b14 = p2["baseline"].isin(["B1", "B4"]) & (p2["task_tag"] == "B")
b23 = p2["baseline"].isin(["B2", "B3"]) & (p2["task_tag"] == "fs5_B")
p2  = p2[b14 | b23].copy()
p2["_model"]     = p2["judge_model"].map(MODEL_MAP_P2)
p2["_model_tok"] = p2["judge_model"].map(P2_TO_P1_TOK)
p2["_ds_raw"]    = p2["source_dataset"].map(P2_DS_TO_P1)

# Rename P2 'baseline' → 'method' to match P1 convention
p2 = p2.rename(columns={"baseline": "method"})

# Add AAG (area above green) and AAB (area above black)
for df in [p1, p2]:
    for mode in ["full", "noamb"]:
        df[f"{mode}_aag"] = df.apply(
            lambda r, m=mode: r[f"{m}_area"] - (GREEN_Y0.get((r["_model_tok"], r["_ds_raw"], m), np.nan) + 1.0) / 2.0,
            axis=1)
        df[f"{mode}_aab"] = df.apply(
            lambda r, m=mode: r[f"{m}_area"] - (BLACK_Y0.get((r["_ds_raw"], m), np.nan) + 1.0) / 2.0,
            axis=1)


# ── helpers ────────────────────────────────────────────────────────────────────
def _green_scalar(p1tok, ds_raw, mode, metric):
    """Single float for the green line's scalar value for a given metric."""
    y0 = GREEN_Y0.get((p1tok, ds_raw, mode))
    if y0 is None:
        return np.nan
    if metric == "area":  return (y0 + 1.0) / 2.0
    if metric == "aag":   return 0.0
    t = 0.90 if metric == "T_90" else 0.95
    return (t - y0) / (1.0 - y0) if y0 < t else 0.0


def _black_scalar(ds_raw, mode, metric):
    """Single float for the black baseline's scalar value for a given metric."""
    y0 = BLACK_Y0.get((ds_raw, mode))
    if y0 is None:
        return np.nan
    if metric == "area":  return (y0 + 1.0) / 2.0
    if metric == "aab":   return 0.0
    t = 0.90 if metric == "T_90" else 0.95
    return (t - y0) / (1.0 - y0) if y0 < t else 0.0


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _wilcoxon_onesided(diffs, alt="greater"):
    """Paired Wilcoxon signed-rank on non-NaN, non-zero diffs."""
    d = diffs[~np.isnan(diffs)]
    d = d[d != 0]
    if len(d) < 4:
        return 1.0
    try:
        _, p = stats.wilcoxon(d, alternative=alt)
        return float(p)
    except Exception:
        return 1.0


def _paired_wilcoxon_p1_p2(p1_df, p2_df, col, higher=True):
    """Paired Wilcoxon for P1 vs P2, matched on (method, pv) within a cell."""
    p1_idx = p1_df.set_index(["method", "pv"])[col]
    p2_idx = p2_df.set_index(["method", "pv"])[col]
    common = p1_idx.index.intersection(p2_idx.index)
    if len(common) < 4:
        return 1.0
    diffs = (p1_idx.loc[common] - p2_idx.loc[common]).dropna().values
    if not higher:
        diffs = -diffs
    return _wilcoxon_onesided(diffs, "greater")


def _curve_xy(row, mode):
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


def _collect_curves(rows, mode):
    mats = []
    for _, r in rows.iterrows():
        c = _curve_xy(r, mode)
        if c is None:
            continue
        mats.append(np.interp(GRID, c[0], c[1]))
    return np.vstack(mats) if mats else np.zeros((0, len(GRID)))


def _plot_band(ax, mat, color, label, lw=2.2):
    if mat.shape[0] == 0:
        return
    mu = mat.mean(0); sd = mat.std(0)
    ax.fill_between(GRID, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
    ax.plot(GRID, mu, color=color, lw=lw, label=f"{label} (n={mat.shape[0]})")
    ax.plot(GRID, mu - sd, color=color, lw=0.7, ls=":", alpha=0.7)
    ax.plot(GRID, mu + sd, color=color, lw=0.7, ls=":", alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════════
# Generic bar grid for any metric (Claim 1 main + appendix variants)
# ══════════════════════════════════════════════════════════════════════════════
_METRIC_META = {
    "area": ("AUC (area under oversight curve)", True,  0.70, 1.05),
    "aag":  ("AAG (area above green baseline)",  True,  0.00, 0.22),
    "T_90": ("$T_{90}$ (fraction reviewed to reach 90%)", False, 0.0, 1.0),
    "T_95": ("$T_{95}$ (fraction reviewed to reach 95%)", False, 0.0, 1.2),
}


def _claim1_bars(metric: str, out_name: str, out_dir=None):
    """Bar grid: rows = models, cols = datasets (FOLIO, MALLS). Bars: Green, P2, P1."""
    mode   = "full"
    ylabel, higher, ylo, yhi = _METRIC_META[metric]
    col    = f"{mode}_{metric}"
    out_dir = out_dir or OUT

    metric_titles = {
        "area": "Metric: area under oversight curve (AUC)",
        "aag":  "Metric: area above green baseline (AAG = AUC − green area)",
        "T_90": "Metric: $T_{90}$ — fraction of data reviewed to reach 90% accuracy (lower is better)",
        "T_95": "Metric: $T_{95}$ — fraction of data reviewed to reach 95% accuracy (lower is better)",
    }

    fig, axes = plt.subplots(len(MODELS), len(DATASETS), figsize=(11, 10), sharey=False)
    fig.suptitle(
        "Claim 1: Pipeline 1 > Pipeline 2 > Green baseline\n"
        f"{metric_titles[metric]}, full dataset\n"
        "All brackets: one-sided paired Wilcoxon signed-rank "
        "(matched on prompting strategy × prompt variant, n=16). "
        "*** p<0.001, ** p<0.01, * p<0.05, ns",
        fontsize=11, fontweight="bold"
    )

    for row_i, model_label in enumerate(MODELS):
        p1tok = MODEL_KEY_P1[model_label]
        for col_i, ds_raw in enumerate(DATASETS):
            ax = axes[row_i][col_i]

            p1_ds = p1[(p1["_model_tok"] == p1tok) & (p1["_ds_raw"] == ds_raw)].copy()
            p2_ds = p2[(p2["_model_tok"] == p1tok) & (p2["_ds_raw"] == ds_raw)].copy()

            p1_vals = p1_ds[col].dropna().values
            p2_vals = p2_ds[col].dropna().values
            gv      = _green_scalar(p1tok, ds_raw, mode, metric)
            bv      = _black_scalar(ds_raw, mode, metric)

            if np.isnan(gv):
                ax.set_title(f"{model_label} — {DS_LABEL[ds_raw]}", fontsize=10)
                continue

            bars = [
                ("Green", "(direct)", gv, 0.0, GREEN_COLOR),
                ("P2", f"(n={len(p2_vals)})", np.mean(p2_vals), np.std(p2_vals), P2_COLOR),
                ("P1", f"(n={len(p1_vals)})", np.mean(p1_vals), np.std(p1_vals), P1_COLOR),
            ]

            labels = [b[0] for b in bars]
            ns_str = [b[1] for b in bars]
            means  = [b[2] for b in bars]
            stds   = [b[3] for b in bars]
            colors = [b[4] for b in bars]
            n_bars = len(bars)
            idx    = {lb: i for i, lb in enumerate(labels)}

            if higher:
                d_p1g = p1_vals - gv
                d_p2g = p2_vals - gv
            else:
                d_p1g = gv - p1_vals
                d_p2g = gv - p2_vals
            p_p1g = _wilcoxon_onesided(d_p1g, "greater")
            p_p2g = _wilcoxon_onesided(d_p2g, "greater")
            p_12  = _paired_wilcoxon_p1_p2(p1_ds, p2_ds, col, higher)

            bar_w  = 0.45
            x_gap  = 0.60
            xs     = [i * x_gap for i in range(n_bars)]
            for xi, (mv, sv, co) in zip(xs, zip(means, stds, colors)):
                ax.bar(xi, mv, width=bar_w, color=co, alpha=0.85,
                       yerr=sv if sv > 0 else None, capsize=4,
                       error_kw={"elinewidth": 1.5})
                label_y = mv + sv + (yhi - ylo) * 0.015
                ax.text(xi, label_y, f"{mv:.3f}", ha="center", va="bottom", fontsize=7.5)

            step  = (yhi - ylo) * 0.065
            y_top = max(mv + sv for mv, sv in zip(means, stds))
            base  = y_top + step * 0.5

            def _bracket(x1, x2, y, stars, ax=ax, step=step):
                ax.plot([x1, x1, x2, x2], [y, y + step * 0.12, y + step * 0.12, y],
                        lw=1.0, color="black")
                ax.text((x1 + x2) / 2, y + step * 0.14, stars,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

            bracket_list = [
                (xs[idx["Green"]], xs[idx["P2"]], _sig_stars(p_p2g)),
                (xs[idx["Green"]], xs[idx["P1"]], _sig_stars(p_p1g)),
                (xs[idx["P2"]], xs[idx["P1"]], _sig_stars(p_12)),
            ]

            for k, (x1, x2, stars) in enumerate(bracket_list):
                _bracket(x1, x2, base + step * k, stars)
            n_brackets = len(bracket_list)

            ax.set_xticks(xs)
            ax.set_xticklabels([f"{lb}\n{ns}" for lb, ns in zip(labels, ns_str)], fontsize=8)
            ax.set_xlim(xs[0] - bar_w, xs[-1] + bar_w)
            ax.set_ylim(ylo, yhi + step * (n_brackets + 2.5))
            ax.set_title(f"{model_label} — {DS_LABEL[ds_raw]}", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis="y", ls="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.axhline(gv, color=GREEN_COLOR, ls="--", lw=0.8, alpha=0.5)
            direction_str = "↑ better" if higher else "↓ better"
            ax.text(0.98, 0.02, direction_str, ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=7, color="gray", style="italic")

    caption = (
        "All pairs: one-sided paired Wilcoxon signed-rank, matched on prompting strategy × prompt variant "
        "within each (model, dataset) cell (n=16 pairs). "
        "No multiple-comparison correction within a cell."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8, style="italic", color="#444444")

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")


def fig_claim1_area_bars():
    _claim1_bars("area", "fig_claim1_area_bars.png")


def fig_appendix_aag_bars():
    _claim1_bars("aag", "fig_appendix_aag_bars.png")


def fig_appendix_t90_bars():
    _claim1_bars("T_90", "fig_appendix_t90_bars.png")


def fig_appendix_t95_bars():
    _claim1_bars("T_95", "fig_appendix_t95_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig Claim 3 — Gemma P1 band vs P2 best single curve
# ══════════════════════════════════════════════════════════════════════════════

def fig_claim3_gemma_bands_best_p2(out_dir=None):
    """P1 shown as a band restricted to B1-B3 × pv1,pv3; P2 shown as the single
    best-AUC curve for each dataset (FOLIO, MALLS)."""
    out_dir = out_dir or OUT
    ds_list   = ["folio_validation", "malls"]
    ds_labels = ["FOLIO", "MALLS"]
    gemma_p1  = p1[(p1["_model"] == "Gemma")].copy()
    gemma_p2  = p2[(p2["_model"] == "Gemma")].copy()

    best_filter = (
        gemma_p1["method"].isin(["B1", "B2", "B3"]) &
        gemma_p1["pv"].isin(["pv1", "pv3"])
    )
    gemma_p1_best = gemma_p1[best_filter].copy()

    gemma_p2_best = (
        gemma_p2.sort_values("full_area", ascending=False)
                .groupby("_ds_raw", sort=False)
                .head(1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(
        "Gemma — Pipeline 1 (band, B1–B3×pv1,pv3) vs Pipeline 2 (best single variant)\n"
        "Oversight curves  |  Full dataset\n"
        "★ = mean $T_{90}$   ■ = mean $T_{95}$",
        fontsize=12, fontweight="bold"
    )

    for col_i, (ds_raw, ds_lbl) in enumerate(zip(ds_list, ds_labels)):
        ax = axes[col_i]

        rows_p1 = gemma_p1_best[gemma_p1_best["_ds_raw"] == ds_raw]
        row_p2  = gemma_p2_best[gemma_p2_best["_ds_raw"] == ds_raw]

        mat_p1 = _collect_curves(rows_p1, "full")

        p2_label = ""
        p2_xy    = None
        if len(row_p2) > 0:
            r        = row_p2.iloc[0]
            p2_xy    = _curve_xy(r, "full")
            p2_auc   = r["full_area"]
            p2_base  = r["method"]
            p2_pv    = r["pv"]
            p2_label = f"P2 best — {p2_base}_{p2_pv} (AUC={p2_auc:.4f})"

        _plot_band(ax, mat_p1, P1_COLOR, "P1 — B1–B3×pv1,pv3", lw=2.5)

        if p2_xy is not None:
            ax.plot(p2_xy[0], p2_xy[1], color=P2_COLOR, lw=2.2,
                    ls="-", label=p2_label, zorder=5)

        p1tok = "gemma_4_31b"
        y0_green = GREEN_Y0.get((p1tok, ds_raw, "full"))
        y0_black = BLACK_Y0.get((ds_raw, "full"))

        if y0_green is not None:
            ax.plot([0, 1], [y0_green, 1], color=GREEN_COLOR, lw=2.0, ls="-",
                    label=f"Green baseline (y₀={y0_green:.3f})")
        if y0_black is not None:
            ax.plot([0, 1], [y0_black, 1], color=BLACK_COLOR, lw=2.0, ls="-",
                    label=f"Black baseline (y₀={y0_black:.3f})")

        for thresh, tlabel in [(0.90, "T=0.90"), (0.95, "T=0.95")]:
            ax.axhline(thresh, ls="--", lw=0.8, color="gray", alpha=0.55)
            ax.text(0.002, thresh + 0.003, tlabel,
                    transform=ax.get_yaxis_transform(), fontsize=7.5, color="gray")

        label_entries = []

        if mat_p1.shape[0] > 0:
            mu_p1 = mat_p1.mean(0)
            for thresh, mk_shape, mk_size in [(0.90, "*", 14), (0.95, "s", 8)]:
                idx = np.searchsorted(mu_p1, thresh)
                if idx >= len(GRID):
                    continue
                t_x = float(np.interp(thresh, mu_p1[max(0, idx - 1):idx + 2],
                                       GRID[max(0, idx - 1):idx + 2]))
                ax.plot([t_x, t_x], [0.50, thresh],
                        color=P1_COLOR, ls=":", lw=1.2, alpha=0.6, zorder=5)
                ax.plot(t_x, thresh, marker=mk_shape, color=P1_COLOR,
                        ms=mk_size, zorder=6, clip_on=False)
                label_entries.append((t_x, thresh, P1_COLOR))

        if p2_xy is not None:
            xs_p2, ys_p2 = p2_xy
            for thresh, mk_shape, mk_size in [(0.90, "*", 14), (0.95, "s", 8)]:
                idx = np.searchsorted(ys_p2, thresh)
                if idx >= len(ys_p2):
                    continue
                t_x = float(np.interp(thresh, ys_p2[max(0, idx - 1):idx + 2],
                                       xs_p2[max(0, idx - 1):idx + 2]))
                ax.plot([t_x, t_x], [0.50, thresh],
                        color=P2_COLOR, ls=":", lw=1.2, alpha=0.6, zorder=4)
                ax.plot(t_x, thresh, marker=mk_shape, color=P2_COLOR,
                        ms=mk_size, zorder=5, clip_on=False)
                label_entries.append((t_x, thresh, P2_COLOR))

        for y0, color, zord in [(y0_green, GREEN_COLOR, 4), (y0_black, BLACK_COLOR, 3)]:
            if y0 is None:
                continue
            for thresh, mk_shape, mk_size in [(0.90, "*", 14), (0.95, "s", 8)]:
                t_base = (thresh - y0) / (1.0 - y0) if y0 < thresh else 0.0
                ax.plot([t_base, t_base], [0.50, thresh],
                        color=color, ls=":", lw=1.2, alpha=0.6, zorder=zord)
                ax.plot(t_base, thresh, marker=mk_shape, color=color,
                        ms=mk_size, zorder=zord + 1, clip_on=False)
                label_entries.append((t_base, thresh, color))

        VERT_OFFSET = 0.013
        for thresh_grp in [0.90, 0.95]:
            grp = sorted([(x, c) for x, t, c in label_entries if t == thresh_grp],
                         key=lambda e: e[0])
            for k, (t_x, color) in enumerate(grp):
                v_off = VERT_OFFSET if k % 2 == 0 else -VERT_OFFSET - 0.010
                ax.text(t_x, thresh_grp + v_off, f"{t_x:.2f}",
                        ha="center", va="bottom" if v_off > 0 else "top",
                        fontsize=7.5, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0))

        ax.set_xlim(0, 1)
        ax.set_ylim(0.50, 1.01)
        ax.set_title(ds_lbl, fontsize=12)
        ax.set_xlabel("Human oversight (fraction reviewed)", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", fontsize=9)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="gray", ms=11, ls="none", label="★  mean $T_{90}$ (P1) / $T_{90}$ (P2/baselines)"),
        Line2D([0], [0], marker="s", color="gray", ms=7, ls="none", label="■  mean $T_{95}$ (P1) / $T_{95}$ (P2/baselines)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.05, 1, 0.91])
    out = out_dir / "fig_claim3_gemma_bands_best_p2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Generating paper plots (FOLIO + MALLS only) ===")
    fig_claim1_area_bars()
    fig_claim3_gemma_bands_best_p2()
    print("\n=== Generating appendix plots ===")
    fig_appendix_aag_bars()
    fig_appendix_t90_bars()
    fig_appendix_t95_bars()
    print(f"\nSaved in: {OUT}/")
