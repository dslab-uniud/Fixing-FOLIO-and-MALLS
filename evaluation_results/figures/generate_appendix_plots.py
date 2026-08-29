"""generate_appendix_plots.py — Appendix figures.

Appendix B: Per-model, per-dataset comparison
  → fig_appB_model_comparison.png

Appendix C: Variant selection — heatmaps (Gemma P1/P2, baseline × prompt version)
  → fig_appC_p1_heatmap.png, fig_appC_p2_heatmap.png

Appendix E: Full oversight band grid (all models × datasets, P1/P2/green)
  → fig_appE_band_grid.png

Appendix F: psi_hat accuracy by quality_score group (prioritization justification)
  → fig_appF_verdict_accuracy.png

Scoped to FOLIO and MALLS only — GGC/Stanford is a private dataset and is
excluded from this public release (data, intermediate NL sentences, and LLM
responses are not redistributed).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

_BASE = Path(__file__).parent.parent   # evaluation_results/
OUT   = Path(__file__).parent          # evaluation_results/figures/

# ── canonical constants (shared with generate_paper_plots.py) ────────────────
GREEN_Y0 = {
    ("gemma_4_31b",    "folio_validation", "full"):  0.7418,
    ("gemma_4_31b",    "malls",            "full"):  0.8600,
    ("gpt4o_mini",     "folio_validation", "full"):  0.6145,
    ("gpt4o_mini",     "malls",            "full"):  0.5700,
    ("qwen_local_30b", "folio_validation", "full"):  0.5855,
    ("qwen_local_30b", "malls",            "full"):  0.6800,
}

MODEL_MAP_P1 = {"gpt4o_mini": "GPT-4o-mini", "qwen_local_30b": "Qwen", "gemma_4_31b": "Gemma"}
MODEL_MAP_P2 = {"gemma": "Gemma", "gpt4omini": "GPT-4o-mini", "qwen3": "Qwen"}
P2_TO_P1_TOK = {"gemma": "gemma_4_31b", "gpt4omini": "gpt4o_mini", "qwen3": "qwen_local_30b"}
P2_DS_TO_P1  = {"folio": "folio_validation", "malls": "malls"}
MODEL_KEY_P1 = {"Gemma": "gemma_4_31b", "GPT-4o-mini": "gpt4o_mini", "Qwen": "qwen_local_30b"}
MODEL_COLORS = {"Gemma": "#9467bd", "GPT-4o-mini": "#17becf", "Qwen": "#8c564b"}

MODELS    = ["Gemma", "GPT-4o-mini", "Qwen"]
DATASETS  = ["folio_validation", "malls"]
DS_LABEL  = {"folio_validation": "FOLIO", "malls": "MALLS"}
BASELINES = ["B1", "B2", "B3", "B4"]
PVS       = ["pv1", "pv2", "pv3", "pv4"]

P1_COLOR    = "#1f77b4"
P2_COLOR    = "#d62728"
GREEN_COLOR = "#2ca02c"

# ── load data ─────────────────────────────────────────────────────────────────
p1 = pd.read_csv(_BASE / "pipeline1/human_oversight/all_datasets_oversight.csv")
p1 = p1[p1["method"].isin(["B1", "B2", "B3", "B4"])].copy()
p1 = p1[~p1["variant"].str.contains("fs3|fs5", na=False)]
p1["_model"]     = p1["model"].map(MODEL_MAP_P1)
p1["_ds_raw"]    = p1["dataset"]
p1["_model_tok"] = p1["model"]

p2 = pd.read_csv(_BASE / "pipeline2/human_oversight/all_datasets_oversight.csv")
p2 = p2[(p2["phi_model"] == p2["judge_model"]) & (p2["phi_prompt"] == "P1")
        & p2["baseline"].isin(["B1", "B2", "B3", "B4"])].copy()
b14 = p2["baseline"].isin(["B1", "B4"]) & (p2["task_tag"] == "B")
b23 = p2["baseline"].isin(["B2", "B3"]) & (p2["task_tag"] == "fs5_B")
p2  = p2[b14 | b23].copy()
p2["_model"]     = p2["judge_model"].map(MODEL_MAP_P2)
p2["_model_tok"] = p2["judge_model"].map(P2_TO_P1_TOK)
p2["_ds_raw"]    = p2["source_dataset"].map(P2_DS_TO_P1)
p2 = p2.rename(columns={"baseline": "method"})

for df in [p1, p2]:
    df["full_aag"] = df.apply(
        lambda r: r["full_area"] - (GREEN_Y0.get((r["_model_tok"], r["_ds_raw"], "full"), np.nan) + 1.0) / 2.0,
        axis=1)

# ── helpers ───────────────────────────────────────────────────────────────────
def _mwu(a, b, alt="greater"):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, 1.0
    try:
        u, p = stats.mannwhitneyu(a, b, alternative=alt)
        return float(u), float(p)
    except Exception:
        return np.nan, 1.0


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════════
# App B — Per-model, per-dataset comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_appB_model_comparison():
    col_map = {"AUC": "full_area", "AAG": "full_aag", "T_90": "full_T_90", "T_95": "full_T_95"}
    metric_cols = [
        ("AUC", "Area (AUC)",            True,  0.70, 1.01),
        ("AAG", "Area above green (AAG)", True,  0.00, 0.22),
        ("T_90", "$T_{90}$",             False, 0.00, 1.00),
        ("T_95", "$T_{95}$",             False, 0.00, 1.20),
    ]
    ds_list   = ["folio_validation", "malls"]
    ds_labels = ["FOLIO", "MALLS"]

    fig, axes = plt.subplots(len(ds_list), 4, figsize=(18, 10))
    fig.suptitle(
        "Gemma outperforms GPT-4o-mini and Qwen across datasets and metrics\n"
        "Pipeline 1, full dataset, per dataset  (mean ± std, all B1–B4 × pv1–pv4)\n"
        "Pairwise MWU with Bonferroni correction (3 pairs)",
        fontsize=11, fontweight="bold"
    )

    for row_i, (ds_raw, ds_lbl) in enumerate(zip(ds_list, ds_labels)):
        p1_ds = p1[p1["_ds_raw"] == ds_raw].copy()

        for col_i, (met, m_label, higher, ylo, yhi) in enumerate(metric_cols):
            ax  = axes[row_i][col_i]
            col = col_map[met]
            model_vals = {m: p1_ds[p1_ds["_model"] == m][col].dropna().values for m in MODELS}
            means  = [model_vals[m].mean() for m in MODELS]
            stds   = [model_vals[m].std()  for m in MODELS]
            ns     = [len(model_vals[m])   for m in MODELS]
            colors = [MODEL_COLORS[m] for m in MODELS]

            for xi, (mv, sv, co) in enumerate(zip(means, stds, colors)):
                ax.bar(xi, mv, width=0.55, color=co, alpha=0.85,
                       yerr=sv, capsize=5, error_kw={"elinewidth": 1.5})
                ax.text(xi, mv + sv + (yhi - ylo) * 0.02, f"{mv:.3f}",
                        ha="center", va="bottom", fontsize=8)

            pairs = [("Gemma", "GPT-4o-mini"), ("Gemma", "Qwen"), ("GPT-4o-mini", "Qwen")]
            p_vals = {}
            for a, b in pairs:
                alt = "greater" if higher else "less"
                _, p_raw = _mwu(model_vals[a], model_vals[b], alt)
                p_vals[(a, b)] = min(p_raw * 3, 1.0)

            y_top  = max(m + s for m, s in zip(means, stds))
            step   = (yhi - ylo) * 0.08
            base   = y_top + step * 0.3
            xi_map = {m: i for i, m in enumerate(MODELS)}

            def _bracket(i, j, y, stars, ax=ax, step=step):
                ax.plot([i, i, j, j], [y, y + step * 0.12, y + step * 0.12, y], lw=1.0, color="black")
                ax.text((i + j) / 2, y + step * 0.14, stars, ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

            _bracket(xi_map["Gemma"], xi_map["GPT-4o-mini"], base,          _sig_stars(p_vals[("Gemma", "GPT-4o-mini")]))
            _bracket(xi_map["Gemma"], xi_map["Qwen"],         base + step,   _sig_stars(p_vals[("Gemma", "Qwen")]))
            _bracket(xi_map["GPT-4o-mini"], xi_map["Qwen"],   base + step*2, _sig_stars(p_vals[("GPT-4o-mini", "Qwen")]))

            ax.set_xticks(range(3))
            ax.set_xticklabels([f"{m}\n(n={n})" for m, n in zip(MODELS, ns)], fontsize=8.5)
            ax.set_title(f"{m_label} — {ds_lbl}", fontsize=10)
            ax.set_ylabel(m_label, fontsize=9)
            ax.set_ylim(ylo, yhi + step * 5)
            ax.grid(axis="y", ls="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.98, 0.02, "↑ better" if higher else "↓ better",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=7, color="gray", style="italic")
            if met == "AAG":
                ax.axhline(0.0, color=GREEN_COLOR, ls="--", lw=0.9, alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT / "fig_appB_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# App C — Heatmaps (Gemma variant selection)
# ══════════════════════════════════════════════════════════════════════════════
def _draw_heatmap(ax, pivot, cmap, title, fmt=".4f", lo=None, hi=None):
    vals = pivot.values.astype(float)
    vmin = lo if lo is not None else np.nanmin(vals)
    vmax = hi if hi is not None else np.nanmax(vals)
    im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(4)); ax.set_xticklabels(PVS, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(BASELINES, fontsize=9)
    ax.set_xlabel("Prompt version", fontsize=8)
    ax.set_ylabel("Baseline", fontsize=8)
    ax.set_title(title, fontsize=10, pad=4)
    mid = (vmin + vmax) / 2
    for i in range(4):
        for j in range(4):
            v = vals[i, j]
            if not np.isnan(v):
                tc = "white" if abs(v - mid) > 0.35 * (vmax - vmin) else "black"
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=8.5, color=tc)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def fig_appC_heatmaps():
    """4×4 AUC heatmaps for Gemma P1 and P2, averaged over FOLIO+MALLS."""
    gemma_p1 = p1[(p1["_model"] == "Gemma") & (p1["_ds_raw"].isin(["folio_validation", "malls"]))]
    gemma_p2 = p2[(p2["_model"] == "Gemma") & (p2["_ds_raw"].isin(["folio_validation", "malls"]))]

    metrics = [
        ("full_area", "AUC",  "YlGn",     True,  ".4f"),
        ("full_T_90", "$T_{90}$", "YlOrRd_r", False, ".3f"),
        ("full_T_95", "$T_{95}$", "YlOrRd_r", False, ".3f"),
    ]

    for pipe_label, df in [("P1", gemma_p1), ("P2", gemma_p2)]:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(
            f"Gemma {pipe_label} — performance heatmap (baseline × prompt version)\n"
            "Mean over FOLIO + MALLS, full dataset",
            fontsize=12, fontweight="bold"
        )
        for ax, (col, label, cmap, higher, fmt) in zip(axes, metrics):
            pivot = df.pivot_table(values=col, index="method", columns="pv", aggfunc="mean")
            pivot = pivot.reindex(index=BASELINES, columns=PVS)
            _draw_heatmap(ax, pivot, cmap, f"{label}  ({'↑' if higher else '↓'} better)", fmt=fmt)

        fig.tight_layout(rect=[0, 0, 1, 0.88])
        out = OUT / f"fig_appC_{pipe_label.lower()}_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# App E — Full oversight band grid: all 3 models × 2 datasets, P1 / P2 / green
# ══════════════════════════════════════════════════════════════════════════════

GRID = np.linspace(0.0, 1.0, 201)


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


def fig_appE_band_grid():
    """3 rows (models) × 2 cols (datasets), P1 (blue) / P2 (red) / green line.

    P1: all 16 variants (B1–B4 × pv1–pv4).
    P2: self-judge, phi_prompt=P1, same 16 variants.
    Both use full-dataset mode.
    """
    P2_MODEL_MAP = {"gemma_4_31b": "gemma", "gpt4o_mini": "gpt4omini", "qwen_local_30b": "qwen3"}

    fig, axes = plt.subplots(3, len(DATASETS), figsize=(12, 15), sharey=True)
    fig.suptitle(
        "Human-oversight curves — all models and datasets\n"
        "Pipeline 1 (blue) vs Pipeline 2 self-judge (red) vs direct LLM translation (green)\n"
        "Mean ± 1 std across all 16 variants (B1–B4 × pv1–pv4)  |  Full dataset",
        fontsize=13, fontweight="bold"
    )

    for row_i, model_label in enumerate(MODELS):
        p1tok = MODEL_KEY_P1[model_label]
        p2tok = P2_MODEL_MAP[p1tok]

        for col_i, ds_raw in enumerate(DATASETS):
            ax = axes[row_i][col_i]

            rows_p1 = p1[(p1["_model_tok"] == p1tok) & (p1["_ds_raw"] == ds_raw)]
            rows_p2 = p2[(p2["judge_model"] == p2tok) & (p2["_ds_raw"] == ds_raw)]

            mat_p1 = _collect_curves(rows_p1, "full")
            mat_p2 = _collect_curves(rows_p2, "full")

            _plot_band(ax, mat_p1, P1_COLOR, "P1", lw=2.3)
            _plot_band(ax, mat_p2, P2_COLOR, "P2", lw=2.0)

            y0 = GREEN_Y0.get((p1tok, ds_raw, "full"))
            if y0 is not None:
                ax.plot([0, 1], [y0, 1], color=GREEN_COLOR, lw=2.0, ls="-",
                        label=f"Green baseline (y₀={y0:.3f})")

            for thresh in [0.90, 0.95]:
                ax.axhline(thresh, ls="--", lw=0.7, color="gray", alpha=0.5)
                ax.text(0.002, thresh + 0.003, f"{thresh:.0%}",
                        transform=ax.get_yaxis_transform(), fontsize=7, color="gray")

            ax.set_xlim(0, 1); ax.set_ylim(0.45, 1.01)
            ax.set_title(f"{model_label} — {DS_LABEL[ds_raw]}", fontsize=11)
            ax.set_xlabel("Human oversight (fraction reviewed)", fontsize=9)
            if col_i == 0:
                ax.set_ylabel("Accuracy", fontsize=9)
            ax.grid(alpha=0.22)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color=P1_COLOR,    lw=2.3, label="Pipeline 1 — mean (band = ±1 std, 16 variants)"),
        Patch(facecolor=P1_COLOR, alpha=0.18, label="Pipeline 1 — ±1 std band"),
        Line2D([0], [0], color=P2_COLOR,    lw=2.0, label="Pipeline 2 self-judge — mean (band = ±1 std, 16 variants)"),
        Patch(facecolor=P2_COLOR, alpha=0.18, label="Pipeline 2 — ±1 std band"),
        Line2D([0], [0], color=GREEN_COLOR, lw=2.0, label="Green baseline — direct LLM translation (y₀ varies by model/dataset)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    out = OUT / "fig_appE_band_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# App F — psi_hat accuracy by quality_score group (prioritization justification)
# ══════════════════════════════════════════════════════════════════════════════

def _backcompute_verdict_accuracy():
    """Back-compute P(psi_hat ≡ gold | verdict=qs) from oversight CSV coordinates.

    For each (model, dataset, pipeline) cell we aggregate over all eligible
    variants (B1-B4 × pv1-pv4 for P1; self-judge B1-B4 for P2) by summing
    counts, then compute:

        gc_yes = round(y3*N) - n_nr - n_qm - n_no
        gc_no  = round((y2-y3)*N) + n_no
        gc_qm  = round((y1-y2)*N) + n_qm

        acc_yes = gc_yes / n_pred_yes
        acc_no  = gc_no  / n_pred_no
        acc_qm  = gc_qm  / n_pred_qm

    where n_nr = n_no_resp (unparseable), n_qm = n_pred_qm, n_no = n_pred_no.
    """
    DS_LABEL_MAP = {"folio_validation": "FOLIO", "malls": "MALLS"}

    rows = []

    # ── Pipeline 1 ────────────────────────────────────────────────────────────
    p1_ov = pd.read_csv(_BASE / "pipeline1/human_oversight/all_datasets_oversight.csv")
    p1_ov = p1_ov[p1_ov["method"].isin(["B1", "B2", "B3", "B4"])].copy()
    p1_ov = p1_ov[~p1_ov["variant"].str.contains("fs3|fs5", na=False)]
    p1_ov["_model"]  = p1_ov["model"].map(MODEL_MAP_P1)
    p1_ov["_ds_lbl"] = p1_ov["dataset"].map(DS_LABEL_MAP)
    p1_ov = p1_ov.dropna(subset=["_model", "_ds_lbl"])

    for (model_label, ds_label), grp in p1_ov.groupby(["_model", "_ds_lbl"]):
        N    = grp["full_N"].sum()
        n_nr = grp["full_n_no_resp"].sum()
        n_qm = grp["full_n_pred_qm"].sum()
        n_no = grp["full_n_pred_no"].sum()
        n_yes = grp["full_n_pred_yes"].sum()
        y0 = (grp["full_y0"] * grp["full_N"]).sum() / N
        y1 = (grp["full_y1"] * grp["full_N"]).sum() / N
        y2 = (grp["full_y2"] * grp["full_N"]).sum() / N
        y3 = (grp["full_y3"] * grp["full_N"]).sum() / N

        gc_yes = round(y3 * N) - n_nr - n_qm - n_no
        gc_no  = round((y2 - y3) * N) + n_no
        gc_qm  = round((y1 - y2) * N) + n_qm
        gc_nr  = round(y0 * N) - gc_qm - gc_no - gc_yes

        rows.append({
            "model": model_label, "dataset": ds_label, "pipeline": "P1",
            "acc_yes": gc_yes / n_yes if n_yes > 0 else np.nan,
            "n_yes":   n_yes,
            "acc_no":  gc_no / n_no if n_no > 0 else np.nan,
            "n_no":    n_no,
            "acc_qm":  gc_qm / n_qm if n_qm > 0 else np.nan,
            "n_qm":    n_qm,
            "gc_nr":   gc_nr,
            "n_nr":    n_nr,
        })

    # ── Pipeline 2 ────────────────────────────────────────────────────────────
    p2_ov = pd.read_csv(_BASE / "pipeline2/human_oversight/all_datasets_oversight.csv")
    p2_ov = p2_ov[(p2_ov["phi_model"] == p2_ov["judge_model"]) &
                  (p2_ov["phi_prompt"] == "P1") &
                  p2_ov["baseline"].isin(["B1", "B2", "B3", "B4"])].copy()
    b14 = p2_ov["baseline"].isin(["B1", "B4"]) & (p2_ov["task_tag"] == "B")
    b23 = p2_ov["baseline"].isin(["B2", "B3"]) & (p2_ov["task_tag"] == "fs5_B")
    p2_ov = p2_ov[b14 | b23].copy()
    p2_ov["_model"]  = p2_ov["judge_model"].map(MODEL_MAP_P2)
    p2_ov["_ds_lbl"] = p2_ov["source_dataset"].map(P2_DS_TO_P1).map(DS_LABEL_MAP)
    p2_ov = p2_ov.dropna(subset=["_model", "_ds_lbl"])

    for (model_label, ds_label), grp in p2_ov.groupby(["_model", "_ds_lbl"]):
        N    = grp["full_N"].sum()
        n_nr = grp["full_n_no_resp"].sum()
        n_qm = grp["full_n_pred_qm"].sum()
        n_no = grp["full_n_pred_no"].sum()
        n_yes = grp["full_n_pred_yes"].sum()
        y0 = (grp["full_y0"] * grp["full_N"]).sum() / N
        y1 = (grp["full_y1"] * grp["full_N"]).sum() / N
        y2 = (grp["full_y2"] * grp["full_N"]).sum() / N
        y3 = (grp["full_y3"] * grp["full_N"]).sum() / N

        gc_yes = round(y3 * N) - n_nr - n_qm - n_no
        gc_no  = round((y2 - y3) * N) + n_no
        gc_qm  = round((y1 - y2) * N) + n_qm
        gc_nr  = round(y0 * N) - gc_qm - gc_no - gc_yes

        rows.append({
            "model": model_label, "dataset": ds_label, "pipeline": "P2",
            "acc_yes": gc_yes / n_yes if n_yes > 0 else np.nan,
            "n_yes":   n_yes,
            "acc_no":  gc_no / n_no if n_no > 0 else np.nan,
            "n_no":    n_no,
            "acc_qm":  gc_qm / n_qm if n_qm > 0 else np.nan,
            "n_qm":    n_qm,
            "gc_nr":   gc_nr,
            "n_nr":    n_nr,
        })

    return pd.DataFrame(rows)


def _wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def fig_appF_verdict_accuracy():
    """3×2 grid (model × dataset): P(psi_hat ≡ gold | verdict=qs) for P1 and P2."""
    df = _backcompute_verdict_accuracy()

    QS_ORDER  = ["?", "no", "yes"]
    QS_LABELS = {"?": "?", "no": "no", "yes": "yes"}
    QS_ACC    = {"yes": "acc_yes", "no": "acc_no", "?": "acc_qm"}
    QS_N      = {"yes": "n_yes",   "no": "n_no",   "?": "n_qm"}
    X_POS     = {"?": 0, "no": 1, "yes": 2}
    WIDTH     = 0.32
    PIPE_OFFSET = {"P1": -WIDTH / 2, "P2": +WIDTH / 2}
    PIPE_COLOR  = {"P1": P1_COLOR, "P2": P2_COLOR}

    ds_labels = ["FOLIO", "MALLS"]

    fig, axes = plt.subplots(3, len(ds_labels), figsize=(12, 14), sharey=False)
    fig.suptitle(
        r"$\hat\psi$ accuracy conditioned on judge verdict — P1 (blue) and P2 (red)"
        "\n"
        r"$P(\hat\psi \equiv \phi_{\mathrm{gold}} \mid \mathrm{verdict}=v)$"
        " — pooled over B1–B4 × pv1–pv4.  Error bars: 95% Wilson CI.",
        fontsize=11, fontweight="bold"
    )

    total_nr = int(df["n_nr"].sum())
    total_gc_nr = int(df["gc_nr"].sum())
    unp_acc = total_gc_nr / total_nr if total_nr > 0 else float("nan")
    unp_note = (f"Unparseable outputs (pooled n={total_nr}) excluded from bars; "
                f"pooled psi_hat accuracy = {unp_acc:.2f}.")

    for row_i, model_label in enumerate(MODELS):
        for col_i, ds_label in enumerate(ds_labels):
            ax = axes[row_i][col_i]

            cell = df[(df["model"] == model_label) & (df["dataset"] == ds_label)]

            for pipe in ["P1", "P2"]:
                row = cell[cell["pipeline"] == pipe]
                if row.empty:
                    continue
                row = row.iloc[0]

                for qs in QS_ORDER:
                    acc = row[QS_ACC[qs]]
                    n   = int(row[QS_N[qs]])
                    if n == 0 or np.isnan(acc):
                        continue
                    k = round(acc * n)
                    lo, hi = _wilson_ci(k, n)
                    x = X_POS[qs] + PIPE_OFFSET[pipe]
                    ax.bar(x, acc, width=WIDTH * 0.92,
                           color=PIPE_COLOR[pipe], alpha=0.82,
                           label=pipe if qs == "?" else "_nolegend_")
                    ax.errorbar(x, acc, yerr=[[max(acc - lo, 0)], [max(hi - acc, 0)]],
                                fmt="none", color="black", capsize=3, lw=1.2)
                    ax.text(x, hi + 0.015, f"{acc:.2f}\n(n={n})",
                            ha="center", va="bottom", fontsize=6.2, color="#333")

            ax.set_xticks(list(X_POS.values()))
            ax.set_xticklabels([QS_LABELS[q] for q in QS_ORDER], fontsize=9)
            ax.set_ylim(0, 1.22)
            ax.set_ylabel(r"$P(\hat\psi \equiv \phi_{\mathrm{gold}} \mid v)$", fontsize=8)
            ax.set_title(f"{model_label} — {ds_label}", fontsize=10)
            ax.grid(axis="y", ls="--", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(color=P1_COLOR, alpha=0.82, label="Pipeline 1"),
               Patch(color=P2_COLOR, alpha=0.82, label="Pipeline 2")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))

    fig.text(0.5, 0.005, unp_note, ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])

    out = OUT / "fig_appF_verdict_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Generating appendix plots (FOLIO + MALLS only) ===")
    fig_appB_model_comparison()
    fig_appC_heatmaps()
    fig_appE_band_grid()
    fig_appF_verdict_accuracy()
    print(f"\nSaved in: {OUT}/")
