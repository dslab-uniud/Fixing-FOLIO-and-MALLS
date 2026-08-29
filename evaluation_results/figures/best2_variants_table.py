"""best2_variants_table.py — LaTeX table: Black/Green + top-2 P1/P2 variants (by AUC)
for each model x dataset (FOLIO, MALLS).

Reuses GREEN_Y0/BLACK_Y0 and the oversight CSVs already computed elsewhere in this
project (evaluation_results/pipeline{1,2}/human_oversight/all_datasets_oversight.csv).

Scoped to FOLIO and MALLS only — GGC/Stanford is a private dataset and is
excluded from this public release.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation_results" / "figures"))

# Reuse the already-verified GREEN_Y0 / BLACK_Y0 constants
import generate_paper_plots as gpp

OUT_DIR = Path(__file__).parent / "best2_variants_results"
OUT_DIR.mkdir(exist_ok=True)

MODELS = ["gemma_4_31b", "gpt4o_mini", "qwen_local_30b"]
MODEL_LABEL = {"gemma_4_31b": "Gemma", "gpt4o_mini": "GPT-4o-mini", "qwen_local_30b": "Qwen"}
DATASETS = ["folio_validation", "malls"]
DS_LABEL = {"folio_validation": "FOLIO", "malls": "MALLS"}
P2_TO_P1_TOK = {"gemma": "gemma_4_31b", "gpt4omini": "gpt4o_mini", "qwen3": "qwen_local_30b"}
P1_TO_P2_TOK = {v: k for k, v in P2_TO_P1_TOK.items()}
P2_DS_TO_P1 = {"folio": "folio_validation", "malls": "malls"}
P1_DS_TO_P2 = {v: k for k, v in P2_DS_TO_P1.items()}


def _scalar_from_y0(y0):
    """AUC/T90/T95 for a straight line from (0,y0) to (1,1)."""
    if y0 is None or np.isnan(y0):
        return None, None, None
    auc = (y0 + 1.0) / 2.0
    t90 = (0.90 - y0) / (1.0 - y0) if y0 < 0.90 else 0.0
    t95 = (0.95 - y0) / (1.0 - y0) if y0 < 0.95 else 0.0
    return auc, t90, t95


def build_table():
    p1 = pd.read_csv(ROOT / "evaluation_results/pipeline1/human_oversight/all_datasets_oversight.csv")
    p1 = p1[p1["method"].isin(["B1", "B2", "B3", "B4"])].copy()
    p1 = p1[~p1["variant"].str.contains("fs3|fs5", na=False)]

    p2 = pd.read_csv(ROOT / "evaluation_results/pipeline2/human_oversight/all_datasets_oversight.csv")
    p2 = p2[(p2["phi_model"] == p2["judge_model"]) & (p2["phi_prompt"] == "P1")
            & p2["baseline"].isin(["B1", "B2", "B3", "B4"])].copy()
    # Same convention as generate_paper_plots.py: B1/B4 use plain 'B', B2/B3 use 5-shot 'fs5_B'
    b14 = p2["baseline"].isin(["B1", "B4"]) & (p2["task_tag"] == "B")
    b23 = p2["baseline"].isin(["B2", "B3"]) & (p2["task_tag"] == "fs5_B")
    p2 = p2[b14 | b23].copy()

    rows = []
    for model in MODELS:
        for ds in DATASETS:
            green_y0 = gpp.GREEN_Y0.get((model, ds, "full"))
            black_y0 = gpp.BLACK_Y0.get((ds, "full"))

            b_auc, b_t90, b_t95 = _scalar_from_y0(black_y0)
            g_auc, g_t90, g_t95 = _scalar_from_y0(green_y0)

            rows.append({"model": model, "dataset": ds, "pipeline": "Black",
                         "baseline": "-", "pv": "-", "auc": b_auc, "t90": b_t90, "t95": b_t95})
            rows.append({"model": model, "dataset": ds, "pipeline": "Green",
                         "baseline": "-", "pv": "-", "auc": g_auc, "t90": g_t90, "t95": g_t95})

            # P1: top 2 of 16 (B1-B4 x pv1-4) by full_area
            sub1 = p1[(p1["model"] == model) & (p1["dataset"] == ds)].copy()
            sub1 = sub1.sort_values("full_area", ascending=False).head(2)
            for _, r in sub1.iterrows():
                rows.append({"model": model, "dataset": ds, "pipeline": "P1",
                             "baseline": r["method"], "pv": r["pv"].replace("pv", ""),
                             "auc": r["full_area"], "t90": r["full_T_90"], "t95": r["full_T_95"]})

            # P2: top 2 of 16 by full_area
            p2_model = P1_TO_P2_TOK[model]
            p2_ds = P1_DS_TO_P2[ds]
            sub2 = p2[(p2["judge_model"] == p2_model) & (p2["source_dataset"] == p2_ds)].copy()
            sub2 = sub2.sort_values("full_area", ascending=False).head(2)
            for _, r in sub2.iterrows():
                rows.append({"model": model, "dataset": ds, "pipeline": "P2",
                             "baseline": r["baseline"], "pv": r["pv"].replace("pv", ""),
                             "auc": r["full_area"], "t90": r["full_T_90"], "t95": r["full_T_95"]})
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame) -> str:
    lines = []
    n_rows_per_ds = 6   # Black, Green, 2xP1, 2xP2
    for model in MODELS:
        lines.append(f"\\multirow{{{n_rows_per_ds * len(DATASETS)}}}{{*}}{{\\textbf{{{MODEL_LABEL[model]}}}}}")
        for i, ds in enumerate(DATASETS):
            sub = df[(df["model"] == model) & (df["dataset"] == ds)]
            lines.append(f"  & \\multirow{{{n_rows_per_ds}}}{{*}}{{{DS_LABEL[ds]}}}")
            for j, (_, r) in enumerate(sub.iterrows()):
                pipe = r["pipeline"]
                pipe_tex = {"Black": "Black", "Green": "Green",
                            "P1": "P\\textsubscript{1}", "P2": "P\\textsubscript{2}"}[pipe]
                b = r["baseline"]
                pv = r["pv"] if r["pv"] == "-" else f"pv{r['pv']}"
                auc = f"{r['auc']:.3f}" if r["auc"] is not None else "-"
                t90 = f"{r['t90']:.3f}" if r["t90"] is not None else "-"
                t95 = f"{r['t95']:.3f}" if r["t95"] is not None else "-"
                row_tex = f"    & {pipe_tex} & {b} & {pv} & {auc} & {t90} & {t95} \\\\"
                if pipe == "P2":
                    lines.append("  & %\\rowcolor{rowgray}")
                if pipe in ("Black", "Green"):
                    lines.append(row_tex + " &")
                elif j == 2:  # first P1 row after Black/Green: no leading '&'
                    lines.append(row_tex)
                else:
                    lines.append("  &" + row_tex)
            if i < len(DATASETS) - 1:
                lines.append("\\cmidrule(l){2-8}")
        if model != MODELS[-1]:
            lines.append("\\midrule")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    df = build_table()
    csv_path = OUT_DIR / "best2_variants.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    tex = to_latex(df)
    tex_path = OUT_DIR / "best2_variants_table.tex"
    tex_path.write_text(tex)
    print(f"Saved {tex_path}")
    print()
    print(tex)
