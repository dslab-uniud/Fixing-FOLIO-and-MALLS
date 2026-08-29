# evaluation_results — Aggregated Outputs and Figures

This directory contains the aggregated tables and figures used in the paper, derived from the per-instance prediction files in `pipeline1/` and `pipeline2/`.

## Layout

```
evaluation_results/
├── pipeline1/human_oversight/       Accuracy when a human reviews only flagged instances
│   ├── compute_human_oversight.py   Recompute all_datasets_oversight.csv
│   └── all_datasets_oversight.csv
├── pipeline2/human_oversight/       Same regime as pipeline1, on generated phi_hat
│   ├── compute_human_oversight.py
│   └── all_datasets_oversight.csv
├── pipeline{1,2}/automatic_regime/
│   └── compute_accuracy.py          Import dependency of compute_human_oversight.py
└── figures/                         Scripts that regenerate every figure/table in the paper
    ├── generate_paper_plots.py      Main-text figures (claim1, claim3, appendix AAG/T90/T95)
    ├── generate_appendix_plots.py   Appendix B/C/E/F figures
    ├── length_ordering_analysis.py  Builds the records cache used by length_ordering_plots.py
    ├── length_ordering_plots.py     Appendix length-ordering figure
    └── best2_variants_table.py      Best-2-variants LaTeX table
```

Figures/tables are computed on **FOLIO and MALLS only** — GGC/Stanford is a private dataset and is excluded from this public release.

## Reproducing

```bash
# 1. Recompute the human-oversight CSVs (depends on pipeline1/2 prediction files)
python evaluation_results/pipeline1/human_oversight/compute_human_oversight.py
python evaluation_results/pipeline2/human_oversight/compute_human_oversight.py

# 2. Regenerate figures/tables from those CSVs
python evaluation_results/figures/generate_paper_plots.py
python evaluation_results/figures/generate_appendix_plots.py
python evaluation_results/figures/length_ordering_analysis.py
python evaluation_results/figures/length_ordering_plots.py
python evaluation_results/figures/best2_variants_table.py
```
