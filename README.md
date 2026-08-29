# DatasetCuration

This repository contains the code and curated datasets accompanying our paper **Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling** on LLM-based curation of First-Order Logic (FOL) datasets.

We evaluate whether LLM prompting strategies (B1–B4) can automatically detect errors and ambiguities in FOL formalizations from two datasets — **FOLIO** (validation split) and **MALLS** (subset of the test split) — and propose corrections. We release human-curated versions of these datasets together with a reproducible evaluation pipeline.

---

## Project Structure

```
DatasetCuration/
├── core/                                Core library
│   ├── dataset/datasets/                Curated datasets (folio, malls) — see "Datasets" below
│   ├── formalism/fol/                   FOL parser, Z3 solver, AST utilities
│   ├── llm/                             LLM interfaces (OpenAI, Qwen, local HF models)
│   ├── strategies/                      B1–B4 implementations + prompt templates
│   └── metrics/                         Metric computation and ensemble aggregation
│
├── pipeline1/                           Audit existing formalizations (Pipeline 1)
│   └── assessing_original/
│       ├── requests/                    Batch request files (per model)
│       └── results/
│           ├── ground_truth/            Z3-precomputed GT labels (cached)
│           └── predictions/             Extracted LLM predictions
│               └── {openai,qwen,gemma}/{folio,malls}/
│
├── pipeline2/                           Judge generated formalizations (Pipeline 2)
│   ├── phi_hat_generation/results/      LLM-generated FOL formulas (Stage 1)
│   └── assessing_phi_hat/results/       B0–B4 judgments of generated formulas (Stage 2)
│
├── evaluation_results/                  Aggregated evaluations and paper figures
│   └── figures/                         Scripts that regenerate every figure/table
│                                         in the paper from the data above
│
├── tests/                               Unit tests for the FOL parser
├── configs/                             YAML configs for the LLM model factory
└── requirements.txt                     Python dependencies
```

Each top-level directory has its own `README.md` describing its contents in more detail.

---

## Datasets

| Dataset      | Size           | Role          | Source |
|--------------|----------------|---------------|--------|
| **FOLIO**    | 275            | Primary (NLI + autoformalization) | Han et al. 2022 |
| **MALLS**    | 100 (subset)   | Secondary (spatial reasoning)     | Yang et al. ACL 2024 |

The curated datasets are the primary artifact of this release:

- `core/dataset/datasets/folio/dataset_rawpairs.jsonl` — 275 FOLIO validation instances, each with the original formula (`FOL_sentence_old`), the corrected formula (`FOL_sentence_new`), ambiguity annotations, and correction explanations.
- `core/dataset/datasets/malls/dataset_rawpairs.jsonl` — 100 MALLS test instances, same schema.

They are also released on a public archive (HuggingFace / Zenodo — link to be added at camera-ready).

---

## Regenerating the Paper's Figures and Tables

All figures and tables reported in the paper can be regenerated from the data shipped in this repository via the scripts in `evaluation_results/figures/`:

```bash
python evaluation_results/figures/generate_paper_plots.py       # Figures: claim1, claim3, appendix AAG/T90/T95
python evaluation_results/figures/generate_appendix_plots.py    # Appendix B/C/E/F figures
python evaluation_results/figures/length_ordering_analysis.py   # (run first) builds a records cache
python evaluation_results/figures/length_ordering_plots.py      # Appendix length-ordering figure
python evaluation_results/figures/best2_variants_table.py       # Best-2-variants LaTeX table
```

Note: figures in this public release are computed on **FOLIO and MALLS only**. The paper's GGC/Stanford panel is omitted, since GGC is a private dataset not redistributed here (data, intermediate NL sentences, and LLM responses are all excluded).

---

## License

- **Code**: MIT
- **Curated FOLIO and MALLS subsets**: CC BY 4.0 (matching upstream licenses)

---

## Citation


