# pipeline1 — Auditing Existing FOL Formalizations

Pipeline 1 judges the **original** (pre-curation) formulas of two FOL datasets — FOLIO and MALLS — with prompting strategies B1–B4, and compares the predictions against ground-truth Z3 equivalence labels.

For each instance `(NL_sentence, FOL_old, FOL_gold)`, strategies decide whether `FOL_old ≡ FOL_gold` and (Configuration B) propose a refined formula ψ when they predict `no`.

## Files

| File                          | Purpose |
|-------------------------------|---------|
| `prepare_batch.py`            | Build B1–B4 batch request files for FOLIO |
| `prepare_malls_batch.py`      | Build B1–B4 batch request files for MALLS |
| `extract_answers.py`          | Parse raw LLM responses → clean prediction records, routed into `results/predictions/{model}/{folio,malls}/` |
| `evaluate.py`                 | Compute GT labels (Z3), per-strategy metric tables, ensemble analysis, hard-instance diagnostics |
| `run_gemma_folio.py`          | Local Gemma run on FOLIO |
| `run_gemma_folio_train.py`    | Local Gemma run on FOLIO train split |
| `run_gemma_malls.py`          | Local Gemma run on MALLS |
| `run_qwen_malls.py`           | Local Qwen run on MALLS |
| `batch/`                      | Helpers shared by `prepare_*` scripts |

## Data layout

```
pipeline1/assessing_original/
├── requests/{openai,qwen,gemma}/    Batch request files (per model)
├── responses/{openai,qwen,gemma}/   Raw LLM responses (per model)
└── results/
    ├── ground_truth/                Z3-precomputed GT labels (cached; safe to ship)
    │   └── folio.jsonl
    ├── predictions/                 Extracted per-instance predictions
    │   ├── openai/
    │   │   ├── folio/               folio_{B1-4}_B.jsonl
    │   │   └── malls/               malls_{B1-4}_B.jsonl
    │   ├── qwen/                    same subfolder structure
    │   └── gemma/                   same subfolder structure
    ├── tables/                      Metric tables (.txt human-readable + .csv)
    │   ├── folio/
    │   └── malls/
    ├── hard_instances/              Top-K hardest instances per dataset/model
```

> The `requests/` and `responses/` directories together total ~30 GB and are **not included in the public release** — they can be regenerated from `prepare_batch.py` at the cost of LLM API calls. The extracted predictions in `results/predictions/` are sufficient to reproduce all reported tables via `evaluate.py`.

## Filename convention

Prediction files (and the request/response files they derive from):
```
{dataset}_{method}_{task}.jsonl
```
where:
- `dataset` ∈ {`folio`, `malls`}
- `method` ∈ {`B1`, `B2`, `B3`, `B4`}
- `task` is always `B` (Configuration B; legacy Configuration A files are in `_Archive/`)

Each prediction record carries: `custom_id`, `instance_id`, `method`, `task`, `pv` (prompt version 1–4), `llm`, `quality_score` ∈ {`yes`,`no`,`?`}, `reasoning`, `comment`, `refinement` (ψ string or null), `logits`, `error`.

## How to reproduce

```bash
# 1. Generate batch request files (only if running new LLM calls)
python pipeline1/prepare_batch.py
python pipeline1/prepare_malls_batch.py

# 2. Send batches to OpenAI / run local models — outputs land in responses/

# 3. Parse raw responses into clean prediction records
python pipeline1/extract_answers.py

# 4. Compute GT (Z3) and write all metric tables
python pipeline1/evaluate.py
```

Step 4 alone reproduces all tables in `results/tables/folio/` from the shipped `predictions/` files.
