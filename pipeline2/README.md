# pipeline2 — Judging LLM-Generated Formalizations

Pipeline 2 has two stages:

1. **Stage 1 (`phi_hat_generation/`)** — an LLM autoformalizer generates a candidate formula `phi_hat` for each NL sentence (using prompt variants P1 and P2).
2. **Stage 2 (`assessing_phi_hat/`)** — prompting strategies B0–B4 judge the quality of those generated formulas against gold via Z3 equivalence.

## Files

| File                                  | Purpose |
|---------------------------------------|---------|
| `evaluate.py`                         | Canonical evaluator: per-strategy metric tables + ensemble analysis |
| `run_gemma_phi_hat.py`                | Local Gemma autoformalizer (Stage 1) |
| `run_qwen_malls_phi_hat.py`           | Local Qwen autoformalizer for MALLS (Stage 1) |
| `prepare_malls_phi_hat_batch.py`      | OpenAI batch prep for Stage 1 on MALLS |
| `run_gemma_assess_phi_hat.py`         | Local Gemma judge over generated phi_hat (Stage 2) |
| `run_qwen_assess.py`                  | Local Qwen judge (Stage 2; B1–B4) |
| `prepare_malls_assess_phi_hat_batch.py` | OpenAI batch prep for Stage 2 on MALLS |
| `process_gemma_responses.py`          | Parse Gemma raw outputs → results.jsonl |

## Data layout

```
pipeline2/
├── phi_hat_generation/                Stage 1
│   ├── requests/                      Batch requests for autoformalization
│   ├── responses/                     Raw generation responses
│   └── results/                       Generated FOL formulas
│       └── {dataset}__{model}__{P1|P2}.jsonl
│
└── assessing_phi_hat/                 Stage 2
    ├── requests/{openai,qwen,gemma}/  Batch requests for judging
    ├── responses/{openai,qwen,gemma}/ Raw judging responses
    └── results/
        ├── B0/                        Trivial baselines (always-yes / always-no / random / majority-class)
        │   └── {dataset}_{judge_model}_{P1|P2}/
        │       └── {variant_id}/
        │           ├── config.json
        │           └── results.jsonl
        ├── B1/ … B4/                  LLM-as-judge prompting strategies
        │   └── {dataset}_{judge_model}_{P1|P2}/
        │       └── {variant_id}/
        │           └── results.jsonl
        │       __Archive/             (legacy Task A run dirs; auto-skipped by evaluate.py)
        └── tables/                    Metric tables
            ├── folio/{per_strategy,ensembles}/
            └── malls/{per_strategy,ensembles}/
```

> The `requests/` and `responses/` directories together total ~3 GB and are **not included in the public release** — extracted `results.jsonl` files are sufficient to reproduce all metric tables via `evaluate.py`.

## Filename and run-id conventions

**Generated phi_hat files** (Stage 1 outputs):
```
{dataset}__{generator_model}__{prompt_id}.jsonl
```
e.g. `folio__gpt-4o-mini__P1.jsonl`. Each record carries: `id`, `id_source_dataset`, `source_dataset`, `generator_model`, `generator_prompt_id`, `p` (NL), `phi_gold`, `phi_hat`, `reasoning`, `ambiguity_flag`.

**Judging run directory** (Stage 2): `B{N}/{dataset}_{judge_model}_{P1|P2}/{variant_id}/results.jsonl`
where `variant_id` encodes the judge configuration:
- B1–B4: `{judge}_v{1-4}_B`

Each judging record carries: `instance_id`, `run_id`, `method`, `task`, `hyperparams`, `llm_name`, `ground_truth_label`, `quality_score`, `reasoning`, `comment`, `refinement`, `eq_i` (refinement Z3-equivalent to gold? boolean), `logprobs`.

## How to reproduce

```bash
# 1. (Optional) Generate phi_hat with your autoformalizer of choice
python pipeline2/run_gemma_phi_hat.py
# (or run_qwen_malls_phi_hat.py / OpenAI batch via prepare_malls_phi_hat_batch.py)

# 2. (Optional) Judge generated phi_hat with B0–B4
python pipeline2/run_gemma_assess_phi_hat.py
python pipeline2/run_qwen_assess.py
# (or OpenAI batch via prepare_malls_assess_phi_hat_batch.py)

# 3. Compute metrics — works on shipped results/B*/.../results.jsonl
python pipeline2/evaluate.py

# Restrict to specific datasets or strategies
python pipeline2/evaluate.py --datasets folio_gpt4omini_P1 folio_qwen3_P1
python pipeline2/evaluate.py --baselines B0 B1 B2
python pipeline2/evaluate.py --no-ensemble       # skip the (slow) ensemble pass
```

Step 3 alone reproduces all tables in `results/tables/` from the shipped `results/B*/` directories.
