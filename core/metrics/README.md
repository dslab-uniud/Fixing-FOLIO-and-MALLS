# core/metrics — Evaluation Metrics

## Modules

| File           | Purpose |
|----------------|---------|
| `compute.py`   | `compute_metrics(predictions, ground_truth, gold_fols)` → safety/usefulness dict |
| `loader.py`    | `ResultSet` and `RunRecord` — load Pipeline 2 `results.jsonl` files |
| `aggregator.py`| Ensemble verdict & refinement aggregation rules |
| `plot.py`      | Plotting helpers |

## Metrics

| Name                  | Definition |
|-----------------------|-----------|
| `safety_yes`          | `n_correct_yes / pred_yes` — precision of `yes` calls |
| `safety_3`            | `(n_correct_yes + n_correct_no) / (pred_yes + pred_no)` — precision of all committed calls |
| `usefulness`          | `(pred_yes + pred_no) / N` — fraction of instances where the model commits |
| `n_correct_refinement`| number of proposed refinements ψ that are Z3-equivalent to the gold formula |

## Ensemble rules

| Rule          | k | Verdict aggregation |
|---------------|---|---------------------|
| `pair_safe`   | 2 | both must agree on the verdict; otherwise `?` |
| `pair_useful` | 2 | one commits + one abstains → take the committed verdict |
| `safe`        | 3 | all three must agree |
| `majority`    | 3 | strict majority among committed verdicts; tie → `?` |

For refinements, **safe** ensembles propose ψ only if all members propose Z3-equivalent strings; **useful** ensembles propose the most-voted refinement.
