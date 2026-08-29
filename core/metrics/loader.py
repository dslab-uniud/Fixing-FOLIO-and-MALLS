"""Load results.jsonl files and group them into method curves for evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunRecord:
    """One evaluated instance from a results.jsonl file."""

    instance_id: str
    run_id: str
    method: str
    task: str          # 'A' or 'B'
    hyperparams: dict
    llm_name: Optional[str]   # model identifier (from config.json)
    y: str             # ground_truth_label: 'yes' | 'no'
    q: str             # quality_score:      'yes' | '?' | 'no'
    refinement: Optional[str]
    eq_i: Optional[bool]   # None means N/A (no ψ produced or not evaluated)
    logit_yes: Optional[float]
    logit_no: Optional[float]
    cost: dict
    error: Optional[str]


@dataclass
class ResultSet:
    """
    A collection of outcomes for one hyperparameter configuration of one method.

    This is one point (or, when swept, one point on a curve) in the
    Safety–Usefulness plane.
    """

    run_id: str
    method: str
    task: str
    hyperparams: dict
    llm_name: Optional[str]
    records: List[RunRecord] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Human-readable label for plots."""
        hp_str = _hp_label(self.hyperparams)
        parts = [self.method, self.task]
        if self.llm_name:
            parts.append(self.llm_name)
        if hp_str:
            parts.append(hp_str)
        return " ".join(parts)


def _hp_label(hyperparams: dict) -> str:
    if not hyperparams:
        return ""
    items = []
    for k, v in hyperparams.items():
        short_k = k.replace("prompt_version", "pv").replace("majority_label", "ml")
        items.append(f"{short_k}={v}")
    return ",".join(items)


def load_results_dir(results_dir: str | Path) -> List[RunRecord]:
    """
    Load all results.jsonl files found under *results_dir*.

    Walks one level deep: each sub-directory is assumed to be one run
    (matching the run_id pattern used by the experiment runner).
    Reads ``config.json`` (if present) to obtain the authoritative
    ``llm_name`` for each run.
    """
    results_dir = Path(results_dir)
    records: List[RunRecord] = []

    for run_dir in sorted(results_dir.iterdir()):
        jsonl_path = run_dir / "results.jsonl"
        if not jsonl_path.exists():
            continue

        # Read llm_name from config.json when available
        llm_name_override: Optional[str] = None
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                with config_path.open() as cf:
                    cfg = json.load(cf)
                    llm_name_override = cfg.get("llm_name") or None
            except (json.JSONDecodeError, OSError):
                pass

        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                logits = row.get("logits") or {}
                eq_raw = row.get("eq_i")
                # Normalise eq_i: True/False → 1/0; None/null → None
                if eq_raw is None:
                    eq_norm: Optional[bool] = None
                else:
                    eq_norm = bool(eq_raw)

                # Skip records with errors that have no quality_score
                qs = row.get("quality_score")
                if qs not in ("yes", "no", "?"):
                    qs = "?"  # treat malformed outputs as abstentions

                y = row.get("ground_truth_label", "")
                # syntactically_incorrect phi_hat is definitively wrong → treat as "no"
                if y == "syntactically_incorrect":
                    y = "no"
                if y not in ("yes", "no", "?"):
                    # 'timeout' or 'uncertain' — skip (no usable ground truth)
                    continue

                records.append(
                    RunRecord(
                        instance_id=row.get("instance_id", ""),
                        run_id=row.get("run_id", run_dir.name),
                        method=row.get("method", ""),
                        task=row.get("task", "A"),
                        hyperparams=row.get("hyperparams") or {},
                        llm_name=llm_name_override,
                        y=y,
                        q=qs,
                        refinement=row.get("refinement"),
                        eq_i=eq_norm,
                        logit_yes=logits.get("yes"),
                        logit_no=logits.get("no"),
                        cost=row.get("cost") or {},
                        error=row.get("error"),
                    )
                )

    return records


def group_into_curves(
    records: List[RunRecord],
    logit_sweep: bool = True,
    logit_steps: int = 20,
) -> Dict[str, List[ResultSet]]:
    """
    Group *records* into method curves.

    Returns a dict mapping a curve key ``"<method>_<task>_<llm>"`` to a list
    of :class:`ResultSet` objects — one per hyperparameter configuration.
    When *logit_sweep* is True **and** logit values are available in the
    records, additional synthetic result-sets are generated by sweeping the
    decision thresholds (T_yes, T_no) over a grid.

    The resulting list for each method key represents one curve in the
    Safety–Usefulness plane (sorted by increasing usefulness for clean plots).
    """
    # Group records by (method, task, llm_name) — the curve key.
    # Within each curve, each distinct run_id is one hyperparameter point.
    run_buckets: Dict[str, Dict[str, list]] = {}

    for rec in records:
        llm = rec.llm_name or "null"
        curve_key = f"{rec.method}_{rec.task}_{llm}"
        run_key = rec.run_id
        run_buckets.setdefault(curve_key, {}).setdefault(run_key, []).append(rec)

    curves: Dict[str, List[ResultSet]] = {}

    for curve_key, run_map in run_buckets.items():
        result_sets: List[ResultSet] = []

        for run_id, recs in sorted(run_map.items()):
            if not recs:
                continue
            r0 = recs[0]
            llm = r0.llm_name or "null"

            # Check whether logit values are available for this run
            has_logits = any(
                r.logit_yes is not None or r.logit_no is not None for r in recs
            )

            if has_logits and logit_sweep:
                # Generate swept result-sets from this run's logits
                swept = _sweep_logit_thresholds(recs, run_id, logit_steps)
                result_sets.extend(swept)
            else:
                result_sets.append(
                    ResultSet(
                        run_id=run_id,
                        method=r0.method,
                        task=r0.task,
                        hyperparams=r0.hyperparams,
                        llm_name=llm,
                        records=recs,
                    )
                )

        curves[curve_key] = result_sets

    return curves


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_llm(run_id: str) -> str:
    """Extract llm_name from a run_id like 'B1_A_pv1_gpt4o_mini_s0'."""
    # Convention: <method>_<task>_<hp>_<llm>_s<seed>
    # We split on '_s' (seed suffix) and take the part before it.
    # Then strip off the known prefix: method_task_hp.
    parts = run_id.rsplit("_s", 1)
    base = parts[0]
    segments = base.split("_")
    # segments[0] = method, segments[1] = task, rest = hp tokens + llm
    # llm is typically the last non-numeric, non-task token, but heuristics
    # can be fragile — just return the whole base after method+task+hp prefix
    # as a simple key.
    if len(segments) >= 3:
        return "_".join(segments[2:]) or "null"
    return "null"


def _sweep_logit_thresholds(
    records: list,
    run_id: str,
    steps: int,
) -> List[ResultSet]:
    """
    For runs where logit values are available, produce *steps* result-sets
    by sweeping a single scalar threshold τ ∈ [0, 1] in probability space:
      - p_yes = exp(logit_yes),  p_no = exp(logit_no)
      - q = 'yes'  if p_yes ≥ τ  (and p_yes ≥ p_no)
      - q = 'no'   if p_no  ≥ τ  (and p_no  >  p_yes)
      - q = '?'    otherwise (abstention)

    Raw logprobs ∈ (−∞, 0] are converted to probabilities ∈ [0, 1] via
    exp() before comparison.  This ensures that the threshold τ ∈ [0, 1]
    is on a meaningful scale (previously logprobs were compared directly
    to τ, making every swept ResultSet abstain on all instances).

    When both probabilities exceed τ, the higher probability wins
    (ties → 'yes').
    """
    import math
    import dataclasses as _dc
    import numpy as np

    result_sets = []
    r0 = records[0]

    thresholds = np.linspace(0.0, 1.0, steps)

    for tau in thresholds:
        swept_records = []
        for rec in records:
            py = math.exp(rec.logit_yes) if rec.logit_yes is not None else 0.0
            pn = math.exp(rec.logit_no)  if rec.logit_no  is not None else 0.0
            if py >= tau or pn >= tau:
                q = "yes" if py >= pn else "no"
            else:
                q = "?"
            swept_records.append(_dc.replace(rec, q=q))

        result_sets.append(
            ResultSet(
                run_id=f"{run_id}_tau{tau:.2f}",
                method=r0.method,
                task=r0.task,
                hyperparams={**r0.hyperparams, "tau": round(float(tau), 3)},
                llm_name=r0.llm_name or _infer_llm(run_id),
                records=swept_records,
            )
        )

    return result_sets


def filter_ambiguous(records: List[RunRecord]) -> List[RunRecord]:
    """Return only records where y ∈ {'yes', 'no'} (drop ambiguous y='?')."""
    return [r for r in records if r.y in ("yes", "no")]


def load_runs_csv(results_dir: str | Path) -> List[dict]:
    """Load the runs.csv registry (for metadata like dataset_split)."""
    path = Path(results_dir) / "runs.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))
