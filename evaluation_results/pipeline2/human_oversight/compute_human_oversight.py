"""compute_human_oversight.py — Pipeline 2 human-oversight regime evaluation.

Same curve structure as Pipeline 1, but applied to pipeline2 phi_hat assessment.

The 'correct final formula' (greedy accuracy):
  - pred='no' + eq_i=True:  refinement used → correct
  - pred='yes' + gt='yes':  phi_hat endorsed → correct
  - pred='yes' + gt≠'yes':  phi_hat endorsed → wrong
  - pred='?', no-response, pred='no'+eq_i≠True: phi_hat kept → correct iff gt='yes'

Curve coordinates (fractions of N):
  y0  = greedy accuracy (automatic only)
  x1  = n_no_resp / N
  y1  = (n_no_resp + greedy_correct_responded) / N
  x2  = (n_no_resp + n_pred_qm) / N
  y2  = (n_no_resp + n_pred_qm + greedy_correct_yes_no) / N
  x3  = (n_no_resp + n_pred_qm + n_pred_no) / N
  y3  = (n_no_resp + n_pred_qm + n_pred_no + greedy_correct_yes) / N

Area = 1/2 * ((1+y3)*(1-x3) + (y3+y2)*(x3-x2) + (y2+y1)*(x2-x1) + x1*y1)
T_90, T_95 = x where curve first reaches 0.9 / 0.95.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

sys.path.insert(0, str(Path(__file__).parent.parent / "automatic_regime"))
from compute_accuracy import (
    _load_jsonl,
    _group_by_variant,
    _variant_tag,
    _gt_normalise,
    load_phi_hat_gt,
    phi_hat_gt_canonical,
    _are_equivalent,
    _clean_ref,
    _VALID_Q,
    _P2_DIR,
)

_OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_oversight_metrics(
    records: list[dict],
    include_ambiguous: bool = True,
    phi_hat_gt: Optional[Dict[str, str]] = None,
) -> dict:
    valid_gt_set = {"yes", "no", "?"} if include_ambiguous else {"yes", "no"}

    # Build index from records
    rec_by_id: Dict[str, dict] = {}
    for rec in records:
        iid = rec.get("id", "")
        if iid:
            rec_by_id[iid] = rec

    # Determine full instance set
    if phi_hat_gt is not None:
        canonical = phi_hat_gt_canonical(phi_hat_gt)
        phi_hat_ids = phi_hat_gt.get("__phi_hat_ids__", canonical)
        all_instances = []
        for ds_id, phi_id in zip(canonical, phi_hat_ids):
            gt = _gt_normalise(phi_hat_gt[ds_id])
            if gt not in valid_gt_set:
                continue
            rec = rec_by_id.get(ds_id) or rec_by_id.get(phi_id) or {}
            all_instances.append((ds_id, gt, rec))
    else:
        all_instances = [
            (rec.get("id", ""), _gt_normalise(rec.get("ground_truth_label", "")), rec)
            for rec in records
            if _gt_normalise(rec.get("ground_truth_label", "")) in valid_gt_set
        ]

    N = len(all_instances)
    if N == 0:
        return {}

    gc_nr = gc_qm = gc_no = gc_yes = 0
    n_no_resp = n_pred_qm = n_pred_no = n_pred_yes = 0

    for iid, gt, rec in all_instances:
        qs   = rec.get("quality_score")
        ref  = _clean_ref(rec.get("refinement"))

        # Recompute eq_i live from current phi_gold (don't trust stale rec['eq_i'])
        if ref and phi_hat_gt is not None:
            phi_gold_str = phi_hat_gt.get("__phi_gold_strs__", {}).get(iid, "")
            eq_i = _are_equivalent(ref, phi_gold_str) if phi_gold_str else None
        else:
            eq_i = None

        ref_correct = bool(ref and eq_i is True)  # refinement≡gold regardless of qs

        # phi_hat ≡ phi_gold: encoded in gt for non-ambiguous; checked explicitly for gt='?'
        if gt == "yes":
            input_ok = True
        elif gt == "?" and phi_hat_gt is not None:
            phi_hat_str  = phi_hat_gt.get("__phi_hat_strs__", {}).get(iid, "")
            phi_gold_str = phi_hat_gt.get("__phi_gold_strs__", {}).get(iid, "")
            input_ok = _are_equivalent(phi_hat_str, phi_gold_str) is True
        else:
            input_ok = False

        # Assign to group and track greedy correctness.
        # psi_hat (final formula emitted):
        #   qs='yes'           → psi_hat = phi_hat  (accept phi_hat, ignore any refinement)
        #   qs='no'/'?' + ref  → psi_hat = refinement
        #   qs='no'/'?' no ref → psi_hat = phi_hat
        #   no response        → no formula emitted → always wrong
        if qs not in _VALID_Q:
            n_no_resp += 1
            # no response → greedy = 0
        elif qs == "?":
            n_pred_qm += 1
            if ref:
                if ref_correct:
                    gc_qm += 1
            elif input_ok:
                gc_qm += 1
        elif qs == "no":
            n_pred_no += 1
            if ref:
                if ref_correct:
                    gc_no += 1
            elif input_ok:
                gc_no += 1
        else:  # yes
            n_pred_yes += 1
            # psi_hat = phi_hat when verdict is 'yes' (refinement ignored)
            if input_ok:
                gc_yes += 1

    if N == 0:
        return {}

    greedy_total = gc_nr + gc_qm + gc_no + gc_yes
    y0 = greedy_total / N

    x1 = n_no_resp / N
    x2 = (n_no_resp + n_pred_qm) / N
    x3 = (n_no_resp + n_pred_qm + n_pred_no) / N

    y1 = (n_no_resp + gc_qm + gc_no + gc_yes) / N
    y2 = (n_no_resp + n_pred_qm + gc_no + gc_yes) / N
    y3 = (n_no_resp + n_pred_qm + n_pred_no + gc_yes) / N

    area = 0.5 * (
        (1 + y3) * (1 - x3)
        + (y3 + y2) * (x3 - x2)
        + (y2 + y1) * (x2 - x1)
        + (y0 + y1) * x1
    )

    def _find_threshold(threshold: float) -> Optional[float]:
        if y0 >= threshold:
            return 0.0
        segments = [
            (0.0, y0, x1, y1),
            (x1, y1, x2, y2),
            (x2, y2, x3, y3),
            (x3, y3, 1.0, 1.0),
        ]
        for xa, ya, xb, yb in segments:
            if ya <= threshold <= yb:
                if yb == ya:
                    return xa
                return xa + (threshold - ya) / (yb - ya) * (xb - xa)
        return None

    T_90 = _find_threshold(0.90)
    T_95 = _find_threshold(0.95)

    return {
        "N":          N,
        "n_no_resp":  n_no_resp,
        "n_pred_qm":  n_pred_qm,
        "n_pred_no":  n_pred_no,
        "n_pred_yes": n_pred_yes,
        "y0":         round(y0, 4),
        "x1":         round(x1, 4),
        "x2":         round(x2, 4),
        "x3":         round(x3, 4),
        "y1":         round(y1, 4),
        "y2":         round(y2, 4),
        "y3":         round(y3, 4),
        "area":       round(area, 4),
        "T_90":       round(T_90, 4) if T_90 is not None else None,
        "T_95":       round(T_95, 4) if T_95 is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Pipeline 2 — Human-Oversight Regime ===\n")

    baselines = ["B0", "B1", "B2", "B3", "B4"]
    all_rows: list[dict] = []
    phi_prompt = "P1"   # only P1 phi_hat generation is shipped/evaluated

    for baseline in baselines:
        baseline_dir = _P2_DIR / baseline
        if not baseline_dir.exists():
            continue

        for dataset_dir in sorted(baseline_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            source = dataset_dir.name  # folio / malls

            for judge_dir in sorted(dataset_dir.iterdir()):
                if not judge_dir.is_dir():
                    continue
                judge = judge_dir.name
                results_f = judge_dir / "results.jsonl"
                if not results_f.exists():
                    continue

                records = _load_jsonl(results_f)

                for (phi_model, pv_num, n_ex), rows in _group_by_variant(records).items():
                    phi_gt = load_phi_hat_gt(source, phi_model, phi_prompt)
                    pv_tag, task_tag = _variant_tag(rows)

                    m_full  = compute_oversight_metrics(rows, include_ambiguous=True, phi_hat_gt=phi_gt)
                    m_noamb = compute_oversight_metrics(rows, include_ambiguous=False, phi_hat_gt=phi_gt)
                    if not m_full:
                        continue
                    all_rows.append(_build_row(
                        baseline, source, phi_model, phi_prompt,
                        judge, pv_tag, task_tag, m_full, m_noamb,
                    ))

    # Per-dataset CSVs
    by_ds: Dict[str, list] = defaultdict(list)
    for row in all_rows:
        by_ds[row["source_dataset"]].append(row)

    for ds, rows in by_ds.items():
        rows.sort(key=lambda r: float(r["full_area"]) if r["full_area"] != "" else -1, reverse=True)
        label = "folio_validation" if ds == "folio" else ds
        _write_csv(rows, _OUT_DIR / f"{label}_oversight.csv")

    all_rows.sort(
        key=lambda r: float(r["full_area"]) if r["full_area"] != "" else -1,
        reverse=True,
    )
    _write_csv(all_rows, _OUT_DIR / "all_datasets_oversight.csv")
    print(f"\nDone. Total rows: {len(all_rows)}")


def _build_row(
    baseline: str, source: str, phi_model: str, phi_prompt: str,
    judge: str, pv: str, task_tag: str,
    m_full: dict, m_noamb: dict,
) -> dict:
    row = {
        "source_dataset": source,
        "phi_model":      phi_model,
        "phi_prompt":     phi_prompt,
        "baseline":       baseline,
        "judge_model":    judge,
        "pv":             pv,
        "task_tag":       task_tag,
    }
    full_keys = ["N", "n_no_resp", "n_pred_qm", "n_pred_no", "n_pred_yes",
                 "y0", "x1", "x2", "x3", "y1", "y2", "y3",
                 "area", "T_90", "T_95"]
    for k in full_keys:
        row[f"full_{k}"] = m_full.get(k, "")
    noamb_keys = ["N", "n_no_resp", "n_pred_no", "n_pred_yes",
                  "y0", "x1", "x2", "x3", "y1", "y2", "y3",
                  "area", "T_90", "T_95"]
    for k in noamb_keys:
        row[f"noamb_{k}"] = m_noamb.get(k, "")
    return row


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"  Wrote {len(rows)} rows → {path}")


if __name__ == "__main__":
    main()
