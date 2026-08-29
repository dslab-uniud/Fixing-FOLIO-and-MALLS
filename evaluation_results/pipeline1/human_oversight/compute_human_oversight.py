"""compute_human_oversight.py — Pipeline 1 human-oversight regime evaluation.

Human reviews instances in priority order:
  1. No model response (invalid/missing quality score)
  2. Pred='?' (model abstained)
  3. Pred='no' (model flagged as wrong)
  4. Pred='yes' (model approved — last to review)

Curve: x = fraction of dataset reviewed by human,
        y = fraction of dataset with CORRECT FINAL FORMULA.

The 'correct final formula' metric (greedy accuracy):
  - pred='no' + refinement:       final = refinement        → correct iff ref≡gold
  - pred='yes':                   endorses original         → correct iff gt='yes'
  - pred='?', no-response, or
    pred='no' without refinement: keeps original formula   → correct iff gt='yes'

Key curve coordinates (all fractions of N):
  y0  = greedy accuracy with no human review (automatic curation only)
  x1  = n_no_resp / N
  y1  = (n_no_resp + greedy_correct_on_responded) / N
        (human fixes no_response group; responded instances keep auto-curation result)
  x2  = (n_no_resp + n_pred_qm) / N
  y2  = (n_no_resp + n_pred_qm + greedy_correct_on_(yes+no)) / N
        (human also fixes pred='?' group; yes/no instances keep their greedy result)
        greedy_correct_on_(yes+no) = A + B  (pred=yes&gt=yes + pred=no&ref≡gold)
  x3  = (n_no_resp + n_pred_qm + n_pred_no) / N
  y3  = (n_no_resp + n_pred_qm + n_pred_no + greedy_correct_on_yes) / N
        (human also fixes pred='no' group; pred=yes instances keep their result)
        greedy_correct_on_yes = A  (pred=yes & gt=yes)

Area = 1/2 * ((1+y3)*(1-x3) + (y3+y2)*(x3-x2) + (y2+y1)*(x2-x1) + x1*y1)
T_90, T_95 = x-values where piecewise-linear curve first reaches 0.9 / 0.95.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

_OUT_DIR = Path(__file__).parent

sys.path.insert(0, str(Path(__file__).parent.parent / "automatic_regime"))
from compute_accuracy import (
    load_folio_gt,
    load_folio_gold_fols,
    load_malls_gt_and_gold,
    load_all_variants,
    _are_equivalent,
    _parse_variant_key,
    _ref_equiv_gold,
    _input_equiv_gold,
    _clean_ref,
    _VALID_Q,
)

# ---------------------------------------------------------------------------
# Human oversight metrics
# ---------------------------------------------------------------------------

def compute_oversight_metrics(
    variant_preds: Dict[str, dict],
    gt: Dict[str, str],
    gold_fols: Dict[str, str],
    input_fols: Dict[str, str],
    include_ambiguous: bool = True,
    theory: Optional[list] = None,
) -> dict:
    valid_gt = {"yes", "no", "?"} if include_ambiguous else {"yes", "no"}
    all_iids = [iid for iid, y in gt.items() if y in valid_gt]
    N = len(all_iids)
    if N == 0:
        return {}

    # Classify instances into groups
    groups: Dict[str, list] = {
        "no_response": [], "pred_?": [], "pred_no": [], "pred_yes": []
    }

    # greedy-correct counts per group
    gc_qm = 0   # greedy correct in pred_? group
    gc_no = 0   # greedy correct in pred_no group
    gc_yes = 0  # greedy correct in pred_yes group
    gc_nr = 0   # greedy correct in no_response group

    for iid in all_iids:
        gt_label = gt[iid]
        pred = variant_preds.get(iid, {})
        qs = pred.get("quality_score")
        ref = _clean_ref(pred.get("refinement"))
        input_ok = _input_equiv_gold(iid, gt_label, gold_fols, input_fols.get(iid), theory=theory)

        # Assign to group and track greedy correctness.
        # psi_hat (final formula emitted):
        #   qs='yes'           → psi_hat = input formula  (accept input, ignore any refinement)
        #   qs='no'/'?' + ref  → psi_hat = refinement
        #   qs='no'/'?' no ref → psi_hat = input formula
        #   no response        → no formula emitted → always wrong
        gold = gold_fols.get(iid, "")
        ref_correct = bool(ref and gold and _are_equivalent(ref, gold, theory=theory) is True)

        if qs not in _VALID_Q:
            groups["no_response"].append(iid)
            # no response → greedy = 0 (no formula emitted)
        elif qs == "?":
            groups["pred_?"].append(iid)
            if ref:
                if ref_correct:
                    gc_qm += 1
            elif input_ok:
                gc_qm += 1
        elif qs == "no":
            groups["pred_no"].append(iid)
            if ref:
                if ref_correct:
                    gc_no += 1
            elif input_ok:
                gc_no += 1
        else:  # yes
            groups["pred_yes"].append(iid)
            # psi_hat = input formula when verdict is 'yes' (refinement ignored)
            if input_ok:
                gc_yes += 1

    n_no_resp  = len(groups["no_response"])
    n_pred_qm  = len(groups["pred_?"])
    n_pred_no  = len(groups["pred_no"])
    n_pred_yes = len(groups["pred_yes"])

    # y0: greedy accuracy with no human review
    greedy_correct_total = gc_nr + gc_qm + gc_no + gc_yes
    y0 = greedy_correct_total / N

    # x-coordinates
    x1 = n_no_resp / N
    x2 = (n_no_resp + n_pred_qm) / N
    x3 = (n_no_resp + n_pred_qm + n_pred_no) / N

    # y1: human fixes no_response → all become correct
    greedy_correct_responded = gc_qm + gc_no + gc_yes
    y1 = (n_no_resp + greedy_correct_responded) / N

    # y2: human also fixes pred_? → all become correct
    greedy_correct_yes_no = gc_no + gc_yes
    y2 = (n_no_resp + n_pred_qm + greedy_correct_yes_no) / N

    # y3: human also fixes pred_no → all become correct
    greedy_correct_yes = gc_yes
    y3 = (n_no_resp + n_pred_qm + n_pred_no + greedy_correct_yes) / N

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
                t = (threshold - ya) / (yb - ya)
                return xa + t * (xb - xa)
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
# Dataset evaluation
# ---------------------------------------------------------------------------

_DATASET_PREFIXES = {
    "folio_validation": ["folio"],
    "malls":            ["malls"],
}


def evaluate_dataset(dataset: str, gt: Dict, gold_fols: Dict, input_fols: Dict) -> list[dict]:
    llm_data = load_all_variants(_DATASET_PREFIXES[dataset])

    rows = []
    for llm_name, vdict in llm_data.items():
        for vkey, preds in vdict.items():
            if vkey.startswith("__"):
                continue
            method, task, pv, _stage = _parse_variant_key(vkey)
            # Filter: only task B
            if task != "B":
                continue
            m_full  = compute_oversight_metrics(preds, gt, gold_fols, input_fols, include_ambiguous=True)
            m_noamb = compute_oversight_metrics(preds, gt, gold_fols, input_fols, include_ambiguous=False)
            if not m_full:
                continue

            row = {
                "dataset": dataset, "method": method, "variant": vkey,
                "model": llm_name, "task": task, "pv": pv,
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
            rows.append(row)

    rows.sort(key=lambda r: float(r["full_area"]) if r["full_area"] != "" else -1, reverse=True)
    return rows


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


def main() -> None:
    print("=== Pipeline 1 — Human-Oversight Regime ===\n")
    all_rows: list[dict] = []

    print("[1/2] FOLIO validation...")
    folio_gt                = load_folio_gt()
    folio_gold, folio_input = load_folio_gold_fols()
    folio_gt                = {iid: y for iid, y in folio_gt.items() if y != "error"}
    rows = evaluate_dataset("folio_validation", folio_gt, folio_gold, folio_input)
    _write_csv(rows, _OUT_DIR / "folio_validation_oversight.csv")
    all_rows.extend(rows)

    print("[2/2] MALLS...")
    malls_gt, malls_gold, malls_input = load_malls_gt_and_gold()
    malls_gt = {iid: y for iid, y in malls_gt.items() if y != "error"}
    rows = evaluate_dataset("malls", malls_gt, malls_gold, malls_input)
    _write_csv(rows, _OUT_DIR / "malls_oversight.csv")
    all_rows.extend(rows)

    all_rows.sort(
        key=lambda r: float(r["full_area"]) if r["full_area"] != "" else -1,
        reverse=True,
    )
    _write_csv(all_rows, _OUT_DIR / "all_datasets_oversight.csv")
    print(f"\nDone. Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
