"""compute_accuracy.py — Pipeline 1 automatic-regime accuracy evaluation.

Pipeline 1: prompting strategies (B1–B4) judge the *original* (pre-curation)
FOL formulas from the FOLIO and MALLS datasets.

For each (dataset, model, method, task, pv):
  greedy_accuracy = fraction of dataset with correct final formula after curation
    Greedy emission rule: emit the refinement when pred='no' and a refinement
    is provided; otherwise keep the input formula. The instance counts as
    correct iff the emitted formula is Z3-equivalent to the gold formula.

Data source — answers/{openai,qwen,gemma}/ flat files.
Filename conventions (all pvs merged in each file):
  {dataset}_{method}_{task}.jsonl              B1-B4 (openai/qwen/gemma)
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

_ANS_DIR = _ROOT / "pipeline1" / "assessing_original" / "results" / "predictions"
_GT_DIR  = _ROOT / "pipeline1" / "assessing_original" / "results" / "ground_truth"
_MALLS_F = _ROOT / "core" / "dataset" / "datasets" / "malls" / "dataset_rawpairs.jsonl"
_OUT_DIR = Path(__file__).parent

_VALID_Q = {"yes", "no", "?"}
_Z3_TIMEOUT = 5_000
_MAX_FORMULA_LEN = 5000


def _clean_ref(ref) -> str | None:
    """Normalise refinement: treat the string 'null' (and blank strings) as None.

    Some prompt variants emit the literal string "null" instead of a JSON
    null/omitted field when no refinement is proposed; left uncleaned, this
    string is truthy in Python and gets treated as a real (wrong) refinement,
    zeroing out otherwise-correct greedy-accuracy credit.
    """
    if ref is None or ref == "null":
        return None
    s = str(ref).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Z3 equivalence cache
# ---------------------------------------------------------------------------
_equiv_cache: Dict[Tuple, Optional[bool]] = {}


def _are_equivalent(a: str, b: str, theory: Optional[list] = None) -> Optional[bool]:
    if not a or not b or not isinstance(a, str) or not isinstance(b, str):
        return None
    if a == b:
        return True
    if len(a) > _MAX_FORMULA_LEN or len(b) > _MAX_FORMULA_LEN:
        return None
    use_theory = bool(theory)
    key = ((a, b) if a < b else (b, a)) + (use_theory,)
    if key in _equiv_cache:
        return _equiv_cache[key]
    try:
        from core.formalism.fol.fol import FOL
        from core.formalism.fol.solver_z3 import FOLSolver
        result = FOLSolver().are_equivalent(FOL(a), FOL(b), theory=theory or None, timeout=_Z3_TIMEOUT)
        _equiv_cache[key] = result
        return result
    except Exception:
        _equiv_cache[key] = None
        return None


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _load_raw_pairs(path: Path) -> list[dict]:
    decoder = json.JSONDecoder(strict=False)
    content = path.read_text(encoding="utf-8").strip()
    content = re.sub(
        r'("|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\n(\s*")',
        r'\1,\n\2', content,
    )
    records, idx = [], 0
    while idx < len(content):
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end
        except json.JSONDecodeError:
            break
    return records


def load_folio_gt() -> Dict[str, str]:
    gt: Dict[str, str] = {}
    cache = _GT_DIR / "folio.jsonl"
    if not cache.exists():
        raise FileNotFoundError(f"GT cache not found: {cache}")
    for line in cache.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gt[r["id"]] = r["y"]
    return gt


def load_folio_gold_fols() -> Tuple[Dict[str, str], Dict[str, str]]:
    gold: Dict[str, str] = {}
    input_fols: Dict[str, str] = {}
    cache = _GT_DIR / "folio.jsonl"
    if not cache.exists():
        return gold, input_fols
    for line in cache.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        iid = r["id"]
        gold[iid] = r.get("FOL_sentence", "")
        raw_old = r.get("FOL_sentence_old", "")
        if "\n" in raw_old:
            lines = [l.strip() for l in raw_old.split("\n") if l.strip()]
            input_fols[iid] = lines[0] if len(lines) == 1 else " ∧ ".join(f"({l})" for l in lines)
        else:
            input_fols[iid] = raw_old
    return gold, input_fols


def load_malls_gt_and_gold() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    records = _load_raw_pairs(_MALLS_F)
    gt: Dict[str, str] = {}
    gold: Dict[str, str] = {}
    input_fols: Dict[str, str] = {}
    for r in records:
        iid = str(r["id"])
        old_fol = r.get("FOL_sentence_old", "")
        new_fol = r.get("FOL_sentence_new", "")
        input_fols[iid] = old_fol
        gold[iid] = new_fol
        if r.get("ambiguity"):
            gt[iid] = "?"
        else:
            if not old_fol or not new_fol:
                gt[iid] = "error"
            else:
                equiv = _are_equivalent(old_fol, new_fol)
                gt[iid] = "yes" if equiv is True else "no" if equiv is False else "error"
    return gt, gold, input_fols


# ---------------------------------------------------------------------------
# Answer loading — reads from answers/{openai,qwen,gemma}/
#
# Returns model_label -> variant_key -> instance_id -> pred_dict
#
# Variant key format:
#   B1-B4 : "{method}|{task}|pv{pv}"
# ---------------------------------------------------------------------------

# Model label shown in output CSV
_FOLDER_TO_MODEL = {
    "openai": "gpt4o_mini",
    "qwen":   "qwen_local_30b",
    "gemma":  "gemma_4_31b",
}


def _ingest_record(
    r: dict,
    model: str,
    b6_raw: Dict[Tuple, list],                  # accumulator for raw B6 samples (unused; kept for signature compat)
    model_variants: Dict[str, Dict[str, Dict[str, dict]]],
) -> None:
    method = r.get("method", "")
    task   = r.get("task", "")
    pv     = r.get("pv", 1)
    iid    = r.get("instance_id", "")
    if not iid:
        return

    pred = {
        "quality_score": r.get("quality_score"),
        "refinement":    _clean_ref(r.get("refinement")),
    }
    vkey = f"{method}|{task}|pv{pv}"
    model_variants[model][vkey][iid] = pred


def load_all_variants(datasets: list[str]) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Load all predictions for the given dataset names.

    datasets: dataset names, e.g. ["folio"] or ["malls"]

    Reads from the unified layout: predictions/{baseline}/{dataset}/{model}/results.jsonl
    """
    model_variants: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    b6_raw: Dict[Tuple, list] = defaultdict(list)

    dataset_set = set(datasets)
    _BASELINES = ["B1", "B2", "B3", "B4"]

    for baseline in _BASELINES:
        baseline_dir = _ANS_DIR / baseline
        if not baseline_dir.exists():
            continue
        for ds in dataset_set:
            ds_dir = baseline_dir / ds
            if not ds_dir.exists():
                continue
            for folder_name, model_label in _FOLDER_TO_MODEL.items():
                fpath = ds_dir / folder_name / "results.jsonl"
                if not fpath.exists():
                    continue
                for r in _load_jsonl(fpath):
                    _ingest_record(r, model_label, b6_raw, model_variants)

    # Aggregate raw B6 samples by (model, task, pv)
    by_instance: Dict[Tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (model, task, pv), samples in b6_raw.items():
        for s in samples:
            by_instance[(model, task, pv)][s["instance_id"]].append(s)

    for (model, task, pv), iid_samples in by_instance.items():
        # 'all_agree' and 'logit_mean' aggregate without needing the input formula
        # — finalise them now.
        for agg_mode in ("all_agree", "logit_mean"):
            vkey = f"B6_{agg_mode}|{task}|pv{pv}"
            for iid, samples in iid_samples.items():
                model_variants[model][vkey][iid] = _agg_b6(samples, agg_mode)
        # 'majority' uses Z3-equivalence partitioning over refinement candidates
        # (with the input formula substituted when a sample emitted no refinement).
        # Defer aggregation until the input FOLs are available.
        vkey_maj = f"B6_majority|{task}|pv{pv}"
        for iid, samples in iid_samples.items():
            model_variants[model][vkey_maj][iid] = {"__raw_samples__": samples}

    return {k: dict(v) for k, v in model_variants.items()}


def finalize_b6_majority(
    model_variants: Dict[str, Dict[str, Dict[str, dict]]],
    input_fols: Dict[str, str],
    theory: Optional[list] = None,
) -> None:
    """Finalize deferred B6 majority aggregation by partitioning samples'
    candidate formulas (refinement, or the input formula when qs=='yes' and no
    refinement was emitted) into Z3-equivalence clusters and selecting the
    largest cluster's representative."""
    for _, vdict in model_variants.items():
        for vkey, preds in vdict.items():
            if not vkey.startswith("B6_majority|"):
                continue
            for iid in list(preds.keys()):
                pred = preds[iid]
                samples = pred.get("__raw_samples__") if isinstance(pred, dict) else None
                if samples is None:
                    continue
                preds[iid] = _agg_b6_majority_z3(
                    samples, input_fols.get(iid), theory=theory
                )


# ---------------------------------------------------------------------------
# B6 sample aggregation
# ---------------------------------------------------------------------------

def _softmax(logits: dict) -> dict:
    vals = {k: v for k, v in logits.items() if v is not None and k in _VALID_Q}
    if not vals:
        return {}
    max_v = max(vals.values())
    exps = {k: math.exp(v - max_v) for k, v in vals.items()}
    total = sum(exps.values())
    return {k: e / total for k, e in exps.items()}


def _normalize_refinement(s: str) -> str:
    # Bucket near-duplicate refinement strings for majority voting.
    return re.sub(r"\s+", " ", s).strip()


def _agg_b6_majority_z3(
    samples: list[dict],
    input_fol: Optional[str],
    theory: Optional[list] = None,
) -> dict:
    """Partition the per-sample emitted formulas by Z3-equivalence and return
    the representative of the largest cluster.

    Each sample contributes one candidate formula:
      - qs == 'no'  + refinement       → the refinement
      - qs == 'yes'                    → the input formula (sample kept input)
      - qs == '?'  or qs == 'no' w/o refinement → no candidate (sample skipped)
    """
    candidates: list[str] = []
    for s in samples:
        qs = s.get("quality_score")
        ref = s.get("refinement") or None
        if qs == "yes":
            if input_fol:
                candidates.append(input_fol)
        elif qs == "no" and ref:
            candidates.append(ref)
        # qs == "?" or unrecognized → skip

    if not candidates:
        return {"quality_score": None, "refinement": None}

    # Cluster by Z3 equivalence. First occurrence in `candidates` becomes the
    # representative for its cluster.
    cluster_reps: list[str] = []
    cluster_size: list[int] = []
    for cand in candidates:
        placed = False
        for i, rep in enumerate(cluster_reps):
            if rep == cand:
                cluster_size[i] += 1
                placed = True
                break
            eq = _are_equivalent(cand, rep, theory=theory)
            if eq is True:
                cluster_size[i] += 1
                placed = True
                break
        if not placed:
            cluster_reps.append(cand)
            cluster_size.append(1)

    # Largest cluster wins (ties → first-seen, which matches "choose one").
    best_idx = max(range(len(cluster_reps)), key=lambda i: cluster_size[i])
    chosen = cluster_reps[best_idx]

    # Verdict: "yes" if the chosen formula matches the input, else "no".
    if input_fol and (chosen == input_fol or _are_equivalent(chosen, input_fol, theory=theory) is True):
        return {"quality_score": "yes", "refinement": None}
    return {"quality_score": "no", "refinement": chosen}


def _agg_b6(samples: list[dict], mode: str) -> dict:
    scores = [s["quality_score"] for s in samples if s.get("quality_score") in _VALID_Q]
    refinements = [s.get("refinement") for s in samples if s.get("refinement")]
    if refinements:
        norm_counts: Counter = Counter()
        norm_to_orig: dict = {}
        for r in refinements:
            key = _normalize_refinement(r)
            norm_counts[key] += 1
            norm_to_orig.setdefault(key, r)
        winner_key = norm_counts.most_common(1)[0][0]
        ref_majority = norm_to_orig[winner_key]
    else:
        ref_majority = None

    if mode == "majority":
        if not scores:
            return {"quality_score": None, "refinement": ref_majority}
        counts = Counter(scores)
        best = max(counts.values())
        winners = [k for k, v in counts.items() if v == best]
        return {"quality_score": winners[0] if len(winners) == 1 else "?", "refinement": ref_majority}

    elif mode == "all_agree":
        if not scores:
            return {"quality_score": None, "refinement": ref_majority}
        unique = set(scores)
        return {"quality_score": unique.pop() if len(unique) == 1 else "?", "refinement": ref_majority}

    elif mode == "logit_mean":
        sum_probs: dict = {"yes": 0.0, "no": 0.0, "?": 0.0}
        n_valid = 0
        for s in samples:
            probs = _softmax(s.get("logits") or {})
            if not probs:
                continue
            for k in _VALID_Q:
                sum_probs[k] += probs.get(k, 0.0)
            n_valid += 1
        if n_valid == 0:
            return _agg_b6(samples, "majority")
        avg = {k: v / n_valid for k, v in sum_probs.items()}
        return {"quality_score": max(avg, key=avg.get), "refinement": ref_majority}

    raise ValueError(f"Unknown B6 mode: {mode}")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _ref_equiv_gold(qs, ref, iid, gold_fols, theory=None) -> bool:
    if qs != "no" or not ref:
        return False
    gold = gold_fols.get(iid, "")
    return bool(gold and _are_equivalent(ref, gold, theory=theory) is True)


def _input_equiv_gold(iid, gt_label, gold_fols, input_fol, theory=None) -> bool:
    if gt_label == "yes":
        return True
    if gt_label == "no":
        return False
    gold = gold_fols.get(iid, "")
    if not gold or not input_fol:
        return False
    return _are_equivalent(input_fol, gold, theory=theory) is True


def compute_accuracy_metrics(
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
        return {"N": 0, "N_responded": 0, "greedy_accuracy": None}

    greedy_correct = N_responded = 0

    for iid in all_iids:
        gt_label = gt[iid]
        pred = variant_preds.get(iid, {})
        qs = pred.get("quality_score")
        ref = pred.get("refinement") or None

        if qs in _VALID_Q:
            N_responded += 1

        input_ok = _input_equiv_gold(iid, gt_label, gold_fols, input_fols.get(iid), theory=theory)

        # Greedy emission rule:
        #   qs == "no"  → emit refinement   (correct iff ref ≡ gold)
        #   qs == "yes" → keep input        (correct iff input ≡ gold, i.e. gt == "yes")
        #   qs == "?"   → abstain           (no formula → wrong)
        #   no response → wrong
        if qs == "yes":
            if input_ok:
                greedy_correct += 1
        elif qs == "no" and ref:
            gold = gold_fols.get(iid, "")
            if gold and _are_equivalent(ref, gold, theory=theory) is True:
                greedy_correct += 1

    return {
        "N":               N,
        "N_responded":     N_responded,
        "greedy_accuracy": round(greedy_correct / N, 4),
    }


# ---------------------------------------------------------------------------
# Variant key parsing: "{method}|{task}|pv{pv}[|nc{k}]"
# ---------------------------------------------------------------------------

def _parse_variant_key(vkey: str) -> Tuple[str, str, str, str]:
    parts = vkey.split("|")
    method = parts[0] if len(parts) > 0 else ""
    task   = parts[1] if len(parts) > 1 else ""
    pv     = parts[2] if len(parts) > 2 else ""
    stage  = parts[3] if len(parts) > 3 else ""
    return method, task, pv, stage


# ---------------------------------------------------------------------------
# Z3 pre-computation
# ---------------------------------------------------------------------------

def _precompute_z3(model_variants, gold_fols, theory=None) -> None:
    print("  Pre-computing Z3 equivalence checks...")
    pairs: set = set()
    for _, vdict in model_variants.items():
        for _, preds in vdict.items():
            for iid, pred in preds.items():
                if pred.get("quality_score") == "no" and pred.get("refinement"):
                    gold = gold_fols.get(iid, "")
                    if gold:
                        pairs.add((pred["refinement"], gold))
    total = len(pairs)
    print(f"  Total unique (ref, gold) pairs to check: {total}")
    for done, (ref, gold) in enumerate(pairs, 1):
        _are_equivalent(ref, gold, theory=theory)
        if done % 200 == 0:
            print(f"    Z3 progress: {done}/{total}", flush=True)
    print(f"  Z3 done ({total} pairs).")


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset_label: str,
    gt: Dict,
    gold_fols: Dict,
    input_fols: Dict,
    model_variants: Dict[str, Dict[str, Dict[str, dict]]],
) -> list[dict]:
    finalize_b6_majority(model_variants, input_fols)
    _precompute_z3(model_variants, gold_fols)

    rows = []
    for model, vdict in model_variants.items():
        for vkey, preds in vdict.items():
            method, task, pv, stage = _parse_variant_key(vkey)
            if task != "B":
                continue
            m_full  = compute_accuracy_metrics(preds, gt, gold_fols, input_fols, include_ambiguous=True)
            m_noamb = compute_accuracy_metrics(preds, gt, gold_fols, input_fols, include_ambiguous=False)
            rows.append({
                "dataset":                dataset_label,
                "method":                 method,
                "variant":                vkey,
                "model":                  model,
                "task":                   task,
                "pv":                     pv,
                "stage":                  stage,
                "N_full":                 m_full["N"],
                "N_responded_full":       m_full["N_responded"],
                "greedy_accuracy_full":   m_full["greedy_accuracy"] if m_full["greedy_accuracy"] is not None else "",
                "N_wo_amb":               m_noamb["N"],
                "N_responded_wo_amb":     m_noamb["N_responded"],
                "greedy_accuracy_wo_amb": m_noamb["greedy_accuracy"] if m_noamb["greedy_accuracy"] is not None else "",
            })
    rows.sort(key=lambda r: float(r["greedy_accuracy_full"]) if r["greedy_accuracy_full"] != "" else -1, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Pipeline 1 — Automatic Regime Accuracy ===\n")
    all_rows: list[dict] = []

    print("[1/2] FOLIO validation...")
    folio_gt              = load_folio_gt()
    folio_gold, folio_input = load_folio_gold_fols()
    folio_gt              = {iid: y for iid, y in folio_gt.items() if y != "error"}
    folio_variants        = load_all_variants(["folio"])
    rows = evaluate_dataset("folio_validation", folio_gt, folio_gold, folio_input, folio_variants)
    _write_csv(rows, _OUT_DIR / "folio_validation_accuracy.csv")
    all_rows.extend(rows)

    print("[2/2] MALLS...")
    malls_gt, malls_gold, malls_input = load_malls_gt_and_gold()
    malls_gt       = {iid: y for iid, y in malls_gt.items() if y != "error"}
    malls_variants = load_all_variants(["malls"])
    rows = evaluate_dataset("malls", malls_gt, malls_gold, malls_input, malls_variants)
    _write_csv(rows, _OUT_DIR / "malls_accuracy.csv")
    all_rows.extend(rows)

    all_rows.sort(
        key=lambda r: float(r["greedy_accuracy_full"]) if r["greedy_accuracy_full"] != "" else -1,
        reverse=True,
    )
    _write_csv(all_rows, _OUT_DIR / "all_datasets_accuracy.csv")
    print(f"\nDone. Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
