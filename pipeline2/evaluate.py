"""evaluate.py — Evaluate Pipeline 2 prompting strategy results for FOLIO dataset curation.

Pipeline 2: An LLM generates a fresh phi_hat for each FOLIO instance; prompting strategies
B0–B4 judge whether the generated formula is correct.
Ground-truth labels are stored in each results.jsonl under 'ground_truth_label'.

GT label values:
    'yes'                  — generated formula is equivalent to the gold FOL
    'no'                   — generated formula is not equivalent
    '?'                    — instance is ambiguous (gold itself is uncertain)
    'syntactically_incorrect' — generated formula cannot be parsed by Z3

Directory structure under --results-dir:
    {baseline}/
        {dataset}/
            {judge_model}/
                results.jsonl   (all phi_model x pv x n_examples combos, as row fields)

Table columns (per run row):
    gt_yes  gt_no  gt_?
    pred_yes  pred_no  pred_?
    n_correct_yes  n_correct_no  n_correct_?  n_correct_refinement
    safety_yes  n_fully_correct

Each table is printed/saved twice: sorted by safety_yes ↓, then by n_fully_correct ↓.
Ensemble tables (pair_safe, pair_useful, triple_safe, triple_majority) are also produced.
For ensemble computation, only the top --top-per-baseline variants per baseline are
included in the pool (default 10, B0 excluded), giving C(60,3) = 34,220 triples instead
of millions.

Usage:
    python pipeline2/evaluate.py
    python pipeline2/evaluate.py --datasets folio malls
    python pipeline2/evaluate.py --baselines B0 B1 B2
    python pipeline2/evaluate.py --no-ensemble
    python pipeline2/evaluate.py --top-per-baseline 10   # default
    python pipeline2/evaluate.py --top-per-baseline 0    # use all variants (original behaviour)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_BASELINE_RESULTS_DIR = _ROOT / "pipeline2" / "assessing_phi_hat" / "results"
_OUT_DIR = _ROOT / "pipeline2" / "assessing_phi_hat" / "results" / "tables"


def _table_path(dataset_name: str, kind: str) -> Path:
    """Route table outputs into per-dataset subfolders.

    kind ∈ {"per_baseline", "ensembles"}.
    """
    for ds in ("folio", "malls"):
        if dataset_name.startswith(ds):
            return _OUT_DIR / ds / kind
    return _OUT_DIR / kind

_POSITIVE_GT  = {"yes"}
_NEGATIVE_GT  = {"no", "syntactically_incorrect"}
_AMBIGUOUS_GT = {"?"}
_VALID_PREDS  = {"yes", "no", "?"}
_VALID_GT     = _POSITIVE_GT | _NEGATIVE_GT | _AMBIGUOUS_GT


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: Path) -> List[dict]:
    """Load results.jsonl from a run directory. Returns list of result dicts."""
    rfile = run_dir / "results.jsonl"
    if not rfile.exists():
        return []
    rows = []
    for line in rfile.open():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _get_iid(row: dict) -> str:
    """Extract instance ID — field name differs between B0 ('instance_id') and B1-B4 ('id')."""
    return row.get("instance_id") or row.get("id") or ""


def _normalize_iid(iid: str) -> str:
    """Normalize instance IDs so B0 and B1-B4 results can be joined for ensembles.

    Ids are stored bare (e.g. 'story_280', 'concl_203_136') in the current
    layout; this remains a passthrough for any legacy '__'-namespaced ids.
    """
    parts = iid.split("__")
    if len(parts) >= 2:
        return parts[1]  # second segment is always the FOLIO base ID
    return iid


def _variant_key(row: dict) -> str:
    """Build a display key for one (phi_model, pv[, n_examples]) variant within a model file."""
    hp = row.get("hyperparams") or {}
    pv = hp.get("prompt_version", "?")
    n_ex = hp.get("n_examples")
    phi_model = row.get("phi_model", "?")
    variant = f"v{pv}" + (f"_fs{n_ex}" if n_ex is not None else "")
    return f"{phi_model}_{variant}_B"


def _load_dataset_runs(dataset_dir: Path) -> Dict[str, List[dict]]:
    """Load all judge-model result files for a dataset directory and split them
    back into per-variant "runs" (one per (judge_model, phi_model, pv[, n_examples])
    combination), matching the granularity the rest of this script expects.

    Returns: {run_id: [result_dict, ...]}
    """
    runs: Dict[str, List[dict]] = defaultdict(list)
    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("__"):
            continue
        judge_model = model_dir.name
        rows = _load_run(model_dir)
        for row in rows:
            run_id = f"{judge_model}_{_variant_key(row)}"
            runs[run_id].append(row)
    return dict(runs)


def _build_variant_preds(
    runs: Dict[str, List[dict]]
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, str]]:
    """Build variant_preds and gt dicts from loaded runs.

    Returns:
        variant_preds: {full_run_id: {iid: {quality_score, refinement, eq_i}}}
        gt:            {iid: ground_truth_label}
    """
    variant_preds: Dict[str, Dict[str, dict]] = {}
    gt: Dict[str, str] = {}

    for run_id, rows in runs.items():
        vp: Dict[str, dict] = {}
        for r in rows:
            raw_iid = _get_iid(r)
            if not raw_iid:
                continue
            iid = _normalize_iid(raw_iid)   # canonical base ID for cross-run joining
            qs  = r.get("quality_score")
            ref = r.get("refinement")
            eq  = r.get("eq_i")
            gt_label = r.get("ground_truth_label", "")
            if gt_label in _VALID_GT:
                gt[iid] = gt_label
            if qs in _VALID_PREDS:
                vp[iid] = {"quality_score": qs, "refinement": ref, "eq_i": eq}
        if vp:
            variant_preds[run_id] = vp

    return variant_preds, gt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics_from_rows(rows: List[dict]) -> dict:
    """Compute metrics directly from a flat list of result records (individual run)."""
    gt_yes = gt_no = gt_qm = 0
    pred_yes = pred_no = pred_qm = 0
    n_correct_yes = n_correct_no = n_correct_qm = 0
    n_correct_refinement = 0
    n_correct_ref_no = 0
    n_total = 0

    for r in rows:
        gt  = r.get("ground_truth_label", "")
        qs  = r.get("quality_score")
        eq_i = r.get("eq_i")

        if qs not in _VALID_PREDS:
            continue
        n_total += 1

        if gt in _POSITIVE_GT:
            gt_yes += 1
        elif gt in _NEGATIVE_GT:
            gt_no += 1
        elif gt in _AMBIGUOUS_GT:
            gt_qm += 1

        if qs == "yes":
            pred_yes += 1
            if gt in _POSITIVE_GT:
                n_correct_yes += 1
        elif qs == "no":
            pred_no += 1
            if gt in _NEGATIVE_GT:
                n_correct_no += 1
        else:
            pred_qm += 1
            if gt in _AMBIGUOUS_GT:
                n_correct_qm += 1

        if eq_i is True:
            n_correct_refinement += 1
            if qs == "no" and gt in _NEGATIVE_GT:
                n_correct_ref_no += 1

    safety_yes = n_correct_yes / pred_yes if pred_yes > 0 else None
    n_fully_correct = n_correct_yes + n_correct_refinement

    return {
        "n_total":              n_total,
        "gt_yes":               gt_yes,
        "gt_no":                gt_no,
        "gt_?":                 gt_qm,
        "pred_yes":             pred_yes,
        "pred_no":              pred_no,
        "pred_?":               pred_qm,
        "n_correct_yes":        n_correct_yes,
        "n_correct_no":         n_correct_no,
        "n_correct_?":          n_correct_qm,
        "n_correct_refinement": n_correct_refinement,
        "n_correct_ref_no":     n_correct_ref_no,
        "safety_yes":           safety_yes,
        "n_fully_correct":      n_fully_correct,
    }


def _compute_metrics_from_preds(
    vp: Dict[str, dict],
    gt: Dict[str, str],
) -> dict:
    """Compute metrics from variant_preds dict for one variant (used in ensembles)."""
    gt_yes = gt_no = gt_qm = 0
    pred_yes = pred_no = pred_qm = 0
    n_correct_yes = n_correct_no = n_correct_qm = 0
    n_correct_refinement = 0
    n_correct_ref_no = 0
    n_total = 0

    for iid, p in vp.items():
        gt_label = gt.get(iid, "")
        if gt_label not in _VALID_GT:
            continue
        qs   = p.get("quality_score")
        eq_i = p.get("eq_i")
        if qs not in _VALID_PREDS:
            continue
        n_total += 1

        if gt_label in _POSITIVE_GT:
            gt_yes += 1
        elif gt_label in _NEGATIVE_GT:
            gt_no += 1
        elif gt_label in _AMBIGUOUS_GT:
            gt_qm += 1

        if qs == "yes":
            pred_yes += 1
            if gt_label in _POSITIVE_GT:
                n_correct_yes += 1
        elif qs == "no":
            pred_no += 1
            if gt_label in _NEGATIVE_GT:
                n_correct_no += 1
        else:
            pred_qm += 1
            if gt_label in _AMBIGUOUS_GT:
                n_correct_qm += 1

        if eq_i is True:
            n_correct_refinement += 1
            if qs == "no" and gt_label in _NEGATIVE_GT:
                n_correct_ref_no += 1

    safety_yes = n_correct_yes / pred_yes if pred_yes > 0 else None
    n_fully_correct = n_correct_yes + n_correct_refinement

    return {
        "n_total":              n_total,
        "gt_yes":               gt_yes,
        "gt_no":                gt_no,
        "gt_?":                 gt_qm,
        "pred_yes":             pred_yes,
        "pred_no":              pred_no,
        "pred_?":               pred_qm,
        "n_correct_yes":        n_correct_yes,
        "n_correct_no":         n_correct_no,
        "n_correct_?":          n_correct_qm,
        "n_correct_refinement": n_correct_refinement,
        "n_correct_ref_no":     n_correct_ref_no,
        "safety_yes":           safety_yes,
        "n_fully_correct":      n_fully_correct,
    }


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------

def _agg_verdict_safe(verdicts: List[str]) -> str:
    """All must agree on a non-? verdict; otherwise '?'."""
    unique = set(verdicts)
    return unique.pop() if len(unique) == 1 else "?"


def _agg_verdict_useful_pair(v1: str, v2: str) -> str:
    """Pair useful: one committed + one abstained → take committed; otherwise '?'."""
    if v1 == v2:
        return v1
    committed = [v for v in (v1, v2) if v != "?"]
    return committed[0] if len(committed) == 1 else "?"


def _agg_verdict_majority(verdicts: List[str]) -> str:
    """Strict majority ignoring abstentions; tie → '?'."""
    committed = [v for v in verdicts if v != "?"]
    if not committed:
        return "?"
    counts = Counter(committed)
    best = counts.most_common(1)[0][1]
    winners = [k for k, v in counts.items() if v == best]
    return winners[0] if len(winners) == 1 else "?"


def _apply_agg_verdict(verdicts: List[str], rule: str) -> str:
    if rule in ("pair_safe", "safe"):
        return _agg_verdict_safe(verdicts)
    if rule == "pair_useful":
        return _agg_verdict_useful_pair(verdicts[0], verdicts[1])
    if rule == "majority":
        return _agg_verdict_majority(verdicts)
    raise ValueError(f"Unknown rule: {rule}")


def _agg_refinement_safe(preds: List[dict]) -> Tuple[Optional[str], Optional[bool]]:
    """Safe: propose refinement only if ALL members proposed the same string AND eq_i=True."""
    non_null = [(p.get("refinement"), p.get("eq_i")) for p in preds if p.get("refinement")]
    if not non_null:
        return None, None
    refs = [r for r, _ in non_null]
    eq_is = [e for _, e in non_null]
    # All same string AND all eq_i=True
    if len(set(refs)) == 1 and all(e is True for e in eq_is):
        return refs[0], True
    return None, None


def _agg_refinement_useful(preds: List[dict]) -> Tuple[Optional[str], Optional[bool]]:
    """Useful: most-voted refinement string; eq_i=True if that string had eq_i=True in any member."""
    non_null = [(p.get("refinement"), p.get("eq_i")) for p in preds if p.get("refinement")]
    if not non_null:
        return None, None
    counts = Counter(r for r, _ in non_null)
    best_ref = counts.most_common(1)[0][0]
    # eq_i=True if any member with this refinement had eq_i=True
    eq_ok = any(e is True for r, e in non_null if r == best_ref)
    return best_ref, True if eq_ok else False


def _select_top_per_baseline(
    variant_preds: Dict[str, Dict[str, dict]],
    gt: Dict[str, str],
    top_k: int = 10,
    exclude_prefixes: Optional[set] = None,
) -> Dict[str, Dict[str, dict]]:
    """Keep the top_k variants per baseline (prefix before '/'), sorted by safety_yes desc.

    Baselines listed in exclude_prefixes are dropped entirely (default: B0).
    Returns a new dict with at most len(baselines) * top_k entries.
    """
    if exclude_prefixes is None:
        exclude_prefixes = {"B0"}

    # Group run_ids by their baseline prefix
    groups: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
    for run_id, vp in variant_preds.items():
        prefix = run_id.split("/")[0]
        if prefix in exclude_prefixes:
            continue
        m = _compute_metrics_from_preds(vp, gt)
        sy  = m["safety_yes"] if m["safety_yes"] is not None else -1.0
        nfc = m["n_fully_correct"]
        groups[prefix].append((run_id, sy, nfc))

    # For each baseline, sort and keep the top_k
    selected: Dict[str, Dict[str, dict]] = {}
    for prefix in sorted(groups):
        ranked = sorted(groups[prefix], key=lambda x: (x[1], x[2]), reverse=True)
        for run_id, _, _ in ranked[:top_k]:
            selected[run_id] = variant_preds[run_id]

    return selected


def compute_ensemble(
    variant_preds: Dict[str, Dict[str, dict]],
    gt: Dict[str, str],
    rule: str,
    n_components: int,
    top_k: int = 30,
) -> List[dict]:
    """Compute ensemble metrics for all n_components combinations of variants.

    Returns up to top_k rows sorted by safety_yes ↓ (further sorting is done by caller).
    """
    is_safe = rule in ("pair_safe", "safe")
    agg_ref = _agg_refinement_safe if is_safe else _agg_refinement_useful

    keys = sorted(variant_preds.keys())
    all_rows = []

    for combo in itertools.combinations(keys, n_components):
        all_iids: set = set()
        for k in combo:
            all_iids.update(variant_preds[k].keys())

        ensemble_vp: Dict[str, dict] = {}
        for iid in all_iids:
            members = [variant_preds[k].get(iid, {}) for k in combo]
            verdicts = [
                m.get("quality_score") if m.get("quality_score") in _VALID_PREDS else "?"
                for m in members
            ]
            agg_q = _apply_agg_verdict(verdicts, rule)
            agg_r, agg_eq = (None, None)
            if agg_q == "no":
                agg_r, agg_eq = agg_ref(members)
            ensemble_vp[iid] = {
                "quality_score": agg_q,
                "refinement":    agg_r,
                "eq_i":          agg_eq,
            }

        m = _compute_metrics_from_preds(ensemble_vp, gt)
        all_rows.append({
            "variants":              " + ".join(combo),
            "rule":                  rule,
            "gt_yes":                m["gt_yes"],
            "gt_no":                 m["gt_no"],
            "gt_?":                  m["gt_?"],
            "pred_yes":              m["pred_yes"],
            "pred_no":               m["pred_no"],
            "pred_?":                m["pred_?"],
            "n_correct_yes":         m["n_correct_yes"],
            "n_correct_no":          m["n_correct_no"],
            "n_correct_?":           m["n_correct_?"],
            "n_correct_refinement":  m["n_correct_refinement"],
            "n_correct_ref_no":      m["n_correct_ref_no"],
            "safety_yes":            m["safety_yes"],
            "n_fully_correct":       m["n_fully_correct"],
        })

    all_rows.sort(
        key=lambda r: (
            r["safety_yes"] if r["safety_yes"] is not None else -1.0,
            r["n_fully_correct"],
        ),
        reverse=True,
    )
    return all_rows[:top_k]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# Columns stored in each row dict (gt_* kept for CSV; excluded from display table)
_ALL_METRICS = [
    "gt_yes", "gt_no", "gt_?",
    "pred_yes", "pred_no", "pred_?",
    "n_correct_yes", "n_correct_no", "n_correct_?", "n_correct_refinement", "n_correct_ref_no",
    "safety_yes", "n_fully_correct",
]

# Columns shown in the fixed-width display table (gt_* shown in header, not repeated per row)
_TABLE_COLS = [
    "pred_yes", "pred_no", "pred_?",
    "n_correct_yes", "n_correct_no", "n_correct_?", "n_correct_refinement", "n_correct_ref_no",
    "safety_yes", "n_fully_correct",
]

# Columns written to CSV (full set, including gt_*)
_CSV_COLS = ["run_id", "gt_yes", "gt_no", "gt_?"] + _TABLE_COLS


def _pct(v: Optional[float]) -> str:
    return f"{v*100:.1f}%" if v is not None else "  —  "


def _gt_header(rows: List[dict]) -> str:
    """Return a one-line GT distribution summary from the row with most coverage."""
    if not rows:
        return "GT distribution: (no data)"
    # Pick the row that has the most instances (highest gt_yes + gt_no + gt_?)
    best = max(rows, key=lambda r: (r.get("gt_yes", 0) + r.get("gt_no", 0) + r.get("gt_?", 0)))
    gy = best.get("gt_yes", 0)
    gn = best.get("gt_no",  0)
    gq = best.get("gt_?",   0)
    return f"GT distribution (constant):  yes={gy}   no={gn}   ?={gq}   (N={gy+gn+gq})"


def _make_table_lines(rows_data: List[dict], name_col: str = "run_id") -> List[str]:
    """Render a list of metric dicts as fixed-width text lines (no gt_* columns)."""
    # Compute name column width dynamically so long ensemble IDs are not truncated
    name_width = max(
        (len(str(row.get(name_col, ""))) for row in rows_data),
        default=10,
    )
    name_width = max(name_width, len(name_col.upper()))

    fmt = (
        f"{{name:<{name_width}}}"
        "  {p_yes:>8}  {p_no:>6}  {p_q:>5}"
        "  {cy:>13}  {cn:>12}  {cq:>10}  {cref:>13}  {crn:>16}"
        "  {sy:>9}  {nfc:>13}"
    )
    hdr = fmt.format(
        name=name_col.upper(),
        p_yes="pred_yes", p_no="pred_no", p_q="pred_?",
        cy="n_correct_yes", cn="n_correct_no", cq="n_correct_?",
        cref="n_correct_ref", crn="n_correct_ref_no",
        sy="safety_yes", nfc="n_fully_correct",
    )
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for row in rows_data:
        name = str(row.get(name_col, ""))
        lines.append(fmt.format(
            name=name,
            p_yes=row.get("pred_yes", ""),  p_no=row.get("pred_no", ""),
            p_q=row.get("pred_?", ""),
            cy=row.get("n_correct_yes", ""), cn=row.get("n_correct_no", ""),
            cq=row.get("n_correct_?", ""),   cref=row.get("n_correct_refinement", ""),
            crn=row.get("n_correct_ref_no", ""),
            sy=_pct(row.get("safety_yes")),  nfc=row.get("n_fully_correct", ""),
        ))
    return lines


def _sort_rows(rows: List[dict], primary: str, secondary: str) -> List[dict]:
    def _key(r):
        v = r.get(primary)
        v2 = r.get(secondary, 0) or 0
        return (v if v is not None else -1.0, v2)
    return sorted(rows, key=_key, reverse=True)


def _write_txt_dual(
    rows: List[dict],
    out_path: Path,
    title: str,
    name_col: str = "run_id",
) -> None:
    """Write rows to a .txt file: GT header block, then two sorted sections."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep80 = "-" * 80

    rows_by_sy  = _sort_rows(rows, "safety_yes",     "n_fully_correct")
    rows_by_nfc = _sort_rows(rows, "n_fully_correct", "safety_yes")

    lines = [
        title,
        "=" * len(title),
        "",
        _gt_header(rows),   # GT distribution shown once here
        "",
        "[Sorted by safety_yes  ↓  (then n_fully_correct  ↓)]",
        sep80,
    ]
    lines += _make_table_lines(rows_by_sy, name_col)
    lines += [
        "",
        "[Sorted by n_fully_correct  ↓  (then safety_yes  ↓)]",
        sep80,
    ]
    lines += _make_table_lines(rows_by_nfc, name_col)
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(
    rows: List[dict],
    out_path: Path,
    name_col: str = "run_id",
) -> None:
    """Write CSV with two sections (sorted by safety_yes, then by n_fully_correct)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = _CSV_COLS
    rows_sy  = _sort_rows(rows, "safety_yes",     "n_fully_correct")
    rows_nfc = _sort_rows(rows, "n_fully_correct", "safety_yes")

    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        f.write("# --- sorted by safety_yes ---\n")
        for row in rows_sy:
            vals = [str(row.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")
        f.write("# --- sorted by n_fully_correct ---\n")
        for row in rows_nfc:
            vals = [str(row.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")


def _print_table_dual(rows: List[dict], title: str, top: int = 15,
                       name_col: str = "run_id") -> None:
    """Print table to stdout twice with different sort orders."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  {_gt_header(rows)}")
    print(sep)
    print(f"[Sorted by safety_yes  ↓]")
    for line in _make_table_lines(_sort_rows(rows, "safety_yes", "n_fully_correct")[:top], name_col):
        print(line)
    print(f"\n[Sorted by n_fully_correct  ↓]")
    for line in _make_table_lines(_sort_rows(rows, "n_fully_correct", "safety_yes")[:top], name_col):
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pipeline 2 baseline results.")
    parser.add_argument(
        "--results-dir", type=Path, default=_BASELINE_RESULTS_DIR,
        help="Root directory containing baseline result folders",
    )
    parser.add_argument(
        "--baselines", nargs="+", default=None,
        help="Specific baseline directories to include (e.g. B0 B1 B2)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Only evaluate datasets whose name contains any of these strings",
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Skip CSV output",
    )
    parser.add_argument(
        "--no-ensemble", action="store_true",
        help="Skip ensemble computation (much faster)",
    )
    parser.add_argument(
        "--ensemble-top-k", type=int, default=30,
        help="Max ensemble rows to keep per rule (default 30)",
    )
    parser.add_argument(
        "--top-per-baseline", type=int, default=10,
        help=(
            "For ensemble computation, keep only the top N variants per baseline "
            "sorted by safety_yes (default 10; set 0 to use all variants)"
        ),
    )
    parser.add_argument(
        "--ensemble-exclude", nargs="*", default=["B0"],
        help="Baselines to exclude from the ensemble pool (default: B0)",
    )
    args = parser.parse_args()

    results_root: Path = args.results_dir
    if not results_root.exists():
        print(f"ERROR: results directory not found: {results_root}")
        sys.exit(1)

    baseline_dirs = sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and not d.name.startswith("__") and d.name != "tables"
    )
    if args.baselines:
        baseline_dirs = [d for d in baseline_dirs if d.name in args.baselines]

    # Group dataset dirs by name across baselines (dataset dirs are now bare
    # dataset names, e.g. "folio" / "malls" — no phi_model/prompt in the name)
    by_dataset: Dict[str, List] = defaultdict(list)
    for baseline_dir in baseline_dirs:
        for ds_dir in sorted(baseline_dir.iterdir()):
            if not ds_dir.is_dir() or ds_dir.name.startswith("__"):
                continue
            if args.datasets and not any(sel in ds_dir.name for sel in args.datasets):
                continue
            by_dataset[ds_dir.name].append((baseline_dir.name, ds_dir))

    if not by_dataset:
        print(f"No dataset directories found under {results_root}")
        sys.exit(0)

    total_datasets = 0
    for dataset_name in sorted(by_dataset):
        # Collect all runs across baselines
        combined_runs: Dict[str, List[dict]] = {}
        for baseline_name, ds_dir in by_dataset[dataset_name]:
            for run_id, rows in _load_dataset_runs(ds_dir).items():
                combined_runs[f"{baseline_name}/{run_id}"] = rows

        if not combined_runs:
            print(f"  WARNING: no results found for {dataset_name}, skipping.")
            continue

        total_datasets += 1

        # GT distribution header
        first_rows = next(iter(combined_runs.values()))
        gt_dist = Counter(r.get("ground_truth_label") for r in first_rows)
        gt_yes = gt_dist.get("yes", 0)
        gt_no  = sum(gt_dist.get(k, 0) for k in ("no", "syntactically_incorrect"))
        gt_qm  = gt_dist.get("?", 0)
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")

        # ---- Per-run metrics ----
        run_rows = []
        for run_id, rows in sorted(combined_runs.items()):
            m = _compute_metrics_from_rows(rows)
            run_rows.append({"run_id": run_id, **{k: m[k] for k in _ALL_METRICS}})  # _ALL_METRICS now includes n_correct_ref_no

        title_baseline = f"Per-run results — {dataset_name}"
        _print_table_dual(run_rows, title_baseline)

        if not args.no_csv:
            out_csv = _table_path(dataset_name, "per_baseline") / f"{dataset_name}_runs.csv"
            _write_csv(run_rows, out_csv, name_col="run_id")
            print(f"  Wrote: {out_csv.relative_to(_ROOT)}")

        out_txt = _table_path(dataset_name, "per_baseline") / f"{dataset_name}_runs.txt"
        _write_txt_dual(run_rows, out_txt, title_baseline, name_col="run_id")
        print(f"  Wrote: {out_txt.relative_to(_ROOT)}")

        # ---- Ensemble analysis ----
        if args.no_ensemble:
            continue

        variant_preds, gt = _build_variant_preds(combined_runs)
        n_variants = len(variant_preds)
        if n_variants < 2:
            print(f"  Skipping ensemble: only {n_variants} variant(s).")
            continue

        # Build the ensemble pool: top --top-per-baseline per baseline, B0 excluded
        if args.top_per_baseline > 0:
            ensemble_pool = _select_top_per_baseline(
                variant_preds, gt,
                top_k=args.top_per_baseline,
                exclude_prefixes=set(args.ensemble_exclude),
            )
            n_pool = len(ensemble_pool)
            included_bases = sorted({r.split("/")[0] for r in ensemble_pool})
            n3 = n_pool * (n_pool - 1) * (n_pool - 2) // 6
            n2 = n_pool * (n_pool - 1) // 2
            print(
                f"\n  Ensemble pool: {n_pool} variants "
                f"(top {args.top_per_baseline} per baseline from {included_bases})\n"
                f"  Pair combos: C({n_pool},2) = {n2:,}   "
                f"Triple combos: C({n_pool},3) = {n3:,}"
            )
        else:
            ensemble_pool = variant_preds
            n_pool = n_variants
            n3 = n_pool * (n_pool - 1) * (n_pool - 2) // 6
            n2 = n_pool * (n_pool - 1) // 2
            print(
                f"\n  Ensemble pool: {n_pool} variants (all)\n"
                f"  Pair combos: C({n_pool},2) = {n2:,}   "
                f"Triple combos: C({n_pool},3) = {n3:,}"
            )

        ens_specs = [
            ("pair_safe",   2, "pairs_safe"),
            ("pair_useful", 2, "pairs_useful"),
            ("safe",        3, "triples_safe"),
            ("majority",    3, "triples_majority"),
        ]
        for rule, n_comp, label in ens_specs:
            if n_pool < n_comp:
                continue
            print(f"    Computing {label} ...")
            ens_rows = compute_ensemble(
                ensemble_pool, gt, rule, n_comp, top_k=args.ensemble_top_k
            )
            if not ens_rows:
                continue
            # Rename key for display
            for r in ens_rows:
                r["run_id"] = r.pop("variants")

            title_ens = f"Ensemble ({label}) — {dataset_name}"
            _print_table_dual(ens_rows, title_ens, top=10, name_col="run_id")

            if not args.no_csv:
                out_csv = _table_path(dataset_name, "ensembles") / f"{dataset_name}_ens_{label}.csv"
                _write_csv(ens_rows, out_csv, name_col="run_id")
                print(f"    Wrote: {out_csv.relative_to(_ROOT)}")

            out_txt = _table_path(dataset_name, "ensembles") / f"{dataset_name}_ens_{label}.txt"
            _write_txt_dual(ens_rows, out_txt, title_ens, name_col="run_id")
            print(f"    Wrote: {out_txt.relative_to(_ROOT)}")

    print(f"\nDone. Evaluated {total_datasets} dataset(s).")


if __name__ == "__main__":
    main()
