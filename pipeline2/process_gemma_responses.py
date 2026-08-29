"""process_gemma_responses.py — Process Gemma assessing_phi_hat responses into results.jsonl."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_RESP_DIR = _ROOT / "pipeline2" / "assessing_phi_hat" / "responses" / "gemma"
_PHI_HAT_DIR = _ROOT / "pipeline2" / "phi_hat_generation" / "results"
_RESULTS_DIR = _ROOT / "pipeline2" / "assessing_phi_hat" / "results"

_PHI_HAT_FILES = {
    "folio_gemma_P1": "folio__gemma-4-31B-it__P1.jsonl",
    "malls_gemma_P1": "malls__gemma-4-31B-it__P1.jsonl",
}


def load_phi_hat_source(filename: str) -> dict[str, dict]:
    path = _PHI_HAT_DIR / filename
    source = {}
    with open(path) as f:
        for line in f:
            try:
                inst = json.loads(line)
                source[inst["id"]] = inst
            except:
                pass
    return source


def find_balanced_json(text: str, start_idx: int = 0) -> tuple[str | None, int]:
    """Find a balanced JSON object starting from start_idx."""
    # Find first '{'
    i = text.find('{', start_idx)
    if i == -1:
        return None, -1

    depth = 0
    in_string = False
    escape = False
    for j in range(i, len(text)):
        c = text[j]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[i:j+1], j + 1
    return None, -1


def extract_quality_data(content: str) -> dict:
    """Extract quality_score, comment, reasoning, refinement from response content."""
    if not isinstance(content, str):
        if isinstance(content, dict):
            return {
                "quality_score": content.get("quality_score"),
                "comment": content.get("comment"),
                "reasoning": content.get("reasoning"),
                "refinement": content.get("refinement"),
            }
        return {"quality_score": None, "comment": None, "reasoning": None, "refinement": None}

    # Try to find JSON objects in the content (search from the end)
    # Look for patterns containing "quality_score"
    result = {"quality_score": None, "comment": None, "reasoning": None, "refinement": None}

    # Find all JSON-like structures with "quality_score"
    candidates = []
    idx = 0
    while idx < len(content):
        json_str, next_idx = find_balanced_json(content, idx)
        if json_str is None:
            break
        if '"quality_score"' in json_str:
            candidates.append(json_str)
        idx = next_idx

    # Try parsing each candidate
    for js in reversed(candidates):  # Prefer last (final answer)
        try:
            obj = json.loads(js)
            if isinstance(obj, dict) and "quality_score" in obj:
                result["quality_score"] = obj.get("quality_score")
                result["comment"] = obj.get("comment")
                result["reasoning"] = obj.get("reasoning")
                result["refinement"] = obj.get("refinement") or obj.get("proposed_refinement")
                return result
        except json.JSONDecodeError:
            # Try cleaning common issues
            cleaned = js.replace('\n', ' ').replace('\t', ' ')
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict) and "quality_score" in obj:
                    result["quality_score"] = obj.get("quality_score")
                    result["comment"] = obj.get("comment")
                    result["reasoning"] = obj.get("reasoning")
                    result["refinement"] = obj.get("refinement") or obj.get("proposed_refinement")
                    return result
            except:
                pass

    # Fallback: regex extraction of quality_score
    qs_match = re.search(r'"quality_score"\s*:\s*"?(yes|no|\?)"?', content, re.IGNORECASE)
    if qs_match:
        result["quality_score"] = qs_match.group(1).lower()

    # Try to extract refinement string (handles multi-line)
    ref_match = re.search(r'"(?:proposed_)?refinement"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    if ref_match:
        try:
            result["refinement"] = json.loads('"' + ref_match.group(1) + '"')
        except:
            result["refinement"] = ref_match.group(1)

    # Extract comment
    comment_match = re.search(r'"comment"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    if comment_match:
        try:
            result["comment"] = json.loads('"' + comment_match.group(1) + '"')
        except:
            result["comment"] = comment_match.group(1)

    return result


def compute_eq_i(refinement: str | None, phi_gold: str) -> bool | None:
    if not refinement or not phi_gold:
        return None
    try:
        from core.formalism.fol.fol import FOL
        from core.formalism.fol.solver_z3 import FOLSolver
        return bool(FOLSolver().are_equivalent(FOL(refinement), FOL(phi_gold)))
    except Exception:
        return None


def process_response_file(resp_file: Path, source_data: dict[str, dict]) -> tuple[str, list[dict]]:
    """Parse one response file. Returns (method, rows)."""
    name_parts = resp_file.stem.split("_")
    method = name_parts[-1]

    rows: list[dict] = []

    with open(resp_file) as f:
        for line in f:
            rec = json.loads(line)
            custom_id = rec["custom_id"]

            parts = custom_id.split("|")
            if len(parts) != 3:
                continue
            _, pv_str, instance_id = parts
            pv = int(pv_str.replace("pv", ""))

            source = source_data.get(instance_id, {})
            phi_gold = source.get("phi_gold", "")
            phi_hat = source.get("phi_hat", "")

            # Determine ground truth label
            ground_truth = source.get("label")
            if ground_truth is None:
                ambiguity = source.get("ambiguity_flag", False)
                if ambiguity:
                    ground_truth = "?"
                elif phi_hat and phi_gold:
                    eq = compute_eq_i(phi_hat, phi_gold)
                    if eq is True:
                        ground_truth = "yes"
                    elif eq is False:
                        ground_truth = "no"
                    else:
                        ground_truth = "syntactically_incorrect" if not phi_hat.strip() else "no"
                else:
                    ground_truth = "syntactically_incorrect"

            error = rec.get("error")
            extracted = {"quality_score": None, "comment": None, "reasoning": None, "refinement": None}

            if rec.get("response") and not error:
                response = rec["response"]
                if isinstance(response, dict) and "choices" in response:
                    msg = response["choices"][0].get("message", {})
                    content = msg.get("content", "")
                    extracted = extract_quality_data(content)

            quality_score = extracted["quality_score"]
            comment = extracted["comment"]
            reasoning = extracted["reasoning"] or comment
            refinement = extracted["refinement"]

            eq_i = None
            if refinement and phi_gold and not error:
                eq_i = compute_eq_i(refinement, phi_gold)

            row = {
                "id": instance_id,
                "method": method,
                "task": "B",
                "hyperparams": {"prompt_version": pv},
                "phi_model": "gemma",
                "phi_prompt": "P1",
                "ground_truth_label": ground_truth,
                "quality_score": quality_score,
                "reasoning": reasoning,
                "comment": comment,
                "refinement": refinement,
                "eq_i": eq_i,
                "logprobs": None,
                "logprobs_consistent": None,
                "error": error,
            }
            rows.append(row)

    return method, rows


def main():
    print("Loading phi_hat source data...")
    source_by_dataset = {}
    for dataset_name, phi_file in _PHI_HAT_FILES.items():
        source_by_dataset[dataset_name] = load_phi_hat_source(phi_file)
        print(f"  {dataset_name}: {len(source_by_dataset[dataset_name])} instances")

    print("\nProcessing response files...")
    total_rows = 0
    total_parsed = 0

    for resp_file in sorted(_RESP_DIR.glob("*.jsonl")):
        stem = resp_file.stem
        match = re.match(r"(folio|malls)_(B[1-4])$", stem)
        if not match:
            continue
        dataset, method_from_name = match.group(1), match.group(2)
        dataset_name = f"{dataset}_gemma_P1"

        if dataset_name not in source_by_dataset:
            continue

        source_data = source_by_dataset[dataset_name]
        method, rows = process_response_file(resp_file, source_data)

        out_dir = _RESULTS_DIR / method / dataset / "gemma"
        out_dir.mkdir(parents=True, exist_ok=True)
        n_file_parsed = 0
        with open(out_dir / "results.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
                if row.get("quality_score"):
                    n_file_parsed += 1
        n_file_total = len(rows)

        total_rows += n_file_total
        total_parsed += n_file_parsed
        parse_rate = n_file_parsed / n_file_total * 100 if n_file_total > 0 else 0
        print(f"  ✓ {resp_file.name}: {n_file_parsed}/{n_file_total} parsed ({parse_rate:.0f}%)")

    if total_rows:
        print(f"\n✓ Total: {total_parsed}/{total_rows} rows parsed ({total_parsed/total_rows*100:.1f}%)")
    else:
        print("\nNo response files found.")


if __name__ == "__main__":
    main()
