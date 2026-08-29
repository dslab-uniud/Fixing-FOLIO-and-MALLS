"""extract_answers.py — Parse raw curation response files into clean prediction records.

Reads:
    responses/openai/{dataset}_{method}.jsonl
    responses/qwen/{dataset}_{method}.jsonl
    responses/gemma/{dataset}[_concl|_story]_{method}.jsonl

Writes:
    results/predictions/{method}/{folio|malls}/{model}/results.jsonl
    (FOLIO's concl/story response files are merged into one results.jsonl)

One record per API call with fields:
    custom_id, instance_id, method, task, pv, llm,
    quality_score, comment, reasoning, logits, error

Usage:
    python pipeline1/extract_answers.py
    python pipeline1/extract_answers.py --llm openai qwen gemma
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RESP_DIR = _ROOT / "pipeline1" / "assessing_original" / "responses"
_ANS_DIR  = _ROOT / "pipeline1" / "assessing_original" / "results" / "predictions"

_VALID_LLM  = ("openai", "qwen", "gemma")
_VALID_Q    = ("yes", "no", "?")


# ---------------------------------------------------------------------------
# Custom-ID parsing
# ---------------------------------------------------------------------------

def _parse_custom_id(custom_id: str) -> dict:
    """Parse custom_id into structured fields.

    Current format:  {method}|pv{pv}|curation_{id}
    Legacy formats (still parsed for backward compatibility with historical
    response files): {method}|{task}|pv{pv}|curation_{id} and
    {method}|{task}|pv{pv}|curation_{split}_{id} (split in {selection, test, malls}).
    """
    parts = custom_id.split("|")
    method = parts[0]          # e.g. "B1", "B2"

    if len(parts) == 3:
        # current format: {method}|pv{pv}|curation_{id}
        pv_str, curation = parts[1], parts[2]
    else:
        # legacy format: {method}|{task}|pv{pv}|curation_[{split}_]{id}
        pv_str, curation = parts[2], parts[3]
    pv = int(pv_str[2:])

    ns    = None
    stage = 1

    # curation = "curation_{instance_id}" or (legacy) "curation_{split}_{instance_id}"
    m = re.match(r"curation_(selection|test|malls)_(.*)", curation)
    if m:
        instance_id = m.group(2)
    else:
        m = re.match(r"curation_(.*)", curation)
        if not m:
            raise ValueError(f"Cannot parse curation field: {curation!r}")
        instance_id = m.group(1)

    return {
        "method": method,
        "pv":     pv,
        "ns":     ns,
        "stage":  stage,
        "instance_id": instance_id,
    }


# ---------------------------------------------------------------------------
# Logprob extraction  (reuse logic from process_baseline_results.py)
# ---------------------------------------------------------------------------

def _extract_verdict_logprobs(raw_logprobs: list | None) -> dict:
    """Scan token stream to find quality_score logprobs for yes/no/?.

    Returns dict with keys 'yes', 'no', '?' (values: float | None).
    """
    empty = {"yes": None, "no": None, "?": None}

    if not raw_logprobs or not isinstance(raw_logprobs, list):
        return empty

    # Find the token that contains 'quality'
    quality_idx = None
    for i, entry in enumerate(raw_logprobs):
        tok = entry.get("token", "").strip().strip('"').lower()
        if tok == "quality":
            quality_idx = i
            break

    if quality_idx is None:
        return empty

    # Try offset +3, then +2 (tokeniser variants)
    for offset in (3, 2, 4):
        idx = quality_idx + offset
        if idx >= len(raw_logprobs):
            continue
        entry = raw_logprobs[idx]
        value_token = entry.get("token", "").strip().strip('"').lower()
        if value_token not in _VALID_Q:
            continue
        # Found the verdict token — collect logprobs
        lp_map: dict = {"yes": None, "no": None, "?": None}
        for lp_entry in entry.get("top_logprobs", []):
            tok = lp_entry.get("token", "").strip().strip('"').lower()
            lp  = lp_entry.get("logprob")
            if tok in lp_map and lp_map[tok] is None:
                lp_map[tok] = lp
        return lp_map

    return empty


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_openai_record(raw: dict) -> dict:
    """Parse one OpenAI Batch API response record."""
    custom_id  = raw.get("custom_id", "")
    resp       = raw.get("response", {})
    status     = resp.get("status_code", 0)
    error_msg  = None
    quality_score = None
    comment    = None
    reasoning  = None
    logits     = {"yes": None, "no": None, "?": None}

    refinement = None
    if status != 200:
        error_msg = f"HTTP {status}"
    else:
        body    = resp.get("body", {})
        choices = body.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            try:
                parsed = json.loads(content)
                quality_score = parsed.get("quality_score")
                comment       = parsed.get("comment")
                reasoning     = parsed.get("reasoning")
                refinement    = parsed.get("refinement") or None
            except (json.JSONDecodeError, AttributeError) as e:
                error_msg = f"JSON parse error: {e}"

        # Logprobs
        raw_lps = body.get("logprobs", {}).get("content")
        logits  = _extract_verdict_logprobs(raw_lps)

    if quality_score not in _VALID_Q:
        if error_msg is None:
            error_msg = f"invalid quality_score: {quality_score!r}"
        quality_score = None

    return {
        "custom_id":     custom_id,
        "quality_score": quality_score,
        "comment":       comment,
        "reasoning":     reasoning,
        "refinement":    refinement,
        "logits":        logits,
        "error":         error_msg,
    }


def _parse_qwen_record(raw: dict) -> dict:
    """Parse one Qwen vLLM response record.

    Handles two formats:
    - Nested OpenAI-compatible: raw["response"]["status_code"] + raw["response"]["body"]
      (produced by run_qwen_malls.py)
    - Flat: raw["content"] + raw["finish_reason"] + raw["logprobs"]
      (produced by older Qwen scripts)
    """
    custom_id  = raw.get("custom_id", "")
    error_msg  = raw.get("error") or None
    quality_score = None
    comment    = None
    reasoning  = None
    logits     = {"yes": None, "no": None, "?": None}
    refinement = None

    # Detect format: OpenAI-compatible nested vs flat
    resp = raw.get("response")
    if isinstance(resp, dict) and "status_code" in resp:
        # OpenAI-compatible nested format (same as _parse_openai_record)
        status = resp.get("status_code", 0)
        if status != 200:
            error_msg = error_msg or f"HTTP {status}"
        else:
            body    = resp.get("body", {})
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content and not error_msg:
                    try:
                        parsed = json.loads(content)
                        quality_score = parsed.get("quality_score")
                        comment       = parsed.get("comment")
                        reasoning     = parsed.get("reasoning")
                        refinement    = parsed.get("refinement") or None
                    except (json.JSONDecodeError, AttributeError) as e:
                        error_msg = f"JSON parse error: {e}"
            raw_lps = body.get("logprobs", {}).get("content") if body.get("logprobs") else None
            if raw_lps:
                logits = _extract_verdict_logprobs(raw_lps)
    else:
        # Flat format
        finish_reason = raw.get("finish_reason", "stop")
        if finish_reason == "length":
            error_msg = "finish_reason=length (truncated)"

        content = raw.get("content", "")
        if content and not error_msg:
            try:
                parsed = json.loads(content)
                quality_score = parsed.get("quality_score")
                comment       = parsed.get("comment")
                reasoning     = parsed.get("reasoning")
                refinement    = parsed.get("refinement") or None
            except (json.JSONDecodeError, AttributeError) as e:
                error_msg = f"JSON parse error: {e}"

        # Logprobs (flat format only)
        raw_lps = (raw.get("logprobs") or {}).get("content")
        logits  = _extract_verdict_logprobs(raw_lps)

    if quality_score not in _VALID_Q:
        if error_msg is None:
            error_msg = f"invalid quality_score: {quality_score!r}"
        quality_score = None

    return {
        "custom_id":     custom_id,
        "quality_score": quality_score,
        "comment":       comment,
        "reasoning":     reasoning,
        "refinement":    refinement,
        "logits":        logits,
        "error":         error_msg,
    }


def _extract_json_from_gemma_content(content: str) -> dict:
    """Extract the JSON payload from Gemma's thinking preamble.

    Gemma outputs:  thought\\n*...reasoning...*\\n{JSON}
    We find the last '{' ... '}' block that parses as valid JSON.
    """
    # Try direct parse first (in case there's no preamble)
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find the last JSON object in the content
    last_brace = content.rfind("{")
    if last_brace == -1:
        raise ValueError("No JSON object found in content")
    try:
        return json.loads(content[last_brace:])
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: scan backwards for a valid JSON block
    for i in range(len(content) - 1, -1, -1):
        if content[i] == "}":
            for j in range(i, -1, -1):
                if content[j] == "{":
                    try:
                        return json.loads(content[j:i+1])
                    except (json.JSONDecodeError, ValueError):
                        continue
    raise ValueError("Cannot extract JSON from Gemma content")


def _parse_gemma_record(raw: dict) -> dict:
    """Parse one Gemma vLLM response record (OpenAI-compatible format with thinking preamble)."""
    custom_id  = raw.get("custom_id", "")
    resp       = raw.get("response") or {}
    error_msg  = raw.get("error") or None
    status     = resp.get("status_code", 0) if resp else 0

    quality_score = None
    comment    = None
    reasoning  = None
    logits     = {"yes": None, "no": None, "?": None}
    refinement = None

    if error_msg:
        pass  # keep error_msg as-is
    elif status != 200:
        error_msg = f"HTTP {status}"
    else:
        body    = resp.get("body", {})
        choices = body.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            try:
                parsed        = _extract_json_from_gemma_content(content)
                quality_score = parsed.get("quality_score")
                comment       = parsed.get("comment")
                reasoning     = parsed.get("reasoning")
                refinement    = parsed.get("refinement") or None
            except (ValueError, AttributeError) as e:
                error_msg = f"JSON extract error: {e}"

        # Logprobs (same structure as OpenAI)
        raw_lps = body.get("logprobs", {}).get("content") if body.get("logprobs") else None
        if raw_lps:
            logits = _extract_verdict_logprobs(raw_lps)

    if quality_score not in _VALID_Q:
        if error_msg is None:
            error_msg = f"invalid quality_score: {quality_score!r}"
        quality_score = None

    return {
        "custom_id":     custom_id,
        "quality_score": quality_score,
        "comment":       comment,
        "reasoning":     reasoning,
        "refinement":    refinement,
        "logits":        logits,
        "error":         error_msg,
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def _extract_one_file(
    resp_path: Path,
    llm_kind: str,
    llm_name: str,
) -> tuple[dict, dict]:
    """Extract one response file. Returns (custom_id -> record, stats dict)."""
    if llm_kind == "openai":
        parse_fn = _parse_openai_record
    elif llm_kind == "gemma":
        parse_fn = _parse_gemma_record
    else:
        parse_fn = _parse_qwen_record

    # Pass 1: collect all valid records, deduplicating by custom_id
    # (keep the last valid entry — response files may have multiple entries per id
    #  due to tunnel-drop retries)
    valid_records: dict[str, dict] = {}   # custom_id → record
    n_total = n_ok = n_err = 0

    with open(resp_path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                n_err += 1
                continue

            parsed = parse_fn(raw)
            custom_id = parsed["custom_id"]
            if not custom_id:
                n_err += 1
                continue

            try:
                meta = _parse_custom_id(custom_id)
            except (ValueError, IndexError):
                n_err += 1
                continue

            if parsed["error"] is not None or parsed["quality_score"] is None:
                n_err += 1
                # Still update if we don't have a valid entry yet for this id
                if custom_id not in valid_records:
                    pass  # skip invalid entries if a valid one exists or will exist
                continue

            record = {
                "custom_id":     custom_id,
                "instance_id":   meta["instance_id"],
                "method":        meta["method"],
                "task":          "B",
                "pv":            meta["pv"],
                "llm":           llm_name,
                "quality_score": parsed["quality_score"],
                "comment":       parsed["comment"],
                "reasoning":     parsed["reasoning"],
                "refinement":    parsed.get("refinement"),
                "logits":        parsed["logits"],
                "error":         parsed["error"],
            }
            if meta["ns"] is not None:
                record["ns"] = meta["ns"]
            if meta.get("stage", 1) == 2:
                record["stage"] = 2

            valid_records[custom_id] = record  # last valid wins
            n_ok += 1

    return valid_records, {"total": n_total, "ok": len(valid_records), "err": n_err}


_NAME_RE = re.compile(r"^(folio|malls)(?:_(?:concl|story))?_(B[1-4])\.jsonl$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract curation answers from raw responses.")
    parser.add_argument("--llm", nargs="+", choices=_VALID_LLM, default=list(_VALID_LLM))
    args = parser.parse_args()

    llm_names = {
        "openai": "gpt4o_mini",
        "qwen":   "qwen_local_30b",
        "gemma":  "gemma_4_31b",
    }

    total_ok = total_err = 0

    for llm_kind in args.llm:
        resp_dir = _RESP_DIR / llm_kind

        resp_files = sorted(resp_dir.glob("*.jsonl"))
        if not resp_files:
            print(f"[{llm_kind}] No response files found in {resp_dir}")
            continue

        print(f"\n[{llm_kind}] Processing {len(resp_files)} files ...")

        # Group response files by (dataset, method) — FOLIO's legacy concl/story
        # response files merge into a single output.
        groups: dict[tuple[str, str], list[Path]] = {}
        for resp_path in resp_files:
            m = _NAME_RE.match(resp_path.name)
            if not m:
                print(f"  SKIP (unrecognized filename): {resp_path.name}")
                continue
            dataset, method = m.group(1), m.group(2)
            groups.setdefault((dataset, method), []).append(resp_path)

        for (dataset, method), paths in sorted(groups.items()):
            merged: dict[str, dict] = {}
            n_total = n_err = 0
            for resp_path in paths:
                records, stats = _extract_one_file(
                    resp_path, llm_kind=llm_kind, llm_name=llm_names[llm_kind],
                )
                merged.update(records)
                n_total += stats["total"]
                n_err += stats["err"]

            out_path = _ANS_DIR / method / dataset / llm_kind / "results.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fout:
                for record in merged.values():
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "OK" if n_err == 0 else f"{n_err} errors"
            src = "+".join(p.name for p in paths)
            print(f"  {method}/{dataset} ← {src}: total={n_total:5d}  ok={len(merged):5d}  [{status}]")
            total_ok  += len(merged)
            total_err += n_err

    print(f"\nDone.  ok={total_ok}  errors={total_err}")


if __name__ == "__main__":
    main()
