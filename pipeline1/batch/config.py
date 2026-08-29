"""Shared configuration: dataset registry, response schemas, constants."""
from __future__ import annotations

from pathlib import Path

# Dataset registry — model-generated FOL formalizations (Pipeline 2: phi_hat_generation results)
_GENERATED_DIR = Path(__file__).parent.parent.parent / "pipeline2" / "phi_hat_generation" / "results"

GENERATOR_DATASETS: dict[str, Path] = {
    "folio_gpt4omini_P1": _GENERATED_DIR / "folio__gpt-4o-mini__P1.jsonl",
    "folio_qwen3_P1":     _GENERATED_DIR / "folio__Qwen3-30B-A3B__P1.jsonl",
}

# OpenAI response_format schemas (strict JSON schema mode)

SCHEMA_A = {
    "type": "json_schema",
    "json_schema": {
        "name": "VerdictOutput",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning":     {"type": "string"},
                "quality_score": {"type": "string", "enum": ["yes", "?", "no"]},
                "comment":       {"type": "string"},
            },
            "required":             ["reasoning", "quality_score", "comment"],
            "additionalProperties": False,
        },
    },
}

SCHEMA_B = {
    "type": "json_schema",
    "json_schema": {
        "name": "VerdictWithRefinement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning":     {"type": "string"},
                "quality_score": {"type": "string", "enum": ["yes", "?", "no"]},
                "comment":       {"type": "string"},
                "refinement":    {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required":             ["reasoning", "quality_score", "comment", "refinement"],
            "additionalProperties": False,
        },
    },
}

_FEATURE_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "passed":        {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required":             ["passed", "justification"],
    "additionalProperties": False,
}

SCHEMA_A_FEAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "FeaturizedOutput",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning":                     {"type": "string"},
                "Syntactic_wellformedness":       _FEATURE_CHECK_SCHEMA,
                "Ontological_faithfulness":       _FEATURE_CHECK_SCHEMA,
                "Truth_Conditional_Equivalence":  _FEATURE_CHECK_SCHEMA,
                "Ambiguity_Resolution":           _FEATURE_CHECK_SCHEMA,
                "quality_score":                  {"type": "string", "enum": ["yes", "?", "no"]},
                "comment":                        {"type": "string"},
            },
            "required": [
                "reasoning",
                "Syntactic_wellformedness", "Ontological_faithfulness",
                "Truth_Conditional_Equivalence", "Ambiguity_Resolution",
                "quality_score", "comment",
            ],
            "additionalProperties": False,
        },
    },
}

SCHEMA_B_FEAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "FeaturizedWithRefinement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning":                     {"type": "string"},
                "Syntactic_wellformedness":       _FEATURE_CHECK_SCHEMA,
                "Ontological_faithfulness":       _FEATURE_CHECK_SCHEMA,
                "Truth_Conditional_Equivalence":  _FEATURE_CHECK_SCHEMA,
                "Ambiguity_Resolution":           _FEATURE_CHECK_SCHEMA,
                "quality_score":                  {"type": "string", "enum": ["yes", "?", "no"]},
                "comment":                        {"type": "string"},
                "refinement":                     {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": [
                "reasoning",
                "Syntactic_wellformedness", "Ontological_faithfulness",
                "Truth_Conditional_Equivalence", "Ambiguity_Resolution",
                "quality_score", "comment", "refinement",
            ],
            "additionalProperties": False,
        },
    },
}


def get_schema_for_method(method_name: str) -> dict:
    """Get the JSON schema for a given method."""
    if method_name == "B4":
        return SCHEMA_B_FEAT
    return SCHEMA_B


def get_schema_for_task() -> dict:
    """Get the base response schema."""
    return SCHEMA_B
