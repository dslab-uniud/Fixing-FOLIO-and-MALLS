"""LLM-as-a-Judge prompting strategies — self-contained implementation.

All prompt text is built inline from prompts/common.py fragments and
prompts/few_shot_examples.py examples.

Prompt versions
---------------
v1 — Neutral-1 (Direct): straightforward evaluation, lenient verdict threshold.
v2 — Safer-1 (Epistemic Humility): strict verdict threshold, default to '?'.
v3 — Neutral-2 (Structured Analysis): same lenient threshold as v1, but reasoning
     is framed as a systematic parse-then-compare process using section headers.
v4 — Safer-2 (Adversarial Flaw-Seeking): same strict threshold as v2, but the
     reasoning strategy is adversarial — assume φ̂ is wrong and try to falsify it
     before granting 'yes'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Type

from pydantic import BaseModel, model_validator

from core.llm.base import LLM
from .base import JudgeMethod
from .prompts.common import (
    task_description,
    verdict_instruction,
    refinement_instruction,
    cot_instruction,
    feature_instruction,
    output_format_instruction,
    feature_output_format_instruction,
    user_message,
)
from .prompts.few_shot_examples import build_few_shot_block, build_ontology_block, ONTOLOGY


# ---------------------------------------------------------------------------
# FOLIO per-instance ontology helper
# ---------------------------------------------------------------------------

def build_folio_ontology_block(id_source_dataset: str) -> str:
    """Build an ontology block string from the FOLIO signature for a given instance.

    Args:
        id_source_dataset: instance id as stored in the dataset (e.g. "story_380" or
                           "concl_380_1" — the story number is parsed automatically).

    Returns:
        A formatted string suitable for inclusion in a user message.
    """
    from core.dataset.signature_loader import (
        get_story_id_from_instance_id,
        load_folio_signature,
        format_folio_vocabulary,
    )
    story_id = get_story_id_from_instance_id(id_source_dataset)
    sig = load_folio_signature(story_id)
    rels_str, consts_str = format_folio_vocabulary(sig)
    lines = [
        "### Ontology",
        "",
        "Relations:",
        rels_str,
        "",
        "Constants:",
        consts_str,
    ]
    return "\n".join(lines)


def build_malls_ontology_block(instance_id: int) -> str:
    """Build an ontology block string from the MALLS per-instance signature.

    MALLS uses the same ``Rel``/``Const`` schema as FOLIO, so
    :func:`~core.dataset.signature_loader.format_folio_vocabulary` is reused.

    Args:
        instance_id: Integer MALLS instance id (1–100).

    Returns:
        A formatted string suitable for inclusion in a user message.
    """
    from core.dataset.signature_loader import (
        load_malls_signature,
        format_folio_vocabulary,
    )
    sig = load_malls_signature(instance_id)
    rels_str, consts_str = format_folio_vocabulary(sig)
    lines = [
        "### Ontology",
        "",
        "Relations:",
        rels_str,
        "",
        "Constants:",
        consts_str,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

_UNCERTAIN_KEYWORDS = ("[AMBIGUITY]", "[UNCERTAINTY]", "[OTHER]")


def _check_uncertain_comment(quality_score: str, comment: str) -> None:
    """Raise ValueError if quality_score is '?' but comment lacks the required keyword."""
    if quality_score == "?" and not any(comment.startswith(kw) for kw in _UNCERTAIN_KEYWORDS):
        raise ValueError(
            'When quality_score is "?", comment must start with one of: '
            + ", ".join(_UNCERTAIN_KEYWORDS)
        )


class VerdictOutput(BaseModel):
    """Structured output for Configuration A (reasoning + verdict + comment)."""
    reasoning: str
    quality_score: Literal["yes", "?", "no"]
    comment: str

    @model_validator(mode="after")
    def check_uncertain_comment(self) -> "VerdictOutput":
        _check_uncertain_comment(self.quality_score, self.comment)
        return self


class VerdictWithRefinement(BaseModel):
    """Structured output for Configuration B (reasoning + verdict + comment + refinement ψ)."""
    reasoning: str
    quality_score: Literal["yes", "?", "no"]
    comment: str
    refinement: str | None = None

    @model_validator(mode="after")
    def check_uncertain_comment(self) -> "VerdictWithRefinement":
        _check_uncertain_comment(self.quality_score, self.comment)
        return self


class _FeatureCheck(BaseModel):
    passed: bool
    justification: str


class FeaturizedOutput(BaseModel):
    """Structured output for B4 (feature-based decomposition).

    Configuration B adds an optional refinement ψ alongside the feature checks.
    """
    Syntactic_wellformedness: _FeatureCheck
    Ontological_faithfulness: _FeatureCheck
    Truth_Conditional_Equivalence: _FeatureCheck
    Ambiguity_Resolution: _FeatureCheck
    reasoning: str
    quality_score: Literal["yes", "?", "no"]
    comment: str
    refinement: str | None = None

    @model_validator(mode="after")
    def check_uncertain_comment(self) -> "FeaturizedOutput":
        _check_uncertain_comment(self.quality_score, self.comment)
        return self


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class _CallRecord:
    tokens_in: int | None
    tokens_out: int | None

    @property
    def tokens_total(self) -> int | None:
        if self.tokens_in is not None and self.tokens_out is not None:
            return self.tokens_in + self.tokens_out
        return None


@dataclass
class CostRecord:
    """Accumulates token-usage statistics across one evaluate() call."""
    _calls: list[_CallRecord] = field(default_factory=list)

    def add(self, tokens_in: int | None, tokens_out: int | None) -> None:
        self._calls.append(_CallRecord(tokens_in, tokens_out))

    def to_dict(self) -> dict:
        totals = [c.tokens_total for c in self._calls if c.tokens_total is not None]
        return {
            "n_calls": len(self._calls),
            "calls": [
                {
                    "tokens_in": c.tokens_in,
                    "tokens_out": c.tokens_out,
                    "tokens_total": c.tokens_total,
                }
                for c in self._calls
            ],
            "total_tokens_eff": sum(totals) if totals else None,
        }


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token (GPT-style heuristic)."""
    return max(1, len(text) // 4)


def _extract_usage(response: str | dict, prompt: str) -> tuple[int | None, int | None]:
    """Extract (tokens_in, tokens_out) from a response, or estimate if unavailable."""
    if isinstance(response, dict) and "usage" in response:
        usage = response["usage"]
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    tokens_in = _estimate_tokens(prompt)
    if isinstance(response, str):
        tokens_out = _estimate_tokens(response)
    elif isinstance(response, dict):
        tokens_out = _estimate_tokens(response.get("text", "") or "")
    else:
        tokens_out = None
    return tokens_in, tokens_out


# ---------------------------------------------------------------------------
# Logit-based decision rule
# ---------------------------------------------------------------------------

def apply_logit_rule(
    logprobs: list[dict],
    threshold_pos: float = 0.7,
    threshold_neg: float = 0.7,
) -> str | None:
    """Secondary logit-based verdict from token log-probabilities.

    Scans logprobs for the first 'yes', 'no', or '?' token and applies the
    threshold rule.  Returns None if no verdict token is found.
    """
    if not logprobs:
        return None
    for entry in logprobs:
        token = entry.get("token", "").strip().strip('"').lower()
        logprob = entry.get("logprob")
        if logprob is None or token not in ("yes", "no", "?"):
            continue
        prob = math.exp(logprob)
        if token == "yes":
            return "yes" if prob >= threshold_pos else "?"
        if token == "no":
            return "no" if prob >= threshold_neg else "?"
        return "?"
    return None


# ---------------------------------------------------------------------------
# Internal prompt-building helpers
# ---------------------------------------------------------------------------

def _common_version(prompt_version: int) -> int:
    """Map prompt_version to the common.py fragment version.

    Each prompt_version now has its own dedicated fragment set in common.py,
    so this is an identity function: 1→1, 2→2, 3→3, 4→4.
    """
    return prompt_version


def _assemble_prompt(
    parts: list[str],
    configuration: str,
    prompt_version: int,
    output_format: str = "",
) -> str:
    """Join non-empty parts with double newlines, append configuration suffix,
    then append the output format description."""
    body = "\n\n".join(p for p in parts if p)
    suffix = _config_suffix(configuration, prompt_version)
    result = body + ("\n\n" + suffix if suffix else "")
    if output_format:
        result += "\n\n" + output_format
    return result


def _verdict_block(prompt_version: int) -> str:
    """Return the verdict instruction for the given prompt_version."""
    return verdict_instruction(version=_common_version(prompt_version))


def _config_suffix(configuration: str, prompt_version: int) -> str:
    """Append refinement instruction for Configuration B."""
    if configuration != "B":
        return ""
    return refinement_instruction(version=_common_version(prompt_version))


# ---------------------------------------------------------------------------
# Per-method system prompt builders
# ---------------------------------------------------------------------------

def _b1_system_prompt(configuration: str, prompt_version: int) -> str:
    """B1 — Zero-Shot CoT.

    The ontology is NOT embedded here — it is provided per-input via the user turn.
    """
    v = _common_version(prompt_version)
    parts = [
        task_description(version=v),
        cot_instruction(version=v),
        _verdict_block(prompt_version),
    ]
    return _assemble_prompt(parts, configuration, prompt_version,
                            output_format=output_format_instruction(configuration))


def _b2_system_prompt(configuration: str, prompt_version: int, n_examples: int | None = None) -> str:
    """B2 — Few-Shot (no CoT).

    The demo ontology for the few-shot examples is embedded in the few-shot
    block (include_ontology=True).  The evaluation-specific ontology for each
    actual input is passed separately in the user turn.
    """
    technique = 2 if configuration == "B" else 1
    v = _common_version(prompt_version)
    parts = [
        task_description(version=v),
        build_few_shot_block(with_cot=False, technique=technique, include_ontology=True,
                             max_examples=n_examples),
        _verdict_block(prompt_version),
    ]
    return _assemble_prompt(parts, configuration, prompt_version,
                            output_format=output_format_instruction(configuration))


def _b3_system_prompt(configuration: str, prompt_version: int, n_examples: int | None = None) -> str:
    """B3 — Few-Shot + CoT.

    The demo ontology for the few-shot examples is embedded in the few-shot
    block (include_ontology=True).  The evaluation-specific ontology for each
    actual input is passed separately in the user turn.
    """
    technique = 2 if configuration == "B" else 1
    v = _common_version(prompt_version)
    parts = [
        task_description(version=v),
        cot_instruction(version=v),
        build_few_shot_block(with_cot=True, technique=technique, include_ontology=True,
                             max_examples=n_examples),
        _verdict_block(prompt_version),
    ]
    return _assemble_prompt(parts, configuration, prompt_version,
                            output_format=output_format_instruction(configuration))


def _b4_system_prompt(configuration: str, prompt_version: int) -> str:
    """B4 — Feature-Based Decomposition.

    The ontology is NOT embedded here — it is provided per-input via the user turn.
    """
    v = _common_version(prompt_version)
    parts = [
        task_description(version=v),
        feature_instruction(version=v),
        _verdict_block(prompt_version),
    ]
    return _assemble_prompt(parts, configuration, prompt_version,
                            output_format=feature_output_format_instruction(configuration))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMJudgeBase(JudgeMethod):
    """Abstract base for all LLM-as-a-Judge variants.

    Args:
        llm: LLM backend.
        configuration: 'A' (verdict + comment) or 'B' (verdict + comment + refinement ψ).
        prompt_version: which prompt phrasing to use (1, 2, or 3).
            v1 — Normal/direct.  v2 — Safe/high-certainty.
            v3 — Mixed (neutral analysis, strict verdict threshold).
        logprobs: request token log-probabilities from the LLM.
        logit_threshold_pos: P('yes') threshold for the logit decision rule.
        logit_threshold_neg: P('no') threshold for the logit decision rule.
    """

    def __init__(
        self,
        llm: LLM,
        configuration: str = "B",
        prompt_version: int = 1,
        logprobs: bool = False,
        logit_threshold_pos: float = 0.7,
        logit_threshold_neg: float = 0.7,
    ):
        if configuration not in ("A", "B"):
            raise ValueError(f"configuration must be 'A' or 'B', got {configuration!r}")
        if prompt_version not in (1, 2, 3, 4):
            raise ValueError(f"prompt_version must be 1, 2, 3, or 4, got {prompt_version!r}")
        self.llm = llm
        self.configuration = configuration
        self.prompt_version = prompt_version
        self.logprobs = logprobs
        self.logit_threshold_pos = logit_threshold_pos
        self.logit_threshold_neg = logit_threshold_neg

    @property
    def _output_schema(self) -> Type[BaseModel]:
        return VerdictWithRefinement if self.configuration == "B" else VerdictOutput

    def _build_system_prompt(self) -> str:
        raise NotImplementedError

    def _single_call(
        self,
        nl: str,
        formula: str,
        cost: CostRecord,
        ontology: dict | None = None,
    ) -> tuple[dict, list[dict] | None]:
        """One-shot LLM call; returns (structured_dict, logprobs_or_None).

        Args:
            ontology: evaluation-specific ontology dict.  When provided it is
                      formatted and prepended to the user message so the judge
                      sees the available predicates alongside the input sentence.
        """
        system = self._build_system_prompt()
        ontology_block = build_ontology_block(ontology) if ontology is not None else None
        v = _common_version(self.prompt_version)
        session = self.llm.chat(system_prompt=system)
        response = session.send(
            user_message(nl, formula, ontology_block=ontology_block, version=v),
            response_format=self._output_schema,
            logprobs=self.logprobs,
        )
        structured = response.get("structured", {}) if isinstance(response, dict) else {}
        raw_lp = response.get("logprobs") if isinstance(response, dict) else None
        cost.add(*_extract_usage(response, system))
        return structured, raw_lp

    def evaluate(
        self,
        nl: str,
        formula: str,
        llm=None,
        formalism=None,
        ontology: dict | None = None,
    ) -> dict:
        cost = CostRecord()
        structured, raw_lp = self._single_call(nl, formula, cost, ontology=ontology)
        result = {
            "quality_score": structured.get("quality_score"),
            "reasoning": structured.get("reasoning"),
            "comment": structured.get("comment"),
            "refinement": structured.get("refinement") if self.configuration == "B" else None,
            "cost": cost.to_dict(),
            "logit_verdict": None,
        }
        if raw_lp is not None:
            result["logit_verdict"] = apply_logit_rule(
                raw_lp, self.logit_threshold_pos, self.logit_threshold_neg
            )
            result["raw_logprobs"] = raw_lp
        return result


# ---------------------------------------------------------------------------
# B1 — Zero-Shot CoT
# ---------------------------------------------------------------------------

class B1_ZeroShotCoT(LLMJudgeBase):
    """Zero-Shot Chain-of-Thought prompting strategy.

    The model reasons step-by-step over (p, φ̂) before producing a verdict.
    No in-context examples are provided.
    """

    def _build_system_prompt(self) -> str:
        return _b1_system_prompt(self.configuration, self.prompt_version)


# ---------------------------------------------------------------------------
# B2 — Few-Shot (no CoT)
# ---------------------------------------------------------------------------

class B2_FewShot(LLMJudgeBase):
    """Few-Shot prompting strategy without chain-of-thought.

    K labeled (p, φ̂) → (q, comm) examples are prepended; no reasoning chain.
    """

    def _build_system_prompt(self) -> str:
        return _b2_system_prompt(self.configuration, self.prompt_version)


# ---------------------------------------------------------------------------
# B3 — Few-Shot + CoT
# ---------------------------------------------------------------------------

class B3_FewShotCoT(LLMJudgeBase):
    """Few-Shot with Chain-of-Thought.

    Combines in-context examples (including reasoning chains) with a CoT
    instruction.
    """

    def _build_system_prompt(self) -> str:
        return _b3_system_prompt(self.configuration, self.prompt_version)


# ---------------------------------------------------------------------------
# B4 — Feature-Based Decomposition
# ---------------------------------------------------------------------------

class B4_FeatureBased(LLMJudgeBase):
    """Feature-Based Decomposition prompting strategy.

    The model evaluates φ̂ against a checklist of sub-properties before
    producing an aggregated verdict. Aggregation rule: q='yes' iff all pass;
    q='no' if any fail with certainty; q='?' if any are indeterminate.
    """

    @property
    def _output_schema(self) -> Type[BaseModel]:
        return FeaturizedOutput

    def _build_system_prompt(self) -> str:
        return _b4_system_prompt(self.configuration, self.prompt_version)

    def evaluate(
        self,
        nl: str,
        formula: str,
        llm=None,
        formalism=None,
        ontology: dict | None = None,
    ) -> dict:
        cost = CostRecord()
        structured, raw_lp = self._single_call(nl, formula, cost, ontology=ontology)
        result = {
            "quality_score": structured.get("quality_score"),
            "reasoning": structured.get("reasoning"),
            "comment": structured.get("comment"),
            "features": {
                "Syntactic_wellformedness": structured.get("Syntactic_wellformedness"),
                "Ontological_faithfulness": structured.get("Ontological_faithfulness"),
                "Truth_Conditional_Equivalence": structured.get("Truth-Conditional_Equivalence"),
                "Ambiguity_Resolution": structured.get("Ambiguity_Resolution")
            },
            "refinement": structured.get("refinement") if self.configuration == "B" else None,
            "cost": cost.to_dict(),
            "logit_verdict": None,
        }
        if raw_lp is not None:
            result["logit_verdict"] = apply_logit_rule(
                raw_lp, self.logit_threshold_pos, self.logit_threshold_neg
            )
            result["raw_logprobs"] = raw_lp
        return result


# ---------------------------------------------------------------------------
# B0 — Trivial Baselines (no LLM calls)
# ---------------------------------------------------------------------------

class B0_AlwaysYes(JudgeMethod):
    """B0a: Always returns quality_score='yes'. Used as a floor reference."""

    def evaluate(
        self,
        nl: str,
        formula: str,
        llm=None,
        formalism=None,
        ontology=None,
    ) -> dict:
        return {
            "quality_score": "yes",
            "signal": 1.0,
            "comment": "Trivial baseline: always 'yes'.",
            "refinement": None,
            "logits": {},
            "cost": CostRecord().to_dict(),
        }


class B0_AlwaysNo(JudgeMethod):
    """B0b: Always returns quality_score='no'. Used as a floor reference."""

    def evaluate(
        self,
        nl: str,
        formula: str,
        llm=None,
        formalism=None,
        ontology=None,
    ) -> dict:
        return {
            "quality_score": "no",
            "signal": 1.0,
            "comment": "Trivial baseline: always 'no'.",
            "refinement": None,
            "logits": {},
            "cost": CostRecord().to_dict(),
        }


class B0_MajorityClass(JudgeMethod):
    """B0c: Returns the majority class label (from training split) for every instance.

    Args:
        majority_label: The majority class label, pre-computed from the training set.
    """

    def __init__(self, majority_label: str = "yes") -> None:
        if majority_label not in ("yes", "no"):
            raise ValueError(f"majority_label must be 'yes' or 'no', got {majority_label!r}")
        self.majority_label = majority_label

    def evaluate(
        self,
        nl: str,
        formula: str,
        llm=None,
        formalism=None,
        ontology=None,
    ) -> dict:
        return {
            "quality_score": self.majority_label,
            "signal": 1.0,
            "comment": f"Trivial baseline: majority class '{self.majority_label}'.",
            "refinement": None,
            "logits": {},
            "cost": CostRecord().to_dict(),
        }


class B0_Random(JudgeMethod):
    """B0d: Samples uniformly from {'yes', 'no'} with a fixed seed.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 0) -> None:
        import random
        self._rng = random.Random(seed)

    def evaluate(
        self,
        nl: str,
        formula: str,
        llm=None,
        formalism=None,
        ontology=None,
    ) -> dict:
        verdict = self._rng.choice(["yes", "no"])
        return {
            "quality_score": verdict,
            "signal": 0.5,
            "comment": "Trivial baseline: random uniform choice.",
            "refinement": None,
            "logits": {},
            "cost": CostRecord().to_dict(),
        }
