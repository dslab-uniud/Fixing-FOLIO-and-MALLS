"""Shared prompt fragments reused across all baseline variants.
Each fragment is versioned (1–4). Access via the getter functions at the
bottom of this module, e.g. ``task_description(version=3)``.

Version philosophy
------------------
V1 — Neutral-1 (Direct).
     Straightforward evaluation. The model judges φ̂ against p directly.
     Standard neutral baseline.

V2 — Safer-1 (Epistemic Humility).
     Prioritizes epistemic humility. "yes"/"no" only with absolute certainty.
     Any ambiguity, doubt, or edge case defaults to "?".

V3 — Neutral-2 (Structured Analysis).
     Same lenient verdict threshold as V1, but reasoning is structured into
     explicit numbered steps with ### section headers. Separates decomposition
     of p from decomposition of φ̂ from comparison, making the analysis
     reproducible and systematic.

V4 — Safer-2 (Structured Analysis).
     Priopritize "?" in case of doubt (as in V2) but the resoning is structured as in V3.
"""

# ---------------------------------------------------------------------------
# TASK_DESCRIPTION
# ---------------------------------------------------------------------------
_TASK_DESCRIPTION: dict[int, str] = {
    1: (
        "You are an expert linguist and logician evaluating whether a candidate First-Order Logic (FOL) formula φ̂  correctly and completely formalizes a natural language sentence p using only the symbols of a given ontology.\n"
    ),
    2: (
        "You are an expert linguist and logician evaluating whether a candidate FOL formula φ̂ against a natural language sentence p in a given ontology. Your primary directive is to avoid false confidence: therefore, if you are not sure about your assessment, please refrainf from judging the formula correct ('yes') or incorrect ('no') and use the '?' symbol.\n "
    ),
    3: (
        "You are an expert linguist and logician evaluating whether a candidate FOL formula φ̂ against a natural language sentence p in a given ontology. "
        "Your task: determine whether a candidate FOL formula φ̂ correctly and completely formalizes a natural language sentence p.\n\n"
        "### Approach\n"
        "Work through your evaluation in explicit, ordered steps before issuing a verdict. "
        "Your reasoning should be reproducible: a second expert reading your steps should reach the same conclusion.\n"
    ),
    4: (
        
        "You are a rigorous proof-checker specializing in finding errors in candidate FOL formalizations. "
        "Your default stance is skeptical: assume φ̂ is incorrect until your adversarial analysis fails to find any flaw.\n\n"
        "### Protocol\n"
        "1. Systematically probe for the most common error types: wrong quantifier (∀ vs ∃), scope inversion, missing condition, predicate mismatch.\n"
        "2. Only if the attempts fail to find a flaw you should consider 'yes'.\n\n"
    ),
}

# ---------------------------------------------------------------------------
# VERDICT_INSTRUCTION
# ---------------------------------------------------------------------------
_VERDICT_INSTRUCTION: dict[int, str] = {
    1: (
        "Assign a quality verdict q ∈ {'yes', '?', 'no'}:\n"
        "  'yes' — φ̂ correctly and completely captures the semantic meaning of p within the provided ontology as a valid FOL formula. Note that the chosen formalism and ontology restrictions limit expressiveness: there will be cases in which not all the information in the NL sentence can be retained when converting it into a FOL formula using the given symbols. "
        "In such cases, the formalization should nonetheless be considered correct if φ̂ is the best possible approximation of p in FOL given the provided ontology.\n"
        "  'no'  — φ̂  is incorrect or incomplete due to syntax errors, semantic hallucination, ontology mismatch, or logical flaws.  The only symbols permitted are the logical ones (∀ ∃ ¬ ∧ ∨ → ↔ ⊕ = ≠ True False T ⊥), and those explicitly defined in the provided ontology.\n"
        "  '?'   — A confident and definitive verdict cannot be assigned because the NL sentence admits multiple non-equivalent interpretations, or because there is uncertainty about its logical correctness, or for other reasons. In this case, your comment MUST begin with exactly one of the following tags: [AMBIGUITY], [UNCERTAINTY], or [OTHER]."
    ),
    2: (
        "Assign a quality verdict q ∈ {'yes', '?', 'no'} with absolute certainty:\n"
        "  'yes' — Only if you are completely confident that φ̂ correctly and completely captures the semantic meaning of p within the provided ontology. Any doubt about expressiveness, interpretation, or logical equivalence should default to '?'.\n"
        "  'no'  — Only if you are certain that φ̂ contains a definite error: syntax errors, semantic hallucination, ontology violation, or logical flaws.\n"
        "  '?'   — Default when uncertain. If the NL sentence admits multiple interpretations, or if there is any doubt about logical correctness, use '?'. Your comment MUST begin with exactly one of the following tags: [AMBIGUITY], [UNCERTAINTY], or [OTHER]."
    ),
    3: (
        "Assign a quality verdict q ∈ {'yes', '?', 'no'}:\n"
        "  'yes' — φ̂ correctly and completely captures the semantic meaning of p within the provided ontology as a valid FOL formula. Note that the chosen formalism and ontology restrictions limit expressiveness: there will be cases in which not all the information in the NL sentence can be retained when converting it into a FOL formula using the given symbols. "
        "In such cases, the formalization should nonetheless be considered correct if φ̂ is the best possible approximation of p in FOL given the provided ontology.\n"
        "  'no'  — φ̂  is incorrect or incomplete due to syntax errors, semantic hallucination, ontology mismatch, or logical flaws.  The only symbols permitted are the logical ones (∀ ∃ ¬ ∧ ∨ → ↔ ⊕ = ≠ True False T ⊥), and those explicitly defined in the provided ontology.\n"
        "  '?'   — A confident and definitive verdict cannot be assigned because the NL sentence admits multiple non-equivalent interpretations, or because there is uncertainty about its logical correctness, or for other reasons. In this case, your comment MUST begin with exactly one of the following tags: [AMBIGUITY], [UNCERTAINTY], or [OTHER]."
    ),
    4: (
        "Assign a quality verdict q ∈ {'yes', '?', 'no'} with rigor:\n"
        "  'yes' — Only after a rigorous adversarial search for flaws. Confirm correctness only when no flaw is found.\n"
        "  'no'  — When a definite flaw is discovered: syntax error, undefined predicate, scope violation, or logical inconsistency.\n"
        "  '?'   — When unable to conclusively rule out flaws or when multiple interpretations exist. Your comment MUST begin with exactly one of the following tags: [AMBIGUITY], [UNCERTAINTY], or [OTHER]."
    ),
}

# ---------------------------------------------------------------------------
# REFINEMENT_INSTRUCTION
# ---------------------------------------------------------------------------
_REFINEMENT_INSTRUCTION: dict[int, str] = {
    1: (
        'When q is "no" or "?", also produce a corrected formalization ψ '
        "that addresses the identified issues. ψ should be a rigorous, improved version of φ̂  "
        "that resolves the specific problems you found. Ensure ψ is "
        "a complete well-formed FOL formula, not a partial suggestion; when p is ambiguous, ψ should represent the other possible interpretation of p. "
        'If q is "yes", put "null" for ψ.'
    ),
    2: (
        'When q is "no" or "?", you must provide a safe, rigorously defined '
        "alternative formalization ψ. Ensure ψ resolves the ambiguity or error "
        "found in φ̂ (in the ambiguity case, ψ should formalize a different interpretation of p). ψ must be a complete, well-formed FOL formula. "
        'If q is "yes", put "null" for ψ. '
        'If you cannot reliably produce a safe ψ, put "null" and explain in the comment why refinement is not possible.'
    ),
    3: (
        "### Refinement\n"
        'When q is "no" or "?", produce a corrected formalization ψ that resolves the specific '
        "issue identified in your analysis steps (it is a better formalization if φ̂  is considered incorrect, or it is a formalization of a different interpretation of p if it is ambiguous). ψ must be a complete, well-formed FOL formula — "
        "not a partial suggestion. Address the root cause identified in your step-by-step analysis, not just the symptom. "
        'If q is "yes", put "null" for ψ.'
    ),
    4: (
        "### Refinement\n"
        'When q is "no" or "?", produce ψ: a corrected FOL formula that eliminates the specific flaw(s) found by your probes. '
        "ψ must be a complete, well-formed FOL formula. "
        'If the flaw stems from a genuine ambiguity ([AMBIGUITY]), choose ψ to be one reading of p distinct from φ̂ . '
        'If q is "yes", put "null" for ψ.'
    ),
}

# ---------------------------------------------------------------------------
# COT_INSTRUCTION
# ---------------------------------------------------------------------------
_COT_INSTRUCTION: dict[int, str] = {
    1: (
        "Before producing the verdict, reason step by step."
    ),
    2: (
        "Before producing the verdict, reason step by step. "
        "You must be absolutely certain about every aspect of the syntax and semantics; if there is any doubt the verdict must be '?'."
    ),
    3: (
        "Reason about the flaws and the strength of the formalization and, then, produce the final evaluation."
    ),
    4: (
        "Before producing the verdict, reason adversarially: actively attempt to find a flaw in φ̂ rather than confirming it. "
        "Only after an adversarial search that finds no flaw you should consider 'yes'."
    ),
}

# ---------------------------------------------------------------------------
# FEATURE_INSTRUCTION
# ---------------------------------------------------------------------------
_FEATURE_INSTRUCTION: dict[int, str] = {
    1: (
        "Evaluate φ̂ step-by-step against this logical and semantic checklist:\n\n"
        "  1. Syntactic Well-Formedness: Is φ̂ a mathematically legal FOL formula? Check for balanced parentheses, correct arity usage (included the impossibility to use a relational symbol as an argument of a relational symbol), and ensure all variables are properly bound by quantifiers. "
        "  2. Ontological Faithfulness: Do the predicates and constants (and their intended meanings) perfectly align with the provided ontology? Ensure no new predicates or constants are created, and lexical boundaries are respected. \n"
        "  3. Truth-Conditional Equivalence: Given the limitations posed by the formalism used and by the ontology given, does φ̂  convey the same meaning of p\n"
        "  4. Ambiguity Resolution: If p is ambiguous and supports multiple valid interpretations, or if you are uncertain about the intended reading, you must flag it as indeterminate.\n"
        "For each sub-property, provide a binary Pass/Fail/Uncertain judgment with a concise logical justification. Then, aggregate the results in a final evaluation.\n"
    ),
    2: (
        "Evaluate φ̂  against this checklist. Any possible source of error in the formalization should be flagged:\n\n"
        "  0. Semantic Autopsy (Pre-step): Strip away all syntactic sugar from p. Isolate the exact, undeniable logical constraints. If the core meaning is blurry, note it immediately.\n"
        "  1. Syntactic Well-Formedness: Is the formula completely devoid of syntax errors, dangling variables, or arity mismatches? (Pass/Fail/Uncertain)\n"
        "  2. Ontological Faithfulness: Does φ̂  use only and in the correct way the symbols provided in the ontology? (Pass/Fail/Uncertain)\n"
        "  3. Truth-Conditional Equivalence: Does φ̂  formalize in the best way p, giving the formalism used and the symbols the formula should adhere to? Consider in this phase quantifiers swaps and scopes, and boolean operators swaps."
        "  4. Ambiguity Resolution: If there is a valid alternative reading of p, or if you are uncertain about its meaning, you MUST fail this check and force an uncertain verdict. (Pass/Fail/Uncertain)\n"
        "For each sub-property, provide a binary Pass/Fail/Uncertain judgment with a concise logical justification. Then, aggregate the results in a final evaluation.\n"
    ),
    3: (
        "### Structured Checklist\n"
        "Evaluate φ̂  against the following criteria in sequence. "
        "For each, state Pass / Fail / Uncertain and provide a one-sentence justification.\n\n"
        "  Step 0 — Semantic decomposition (pre-step): identify the entities, predicates, quantifiers, and connectives that p asserts before examining φ̂.\n"
        "  Check 1 — Syntactic well-formedness: Is φ̂ a syntactically legal FOL formula? (Balanced parentheses, correct predicate arity, all variables bound, no nested relational symbols.)\n"
        "  Check 2 — Ontological faithfulness: Do all predicates and constants in φ̂ appear in the provided ontology with the correct arity? Do they appear in the formula with the intended meaning? (No hallucinated predicates.)\n"
        "  Check 3 — Truth-Conditional Equivalence: Does the quantifier structure of φ̂  match the most natural and defensible reading of p? (∀/∃ choice; nesting order relative to conditions).  Do, in general, p and φ̂ describe exactly the same set of situations (same models across all possible domains)? Evaluate also the completeness: Does φ̂ capture every semantic condition explicit in p, without adding or removing content?\n"
        "  Check 4 — Ambiguity Resolution: Does p have different meanings? Are they all equivalent to what expressed in φ̂ ?\n\n"
        "### Aggregation\n"
        "Exploiting the information above, and maybe other remarks that are not encompassed by the above checklist, aggregate the scores above to obtain a final evaluation."
    ),
    4: (
        "### Adversarial Checklist\n"
        "Apply each probe below to actively search for a flaw in φ̂. Report the outcome of each probe.\n\n"
        "  Probe 1 — Syntactic well-formedness: Is φ̂ a syntactically legal FOL formula? (Balanced parentheses, correct predicate arity, all variables bound, no nested relational symbols."
        "  Probe 2 — Ontological faithfulness: Does φ̂ use any predicate absent from the provided ontology, or a predicate with incorrect arity? Have the predicates used the intended meaning in the formula? (Violation → Fail / Clean → Pass) "
        "  Probe 3 — Truth-Conditional Equivalence: Is φ̂  an incorrect formalization of p whithin the formalism and the ontology chosen? Swapping a quantifier/boolean operator leads to a better formalization? Is there a semantic component of p with no corresponding element in φ̂ ? (Yes → Fail / No → Pass) "
        "  Probe 4 — Ambiguity Resolution: Is there another meaningful interpretation of p beyond the one given in φ̂ ? (Yes → Fail / No → Pass).\n\n "
        "### Aggregation\n"
        "Using what discussed above, and maybe other remarks that are not encompassed in the probes, aggregate the scores to obtain a final evaluation."
    ),
}

# ---------------------------------------------------------------------------
# Getter functions
# ---------------------------------------------------------------------------
# Prompt versions available for all fragment getters (task_description, verdict_instruction, etc.)
# Note: user_message() additionally accepts version=3/4 for DSPy-style formatting; those
# are NOT prompt content versions — they only change delimiter style.
VERSIONS = (1, 2, 3, 4)

def _get(d: dict[int, str], version: int, name: str) -> str:
    if version not in d:
        raise ValueError(f"{name} has no version {version}; available: {sorted(d)}")
    return d[version]

def task_description(version: int = 1) -> str:
    return _get(_TASK_DESCRIPTION, version, "task_description")

def verdict_instruction(version: int = 1) -> str:
    return _get(_VERDICT_INSTRUCTION, version, "verdict_instruction")

def refinement_instruction(version: int = 1) -> str:
    return _get(_REFINEMENT_INSTRUCTION, version, "refinement_instruction")

def cot_instruction(version: int = 1) -> str:
    return _get(_COT_INSTRUCTION, version, "cot_instruction")

def feature_instruction(version: int = 1) -> str:
    return _get(_FEATURE_INSTRUCTION, version, "feature_instruction")

def user_message(
    nl: str,
    formula: str,
    ontology_block: str | None = None,
    version: int = 1,
) -> str:
    """Build the user-turn message for an evaluation request.

    Args:
        nl:             Natural language sentence p.
        formula:        Candidate formula φ̂.
        ontology_block: Pre-formatted ontology string (from build_ontology_block()).
                        When provided it is prepended to the message so the judge
                        sees the available predicates alongside the input.
                        This is the standard way to pass ontology context — it
                        belongs in the user turn, not in the system prompt.
        version:        Formatting style (1/2: standard; 3/4: DSPy-style with
                        delimiters).  Versions 3/4 are only used by DSPy
                        optimisation passes.
    """
    prefix = f"{ontology_block}\n\n" if ontology_block else ""
    if version in (3, 4):
        return f"{prefix}---\nSentence (p): {nl}\nCandidate Formula (φ̂): {formula}\n---"
    return f"{prefix}Sentence: {nl}\nFormalization: {formula}"


# ---------------------------------------------------------------------------
# OUTPUT_FORMAT_INSTRUCTION
# ---------------------------------------------------------------------------

def output_format_instruction(configuration: str) -> str:
    """Return the JSON output format description for the standard verdict schema.

    Configuration A: reasoning + quality_score + comment.
    Configuration B: reasoning + quality_score + comment + refinement.
    """
    lines = [
        "### Output Format",
        "",
        "Produce a JSON object with exactly these fields:",
        '  "reasoning"     : string  — step-by-step analysis and reasoning',
        '  "quality_score" : "yes" | "?" | "no"',
        '  "comment"       : string  — explanation of the verdict',
    ]
    if configuration == "B":
        lines.append(
            '  "refinement"    : string | null'
            '  — corrected FOL formula when quality_score ≠ "yes"; null otherwise'
        )
    return "\n".join(lines)


def feature_output_format_instruction(configuration: str = "A") -> str:
    """Return the JSON output format description for the feature-based schema (B4).

    Configuration B adds an optional 'refinement' field matching the other methods.
    """
    lines = [
        "### Output Format",
        "",
        "Produce a JSON object with exactly these fields:",
        '  "Syntactic Well-Formedness"            : {"passed": bool, "justification": string}',
        '  "Ontological faithfulness"     : {"passed": bool, "justification": string}',
        '  "Truth-Conditional Equivalence"       : {"passed": bool, "justification": string}',
        '  "Ambiguity Resolution"            : {"passed": bool, "justification": string}',
        '  "reasoning"              : string  — step-by-step feature analysis',
        '  "quality_score"          : "yes" | "?" | "no"',
        '  "comment"                : string  — overall summary of the evaluation',
    ]
    if configuration == "B":
        lines.append(
            '  "refinement"    : string | null'
            '  — corrected FOL formula when quality_score ≠ "yes"; null otherwise'
        )
    return "\n".join(lines)