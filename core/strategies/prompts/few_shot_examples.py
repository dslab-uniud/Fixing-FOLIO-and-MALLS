"""Few-shot examples shared by B2, B3, and B4.

Correctness is **ontology-relative**: φ̂ is 'yes' iff it correctly expresses
what p means *using only the predicates and constants declared in ONTOLOGY*.
If the ontology can only partially describe the full sentence but the formula
correctly uses what is available, the verdict is still 'yes' — the comment
may note the expressive limitation, but the formula is not penalised for it.

Two-dimensional taxonomy
------------------------
Each example carries two orthogonal type tags:

  outcome_type    — the logical outcome of the evaluation, independent of how
                    the judge reasons about it.  See OUTCOME_* constants below.

  reasoning_type  — the style of reasoning demonstrated in the 'reasoning'
                    dict for this example.  See REASONING_* constants below.
                    Each example stores reasoning traces for all supported
                    styles so that any method can reuse the same example.

Outcome types
~~~~~~~~~~~~~
OUTCOME_CORRECT                — formula is fully equivalent to p within the
                                  given ontology
OUTCOME_CORRECT_ONTOLOGY_PARTIAL — formula is the best achievable translation;
                                  ontology is too coarse to capture every nuance
                                  of p, but no better formula can be written
OUTCOME_CORRECT_CONSTANTS      — correct formula that uses named constants
OUTCOME_QUANTIFIER_ERROR       — wrong quantifier type (∀ vs ∃)
OUTCOME_SCOPE_ERROR            — quantifier scope reversed or mis-nested
OUTCOME_PREDICATE_ERROR        — wrong predicate used (not in ontology, or
                                  semantically mismatched)
OUTCOME_MISSING_CONDITION      — a semantic condition from p is absent in φ̂
OUTCOME_AMBIGUOUS              — the sentence is genuinely ambiguous between two
                                  logically distinct FOL readings; verdict '?'

Reasoning types
~~~~~~~~~~~~~~~
REASONING_COT           — free chain-of-thought: narrative, step-by-step trace
                          (used by B1, B2, B3)
REASONING_FEATURE_BASED — structured sub-property checklist with pass/fail per
                          criterion and an explicit aggregation step (used by B4)

The 'reasoning' field is a dict keyed by REASONING_* constants.  Callers
specify which style they want; _format_example defaults to REASONING_COT.

Ambiguity taxonomy (OUTCOME_AMBIGUOUS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Natural-language sentences can be scope-ambiguous, referentially ambiguous, or
connective-ambiguous.  When no single FOL formula is the uniquely correct
formalisation, the verdict is '?' and the comment must start with [AMBIGUITY].

Group A — Quantifier structure
  A1  Scope reversal (∃∀ vs ∀∃)        "A professor teaches every student"
  A2  Negation–quantifier scope        ∀x ¬P  ("none") vs ¬∀x P  ("not all")
  A3  Generic vs strict universal      ∀ (no exceptions) vs characteristic/typical
  A4  "Any" — free-choice vs existential  "loves any student" as ∀ vs ∃
  A5  Numeral quantifiers              "two X" as at-least-two vs exactly-two

Group B — Referential / binding
  B1  Donkey anaphora                  pronoun bound by indefinite in rel. clause
  B2  Collective vs distributive       group-as-unit vs each independently
  B3  Definite-description uniqueness  ∃x (simplified) vs ∃!x (Russellian)

Group C — Connective
  C1  Conditional vs biconditional     "if" as → vs ↔ (rule-like implicature)
  C2  "Unless" scope                   ¬B → A  vs  ¬B ↔ A
  C3  Inclusive vs exclusive "or"      ∨ (at-least-one) vs ⊕ (exactly-one)

Group D — Modifier / attachment
  D1  Restrictive vs non-restrictive   relative clause restricts range vs aside

Group E — Semantic multiplicity
  E1  Reciprocal / symmetric           "each other" — all pairs vs some pair
  E2  Branching / Henkin quantifiers   independent choices not expressible in FOL
  E3  Cumulative quantification        "3 profs taught 5 students" — sets, not pairs
"""

from __future__ import annotations

import json as _json

# ---------------------------------------------------------------------------
# Taxonomy constants
# ---------------------------------------------------------------------------

# — outcome —
OUTCOME_CORRECT                = "correct"
OUTCOME_CORRECT_ONTOLOGY_PARTIAL = "correct_ontology_partial"
OUTCOME_CORRECT_CONSTANTS      = "correct_constants"
OUTCOME_QUANTIFIER_ERROR       = "quantifier_type_error"
OUTCOME_SCOPE_ERROR            = "scope_error"
OUTCOME_PREDICATE_ERROR        = "predicate_error"
OUTCOME_MISSING_CONDITION      = "missing_condition"
OUTCOME_AMBIGUOUS              = "ambiguous"   # genuine NL ambiguity; verdict '?'

# — reasoning style —
REASONING_COT           = "cot"
REASONING_FEATURE_BASED = "feature_based"

# ---------------------------------------------------------------------------
# Shared ontology
# ---------------------------------------------------------------------------

ONTOLOGY: dict = {
    "relations": {
        # Unary
        "Course":     {"arity": 1,
                       "positive_meaning": "x is a course",
                       "negative_meaning": "x is not a course"},
        "Student":    {"arity": 1,
                       "positive_meaning": "x is a student",
                       "negative_meaning": "x is not a student"},
        "GradStudent":{"arity": 1,
                       "positive_meaning": "x is a graduate student",
                       "negative_meaning": "x is not a graduate student"},
        "Professor":  {"arity": 1,
                       "positive_meaning": "x is a professor",
                       "negative_meaning": "x is not a professor"},
        "Tenured":    {"arity": 1,
                       "positive_meaning": "x has tenure",
                       "negative_meaning": "x does not have tenure"},
        "Graduates":  {"arity": 1,
                       "positive_meaning": "x graduates",
                       "negative_meaning": "x does not graduate"},
        "Dog":        {"arity": 1,
                       "positive_meaning": "x is a dog",
                       "negative_meaning": "x is not a dog"},
        "Barks":      {"arity": 1,
                       "positive_meaning": "x barks (no temporal qualifier — the ontology does not distinguish time of barking)",
                       "negative_meaning": "x does not bark"},
        "Tourist":    {"arity": 1,
                       "positive_meaning": "x is a tourist",
                       "negative_meaning": "x is not a tourist"},
        "City":       {"arity": 1,
                       "positive_meaning": "x is a city",
                       "negative_meaning": "x is not a city"},
        "Animal":     {"arity": 1,
                       "positive_meaning": "x is an animal",
                       "negative_meaning": "x is not an animal"},
        # Binary
        "Passes":     {"arity": 2,
                       "positive_meaning": "x passes exam e",
                       "negative_meaning": "x does not pass exam e"},
        "WakesUp":    {"arity": 2,
                       "positive_meaning": "x wakes up y",
                       "negative_meaning": "x does not wake up y"},
        "Visits":     {"arity": 2,
                       "positive_meaning": "x visits location y",
                       "negative_meaning": "x does not visit location y"},
        "Owns":       {"arity": 2,
                       "positive_meaning": "x owns y",
                       "negative_meaning": "x does not own y"},
        "Loves":      {"arity": 2,
                       "positive_meaning": "x loves y",
                       "negative_meaning": "x does not love y"},
        "Teaches":    {"arity": 2,
                       "positive_meaning": "x teaches y",
                       "negative_meaning": "x does not teach y"},
        # Ternary
        "Recommends": {"arity": 3,
                       "positive_meaning": "x recommends y to z (e.g., Professor recommends Course to Student)",
                       "negative_meaning": "x does not recommend y to z"},
    },
    "constants": {
        "exam":      "a specific exam",
        "neighbors": "the group of neighbors",
        "Alice":     "a specific person named Alice",
        "Bob":       "a specific person named Bob",
    },
}


def build_ontology_block(ontology: dict | str | None = None) -> str:
    """Render an ontology dict as a human-readable string for inclusion in prompts.

    Args:
        ontology: dict with 'relations' and 'constants' keys (same structure as
                  ONTOLOGY).  If None, the shared demo ONTOLOGY is used.
                  Each relation entry may use either 'positive_meaning' or the
                  legacy 'description' key; only the positive meaning is rendered.
    """
    if isinstance(ontology, str):
        return ontology
    ont = ontology if ontology is not None else ONTOLOGY
    lines = ["### Ontology\n"]
    lines.append("Relations (name/arity — description):")
    for name, meta in ont["relations"].items():
        desc = meta.get("positive_meaning", meta.get("description", ""))
        lines.append(f"  - {name}/{meta['arity']}: {desc}")
    lines.append("\nConstants:")
    for name, desc in ont["constants"].items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

EXAMPLES: list[dict] = [

    # =========================================================================
    # 1. OUTCOME_CORRECT / CoT + Feature-Based
    #    Deep nesting with a ternary predicate and compound antecedent.
    # =========================================================================
    {
        "outcome_type":   OUTCOME_CORRECT,
        "reasoning_type": REASONING_COT,
        "p":       "Every tenured professor recommends at least one course to every graduate student they teach who passes the exam.",
        "formula": "∀x ((Professor(x) ∧ Tenured(x)) → ∀y ((GradStudent(y) ∧ Teaches(x, y) ∧ Passes(y, exam)) → ∃z (Course(z) ∧ Recommends(x, z, y))))",
        "q":       "yes",
        "comm": (
            "This formula perfectly captures a highly complex dependency chain. It correctly identifies the "
            "subject (tenured professors), the restricted indirect object (graduate students taught by the "
            "professor who pass the constant 'exam'), and the existentially quantified direct object (a course). "
            "The 3-arity predicate Recommends(x, z, y) maintains the correct argument order (recommender, recommended_item, recipient)."
        ),
        "reasoning": {
            REASONING_COT: (
                "The sentence structurally requires three layers of quantification. First, it universally quantifies "
                "over tenured professors: ∀x (Professor(x) ∧ Tenured(x)). Second, for any such professor, it universally "
                "quantifies over a restricted subset of students: those who are graduate students, taught by x, and pass "
                "the exam: ∀y (GradStudent(y) ∧ Teaches(x, y) ∧ Passes(y, exam)). Finally, the consequent asserts the existence "
                "of a course recommended by x to y: ∃z (Course(z) ∧ Recommends(x, z, y)). The formula matches this architecture "
                "perfectly. All predicates, including the ternary Recommends/3, are in the ontology and use the correct arities.\n"
                "Verdict: yes."
            ),
            REASONING_FEATURE_BASED: (
                "Syntactic well-formedness:          Pass — syntactically valid FOL formula with properly closed scopes.\n"
                "Ontological faithfulness:       Pass — Professor/1, Tenured/1, GradStudent/1, Teaches/2, Passes/2, Course/1, "
                "and Recommends/3 all exist. The ternary argument order (recommender x, course z, student y) is strictly correct.\n"
                "Truth-Conditional Equivalence:         Pass — ∃z is correctly nested inside ∀x and ∀y, allowing a different course per student/professor pair. No unstated conditions added.\n"
                "Ambiguity Resolution: Pass - no meaningful other interpretation"
                "Aggregation: all Pass → Verdict: yes."
            ),
        },
    },

    # =========================================================================
    # 2. OUTCOME_SCOPE_ERROR / CoT + Feature-Based
    #    Subtle de re vs. de dicto quantifier inversion affecting a ternary relation.
    # =========================================================================
    {
        "outcome_type":   OUTCOME_SCOPE_ERROR,
        "reasoning_type": REASONING_FEATURE_BASED,
        "p":       "There is a specific course that every professor recommends to Alice.",
        "formula": "∀x (Professor(x) → ∃y (Course(y) ∧ Recommends(x, y, Alice)))",
        "q":       "no",
        "comm": (
            "A classic scope inversion. The natural language explicitly demands a 'de re' reading ('a specific course') "
            "where the existential quantifier must scope over the universal. The formula provides a 'de dicto' reading "
            "where the existential is bound inside the universal, incorrectly allowing each professor to recommend a *different* course."
        ),
        "reasoning": {
            REASONING_FEATURE_BASED: (
                "Syntactic well-formedness:          Pass — valid FOL syntax.\n"
                "Ontological faithfulness:       Pass — All predicates and the constant Alice are valid.\n"
                "Truth-Conditional Equivalence:         Fail — The phrase 'a specific course' forces ∃y (Course(y)) to be the "
                "outermost quantifier. By placing ∀x (Professor(x)) on the outside, φ̂ asserts that each professor recommends *some* "
                "course, failing to guarantee they all recommend the *same* course. Scope order is logically fatal.\n"
                "Ambiguity Resolution:      Pass — no, clear interpretation.\n"
                "Aggregation: Fail on scope → Verdict: no."
            ),
            REASONING_COT: (
                "The English sentence emphasizes uniqueness and specificity with 'a specific course'. This requires the existential "
                "quantifier to take wide scope: ∃y (Course(y) ∧ ∀x (Professor(x) → Recommends(x, y, Alice))). The provided formula places "
                "the universal quantifier first: ∀x ∃y. This means Professor A might recommend Course 1, and Professor B might recommend Course 2. "
                "While the formula is well-formed, it models a much weaker claim than the sentence requires. The error lies entirely in quantifier nesting.\n"
                "Verdict: no."
            ),
        },
        "refinement": "∃y (Course(y) ∧ ∀x (Professor(x) → Recommends(x, y, Alice)))",
    },

    # =========================================================================
    # 3. OUTCOME_AMBIGUOUS / Donkey Anaphora & Dynamic Semantics
    #    Tackles the "Donkey Sentence" problem where an existential in the
    #    antecedent requires universal binding in the consequent.
    # =========================================================================
    {
        "outcome_type":   OUTCOME_AMBIGUOUS,
        "reasoning_type": REASONING_COT,
        "p":       "Every professor who teaches a course recommends it to Bob.",
        "formula": "∀x ∀y ((Professor(x) ∧ Course(y) ∧ Teaches(x, y)) → Recommends(x, y, Bob))",
        "q":       "?",
        "comm": (
            "[AMBIGUITY] This is a classic 'donkey sentence'. The indefinite article 'a course' normally translates to an existential (∃). "
            "However, an existential in the antecedent cannot bind the pronoun 'it' in the consequent in standard FOL. The standard workaround "
            "is to convert 'a course' into a universal quantifier (∀y) scoping over the entire implication, which the formula does. "
            "However, this assumes the 'strong' donkey reading (recommends *every* course they teach). A 'weak' donkey reading (recommends *at least one* "
            "course they teach) is equally plausible in English but requires a completely different formalization. Thus, it remains ambiguous."
        ),
        "reasoning": {
            REASONING_COT: (
                "The pronoun 'it' in the consequent refers back to 'a course' in the antecedent. If formalized literally with ∃, "
                "the variable remains free in the consequent: ∀x ((Professor(x) ∧ ∃y(Course(y) ∧ Teaches(x, y))) → Recommends(x, y, Bob)) is malformed. "
                "The formula provided solves this by raising the quantifier to a universal over the whole sentence: ∀x ∀y. "
                "This yields the 'strong' donkey reading: if a professor teaches 5 courses, they recommend all 5 to Bob. "
                "However, natural language also permits the 'weak' reading: they recommend *some* course they teach to Bob, requiring "
                "∀x (Professor(x) ∧ ∃y(Course(y) ∧ Teaches(x, y)) → ∃z(Course(z) ∧ Teaches(x, z) ∧ Recommends(x, z, Bob))). "
                "Because standard FOL cannot smoothly resolve dynamic binding, and English allows both strong and weak truth conditions, "
                "the formula is an ambiguous approximation.\n"
                "Verdict: ?"
            ),
            REASONING_FEATURE_BASED: (
                "Syntactic well-formedness:          Pass — syntactically valid, avoids the free variable trap.\n"
                "Ontological faithfulness:       Pass — Recommends/3 used correctly.\n"
                "Truth-Conditional Equivalence:         Uncertain — Escapes the dynamic binding problem via universal expansion, but forces a 'strong' reading. Matches strong donkey truth conditions, fails weak donkey truth conditions.\n"
                "Ambiguity Resolution:      Fail — There are different interpretations (strong donkey/ weak donkey).\n"
                "Aggregation: Uncertain on anaphoric resolution → Verdict: ?"
            ),
        },
    },

    # =========================================================================
    # 4. OUTCOME_PREDICATE_ERROR / Argument Transposition in Ternary
    #    Checks for strict adherence to defined argument positions.
    # =========================================================================
    {
        "outcome_type":   OUTCOME_PREDICATE_ERROR,
        "reasoning_type": REASONING_FEATURE_BASED,
        "p":       "Alice recommends Bob to every professor.",
        "formula": "∀x (Professor(x) → Recommends(Alice, x, Bob))",
        "q":       "no",
        "comm": (
            "The formula uses the correct predicates but transposes the arguments in the ternary relation. "
            "The ontology defines Recommends(x, y, z) as 'x recommends y to z'. The formula translates to "
            "'Alice recommends every professor to Bob', which inverses the direct and indirect objects of the English sentence."
        ),
        "reasoning": {
            REASONING_FEATURE_BASED: (
                "Syntactic well-formedness:          Pass — syntax is valid.\n"
                "Ontological faithfulness:       Fail — The ontology specifies Recommends(recommender, recommended_item, recipient). "
                "The sentence requires Alice to be the recommender, Bob to be the recommended item, and the professor to be the recipient: Recommends(Alice, Bob, x). "
                "The formula writes Recommends(Alice, x, Bob), meaning Alice recommends the professor to Bob. Argument semantics are violated.\n"
                "Truth-Conditional Equivalence:          Fail — The formula represents a completely different relational exchange.\n"
                "Ambiguity Resolution:      Pass.\n"
                "Aggregation: Fail on predicate-argument (transposition) → Verdict: no."
            ),
            REASONING_COT: (
                "While the vocabulary is correct, the ternary relation Recommends/3 requires strict argument ordering. "
                "According to the ontology, Recommends(x, y, z) means x recommends y (the item) to z (the recipient). "
                "In the sentence 'Alice recommends Bob to every professor', Alice is x, Bob is y, and the professor (variable x in the formula) is z. "
                "The correct formulation is Recommends(Alice, Bob, x). By writing Recommends(Alice, x, Bob), the formula mathematically asserts "
                "that Alice is recommending the professors to Bob. This breaks the truth conditions fundamentally.\n"
                "Verdict: no."
            ),
        },
        "refinement": "∀x (Professor(x) → Recommends(Alice, Bob, x))",
    },

    # =========================================================================
    # 5. OUTCOME_CORRECT_ONTOLOGY_PARTIAL / Missing Adverbial Nuance
    #    Demonstrates robust fallback when the ontology lacks expressive power.
    # =========================================================================
    {
        "outcome_type":   OUTCOME_CORRECT_ONTOLOGY_PARTIAL,
        "reasoning_type": REASONING_COT,
        "p":       "Bob strongly recommends a course to Alice.",
        "formula": "∃x (Course(x) ∧ Recommends(Bob, x, Alice))",
        "q":       "yes",
        "comm": (
            "The adverb 'strongly' modifies the recommendation action, but standard FOL and the provided ontology "
            "do not support event semantics (like Neo-Davidsonian event variables) or adverbial modifiers. "
            "The formula is the best possible approximation given the ontology's limits, correctly identifying the core relation."
        ),
        "reasoning": {
            REASONING_COT: (
                "The sentence features an adverbial modifier ('strongly') applied to the verb 'recommends'. "
                "Because our ontology only contains the basic ternary predicate Recommends/3 and lacks event-based variables "
                "(e.g., ∃e(Recommends(e, Bob, x, Alice) ∧ Strong(e))), we cannot represent the intensity of the recommendation. "
                "However, stripping away the unrepresentable nuance, the core logical claim is an existential relation: Bob recommends "
                "some course to Alice. The formula ∃x (Course(x) ∧ Recommends(Bob, x, Alice)) maps this perfectly. The omission of "
                "'strongly' is an expressive limitation of the ontology, not a formalization error.\n"
                "Verdict: yes."
            ),
            REASONING_FEATURE_BASED: (
                "Syntactic well-formedness:          Pass — valid FOL.\n"
                "Ontological Faithfulness:       Pass — Course/1 and Recommends/3 used accurately with proper argument mapping.\n"
                "Truth-Conditional Equivalence:         Pass — Existential quantifier naturally binds the course and within the confines of the provided ontology, this is the tightest possible match. "
                "The unmapped 'strongly' modifier is pardoned under ontology-relative correctness.\n"
                "Ambiguity Resolution:      Pass \n"
                "Aggregation: all Pass → Verdict: yes."
            ),
        },
    },
]

# ---------------------------------------------------------------------------
# Sample inputs for rendered-prompt demos
#
# These are realistic (sentence, formula) pairs drawn from the shared ontology
# and cover the three possible verdict outcomes.  They are used to show a
# complete prompt (system + user turn) in the rendered/ artefacts.
# ---------------------------------------------------------------------------

SAMPLE_INPUTS: list[dict] = [
    # ---- OUTCOME_CORRECT -------------------------------------------------------
    {
        "outcome_type": OUTCOME_CORRECT,
        "p": "Every graduate student taught by Alice visits some city.",
        "formula": "∀x (GradStudent(x) ∧ Teaches(Alice, x) → ∃y (City(y) ∧ Visits(x, y)))",
        "expected_q": "yes",
        "note": (
            "Correct formalization: universally quantifies over grad students taught "
            "by the constant Alice, then existentially asserts they visit some city. "
            "All predicates and constants are in the ontology with correct arities."
        ),
    },
    {
        "outcome_type": OUTCOME_CORRECT,
        "p": "There is a professor who teaches every graduate student.",
        "formula": "∃x (Professor(x) ∧ ∀y (GradStudent(y) → Teaches(x, y)))",
        "expected_q": "yes",
        "note": (
            "Correct formalization: the existential quantifier scopes outside the "
            "universal, fixing one professor who teaches all grad students. "
            "Professor/1, GradStudent/1, and Teaches/2 are all in the ontology."
        ),
    },
    {
        "outcome_type": OUTCOME_CORRECT,
        "p": "No tenured professor teaches Bob.",
        "formula": "¬∃x (Professor(x) ∧ Tenured(x) ∧ Teaches(x, Bob))",
        "expected_q": "yes",
        "note": (
            "Correct negated-existential: 'no X such that ...' is faithfully encoded "
            "as ¬∃x (...). Equivalent to ∀x ((Professor(x) ∧ Tenured(x)) → ¬Teaches(x, Bob)). "
            "All predicates and the constant Bob are in the ontology."
        ),
    },
    {
        "outcome_type": OUTCOME_CORRECT,
        "p": "Not every animal that Bob owns barks.",
        "formula": "¬∀x (Animal(x) ∧ Owns(Bob, x) → Barks(x))",
        "expected_q": "yes",
        "note": (
            "Correct negation of a universal: ¬∀x P(x) is the standard FOL encoding "
            "of 'not all X are P'. Equivalent to ∃x (Animal(x) ∧ Owns(Bob, x) ∧ ¬Barks(x)). "
            "All predicates and the constant Bob are in the ontology."
        ),
    },
    # ---- OUTCOME_CORRECT_CONSTANTS ---------------------------------------------
    {
        "outcome_type": OUTCOME_CORRECT_CONSTANTS,
        "p": "Alice owns an animal if and only if Bob owns an animal.",
        "formula": "∃x (Animal(x) ∧ Owns(Alice, x)) ↔ ∃y (Animal(y) ∧ Owns(Bob, y))",
        "expected_q": "yes",
        "note": (
            "Correct biconditional with two existential quantifiers and constants "
            "Alice and Bob. Both branches correctly assert existence of owned animals. "
            "All predicates and constants are in the ontology."
        ),
    },
    {
        "outcome_type": OUTCOME_CORRECT_CONSTANTS,
        "p": "Some tenured professor teaches Bob.",
        "formula": "∃x (Professor(x) ∧ Tenured(x) ∧ Teaches(x, Bob))",
        "expected_q": "yes",
        "note": (
            "Correct existential with conjunction of Professor/1, Tenured/1, "
            "and Teaches/2 with the constant Bob. "
            "All predicates and constants are in the ontology with correct arities."
        ),
    },
    # ---- OUTCOME_CORRECT_ONTOLOGY_PARTIAL -------------------------------------
    {
        "outcome_type": OUTCOME_CORRECT_ONTOLOGY_PARTIAL,
        "p": "Most students who graduate become professors.",
        "formula": "∀x (Student(x) ∧ Graduates(x) → Professor(x))",
        "expected_q": "?",
        "note": (
            "[UNCERTAINTY] 'Most' is a generalized quantifier not expressible in "
            "standard FOL. The formula captures 'all graduating students become "
            "professors', which is strictly stronger than 'most'. The ontology "
            "offers no predicate to distinguish majority quantification."
        ),
    },
    # ---- OUTCOME_QUANTIFIER_ERROR ---------------------------------------------
    {
        "outcome_type": OUTCOME_QUANTIFIER_ERROR,
        "p": "Some dog that barks wakes up Alice.",
        "formula": "∃x (Dog(x) ∧ Barks(x) → WakesUp(x, Alice))",
        "expected_q": "no",
        "note": (
            "Error: '∃x (P(x) → Q(x))' is logically weaker than intended — it is "
            "satisfied by any non-dog. The correct existential formula uses conjunction: "
            "∃x (Dog(x) ∧ Barks(x) ∧ WakesUp(x, Alice))."
        ),
    },
    {
        "outcome_type": OUTCOME_QUANTIFIER_ERROR,
        "p": "Every professor who teaches a graduate student is tenured.",
        "formula": "∃x (Professor(x) ∧ ∃y (GradStudent(y) ∧ Teaches(x, y)) → Tenured(x))",
        "expected_q": "no",
        "note": (
            "Error: ∃x with implication is logically distinct from ∀x. "
            "∃x (P(x) → Q(x)) is satisfied even when no x satisfies P(x). "
            "Correct formula: ∀x ((Professor(x) ∧ ∃y (GradStudent(y) ∧ Teaches(x, y))) → Tenured(x))."
        ),
    },
    # ---- OUTCOME_SCOPE_ERROR --------------------------------------------------
    {
        "outcome_type": OUTCOME_SCOPE_ERROR,
        "p": "Alice teaches every student who passes the exam.",
        "formula": "∀x (Student(x) → Passes(x, exam) ∧ Teaches(Alice, x))",
        "expected_q": "no",
        "note": (
            "Error: operator precedence makes this ∀x (Student(x) → (Passes(x, exam) ∧ Teaches(Alice, x))), "
            "which says every student passes the exam and is taught by Alice — stronger than p. "
            "Correct formula: ∀x ((Student(x) ∧ Passes(x, exam)) → Teaches(Alice, x))."
        ),
    },
    # ---- OUTCOME_PREDICATE_ERROR ----------------------------------------------
    {
        "outcome_type": OUTCOME_PREDICATE_ERROR,
        "p": "Some tourist visits every city.",
        "formula": "∃x (Tourist(x) ∧ ∀y (City(y) → Visits(x, y, Alice)))",
        "expected_q": "no",
        "note": (
            "Error: Visits is used with 3 arguments but is defined as Visits/2 in the ontology. "
            "Alice is a spurious third argument. "
            "Correct formula: ∃x (Tourist(x) ∧ ∀y (City(y) → Visits(x, y)))."
        ),
    },
    # ---- OUTCOME_MISSING_CONDITION --------------------------------------------
    {
        "outcome_type": OUTCOME_MISSING_CONDITION,
        "p": "Every dog that barks wakes up Bob.",
        "formula": "∀x (Barks(x) → WakesUp(x, Bob))",
        "expected_q": "no",
        "note": (
            "Missing condition: the formula applies to everything that barks, not just dogs. "
            "Dog(x) must appear in the antecedent. "
            "Correct formula: ∀x (Dog(x) ∧ Barks(x) → WakesUp(x, Bob))."
        ),
    },
    {
        "outcome_type": OUTCOME_MISSING_CONDITION,
        "p": "Every city that some tourist visits is loved by Alice.",
        "formula": "∀x (City(x) → Loves(Alice, x))",
        "expected_q": "no",
        "note": (
            "Missing condition: the formula applies to all cities, not just those "
            "visited by some tourist. The antecedent is missing ∃y (Tourist(y) ∧ Visits(y, x)). "
            "Correct formula: ∀x ((City(x) ∧ ∃y (Tourist(y) ∧ Visits(y, x))) → Loves(Alice, x))."
        ),
    },
    # ---- HARD CORRECT cases ------------------------------------------------
    {
        "outcome_type": OUTCOME_CORRECT,
        "p": "Alice loves Bob if and only if Bob teaches Alice.",
        "formula": "Loves(Alice, Bob) ↔ Teaches(Bob, Alice)",
        "expected_q": "yes",
        "note": (
            "Correct biconditional between two ground atoms — no quantifiers needed. "
            "Alice and Bob are ontology constants; Loves/2 and Teaches/2 are in the "
            "ontology with the correct argument order (Loves(lover, loved), "
            "Teaches(teacher, student))."
        ),
    },
    {
        "outcome_type": OUTCOME_CORRECT,
        "p": "Not every professor who teaches Bob is tenured.",
        "formula": "¬∀x ((Professor(x) ∧ Teaches(x, Bob)) → Tenured(x))",
        "expected_q": "yes",
        "note": (
            "Correct outer negation of a universal: ¬∀x P(x) encodes 'not every X "
            "satisfies P'. Equivalent to ∃x (Professor(x) ∧ Teaches(x, Bob) ∧ ¬Tenured(x)). "
            "Applying De Morgan or moving the negation inside both give the same truth "
            "conditions. All predicates and the constant Bob are in the ontology."
        ),
    },
    # ---- HARD AMBIGUOUS cases ----------------------------------------------
    {
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "A professor teaches every student.",
        "formula": "∃x (Professor(x) ∧ ∀y (Student(y) → Teaches(x, y)))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] Classic ∃∀ vs ∀∃ scope ambiguity. The strong reading "
            "(one specific professor teaches all students) matches the given formula "
            "(∃ outside ∀); the weak reading (every student is taught by some, "
            "possibly different, professor) requires ∀y (Student(y) → ∃x (Professor(x) "
            "∧ Teaches(x, y))). English 'a professor teaches every student' does not "
            "disambiguate between these two logically distinct readings."
        ),
    },
    {
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Every professor who owns a dog teaches it.",
        "formula": "∀x ∀y ((Professor(x) ∧ Dog(y) ∧ Owns(x, y)) → Teaches(x, y))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] Donkey anaphora: 'it' is anaphoric to the indefinite 'a dog'. "
            "The formula adopts the universal donkey reading (for every professor–dog "
            "pair where the professor owns the dog, the professor teaches it). An "
            "alternative existential donkey reading would only assert one such dog "
            "is taught. Additionally, Teaches/2 applies to a dog, which is "
            "ontologically unusual (typically a student-type entity) though not "
            "prohibited by arity. The correct interpretation is genuinely underdetermined."
        ),
    },
    # ---- HARD ERROR cases --------------------------------------------------
    {
        "outcome_type": OUTCOME_MISSING_CONDITION,
        "p": "Only tourists who visit some city are loved by Alice.",
        "formula": "∀x (Tourist(x) → Loves(Alice, x))",
        "expected_q": "no",
        "note": (
            "Error: the formula applies to all tourists, not only those who visit "
            "some city. The guard ∃y (City(y) ∧ Visits(x, y)) is absent from the "
            "antecedent. 'Only A who B' requires the B condition in the antecedent. "
            "Correct formula: ∀x ((Tourist(x) ∧ ∃y (City(y) ∧ Visits(x, y))) → Loves(Alice, x))."
        ),
    },
    {
        "outcome_type": OUTCOME_SCOPE_ERROR,
        "p": "Only tenured professors teach graduate students.",
        "formula": "∀x (Professor(x) ∧ ∃y (GradStudent(y) ∧ Teaches(x, y)) → Tenured(x))",
        "expected_q": "no",
        "note": (
            "Error: 'Only A do B' encodes as ∀x (B(x) → A(x)), not ∀x (part-of-A(x) ∧ B(x) → "
            "rest-of-A(x)). The formula only checks that professors who teach grad students "
            "are tenured, but does not block non-professors from teaching grad students. "
            "Correct formula: ∀x (∃y (GradStudent(y) ∧ Teaches(x, y)) → (Professor(x) ∧ Tenured(x))). "
            "The full 'tenured professor' description belongs in the consequent."
        ),
    },

    # =========================================================================
    # Ambiguity type examples — one per taxonomy type (A2–E3)
    # =========================================================================

    # A2 — Negation–quantifier scope
    {  # for me not ambiguous
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Every student doesn't love Bob.",
        "formula": "∀x (Student(x) → ¬Loves(x, Bob))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] A2 — Negation–quantifier scope. English 'Every X doesn't P' "
            "is ambiguous between: (1) ∀ > ¬ (strong): no student loves Bob — formula "
            "given; (2) ¬ > ∀ (weak): not every student loves Bob — "
            "¬∀x (Student(x) → Loves(x, Bob)). Reading (1) entails (2) but not vice "
            "versa. The sentence does not fix the precedence of negation and the universal."
        ),
    },
    # A3 — Generic vs strict universal (also in EXAMPLES as #11)
    { # not ambiguous, only for all possible
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Dogs bark.",
        "formula": "∀x (Dog(x) → Barks(x))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] A3 — Generic vs strict universal. English bare-plural generics "
            "('dogs bark') describe characteristic properties, not exceptionless universals. "
            "The formula encodes the strict universal; a mute dog falsifies it but not the "
            "generic. Standard FOL has no generic quantifier — ∀ is the only approximation "
            "but is strictly stronger than the intended generic statement."
        ),
    },
    # A4 — "Any" as free-choice (∀) vs existential (∃)
    { # not ambiguous
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Every professor loves any student who visits a city.",
        "formula": "∀x (Professor(x) → ∀y ((Student(y) ∧ ∃z (City(z) ∧ Visits(y, z))) → Loves(x, y)))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] A4 — 'Any' as universal vs existential. 'Any student' in positive "
            "contexts can be a free-choice universal (every such student) — formula given — "
            "or a plain existential (at least one such student) — "
            "∀x (Professor(x) → ∃y (Student(y) ∧ ∃z (City(z) ∧ Visits(y, z)) ∧ Loves(x, y))). "
            "The two readings are logically distinct: the universal is strictly stronger."
        ),
    },
    # A5 — Numeral quantifiers: at-least-two vs exactly-two
    { # ok
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Two professors teach Bob.",
        "formula": "∃x ∃y (Professor(x) ∧ Professor(y) ∧ Teaches(x, Bob) ∧ Teaches(y, Bob) ∧ x ≠ y)",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] A5 — Numeral quantifiers. 'Two professors' is ambiguous between "
            "'at least two' and 'exactly two'. The formula encodes at-least-two (x and y "
            "may be the same individual without identity in the ontology). The exactly-two "
            "reading requires a counting/uniqueness condition using identity (= / ≠), which "
            "this ontology does not provide. The formula is thus under-specified relative "
            "to the exact-count reading."
        ),
    },
    # B2 — Collective vs distributive
    { # interesting, but not ambiguous in FOL?
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Some tourists visited every city.",
        "formula": "∃x (Tourist(x) ∧ ∀y (City(y) → Visits(x, y)))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] B2 — Collective vs distributive. 'Some tourists' is plural and "
            "allows a collective reading (the group together covered every city) alongside "
            "the distributive reading (one individual visited every city). The formula "
            "captures only the distributive reading; collective readings require "
            "higher-order or plural logic, unavailable in standard FOL."
        ),
    },
    # B3 — Definite description uniqueness (Russellian vs simplified)
    { # ok, probably the Russellian reading is the correct one
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "The professor who teaches Alice is tenured.",
        "formula": "∃x (Professor(x) ∧ Teaches(x, Alice) ∧ Tenured(x))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] B3 — Definite description uniqueness. English 'the' presupposes "
            "exactly one professor teaches Alice (uniqueness). The formula only asserts "
            "existence of some tenured professor who teaches Alice (simplified reading), "
            "dropping the uniqueness condition. The Russellian analysis would add "
            "∀y (Professor(y) ∧ Teaches(y, Alice) → y = x), requiring identity (= not in "
            "ontology). Whether to include uniqueness is a genuine formalization choice."
        ),
    },
    # C1 — Conditional vs biconditional (also in EXAMPLES as #12)
    { # ok
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "A student graduates if they pass the exam.",
        "formula": "∀x (Student(x) → (Passes(x, exam) ↔ Graduates(x)))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] C1 — Conditional vs biconditional. 'If' in rule-like sentences "
            "often carries a biconditional (↔) implicature in natural language. The formula "
            "uses ↔, asserting passing is both necessary and sufficient. The literal "
            "semantic content of 'if P then Q' is → (sufficiency only): "
            "∀x (Student(x) ∧ Passes(x, exam) → Graduates(x)). Neither reading is "
            "uniquely forced by the English sentence."
        ),
    },
    # C2 — "Unless" as if-not (→) vs exclusive-conditional (↔)
    { # ok
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Every student visits a city unless they graduate.",
        "formula": "∀x (Student(x) → (¬Graduates(x) ↔ ∃y (City(y) ∧ Visits(x, y))))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] C2 — 'Unless' scope. 'A unless B' can mean (1) if ¬B then A "
            "(one-directional): ∀x (Student(x) ∧ ¬Graduates(x) → ∃y (City(y) ∧ Visits(x, y))); "
            "or (2) ¬B if and only if A (biconditional): formula given. Reading (2) adds "
            "the claim that graduating students never visit a city — stronger than what the "
            "sentence states. 'Unless' is standardly → in formal semantics, but the ↔ "
            "implicature is pervasive in natural language."
        ),
    },
    # C3 — Inclusive vs exclusive "or"
    { # ok
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Every professor teaches Bob or Alice.",
        "formula": "∀x (Professor(x) → (Teaches(x, Bob) ⊕ Teaches(x, Alice)))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] C3 — Inclusive vs exclusive disjunction. Natural-language 'or' "
            "is typically inclusive (∨: at least one): ∀x (Professor(x) → (Teaches(x, Bob) "
            "∨ Teaches(x, Alice))). The formula uses exclusive-or (⊕: exactly one), "
            "which additionally rules out teaching both. The ⊕ reading is strictly "
            "stronger; the sentence does not force exclusivity."
        ),
    },
    # D1 — Restrictive vs non-restrictive relative clause
    {  # I see the point but this case is almost not ambiguos: the restricive reading is the default. Missing Tenured(x).
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Professors who are tenured teach graduate students.",
        "formula": "∀x (Professor(x) → ∃y (GradStudent(y) ∧ Teaches(x, y)))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] D1 — Restrictive vs non-restrictive relative clause. The "
            "restrictive reading (default in written English without commas): only "
            "tenured professors teach grad students — "
            "∀x ((Professor(x) ∧ Tenured(x)) → ∃y (GradStudent(y) ∧ Teaches(x, y))). "
            "The non-restrictive reading (all professors, who happen to all be tenured): "
            "formula given (no Tenured guard). Under the restrictive reading the formula "
            "is missing Tenured(x); under the non-restrictive reading it is correct. "
            "The intended reading is ambiguous without punctuation or prosodic cues."
        ),
    },
    # E1 — Reciprocal / symmetric (identity absent from ontology)
    { # = is required! 
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Every professor loves every other professor.",
        "formula": "∀x ∀y (Professor(x) ∧ Professor(y) → Loves(x, y))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] E1 — Reciprocal / symmetric. 'Every other professor' intends "
            "x ≠ y, excluding self-love. The formula is the closest FOL encoding but "
            "includes Loves(x, x) (a professor loving themselves) because the ontology "
            "lacks an identity predicate (=). The formula is the unique best "
            "approximation but is strictly weaker than the intended reading."
        ),
    },
    # E2 — Branching / Henkin (not expressible in standard FOL)
    { # ok, FOL limitation
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Some student of every professor and some tourist of every city visited each other.",
        "formula": "∀x ∀y (Professor(x) ∧ City(y) → ∃u ∃v (Student(u) ∧ Tourist(v) ∧ Visits(u, y) ∧ Visits(v, y)))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] E2 — Branching / Henkin quantifiers. The sentence intends: "
            "for each professor there is a (possibly different) student, and for each city "
            "there is a tourist, and those two individuals visited each other — where the "
            "choice of student depends only on the professor and the choice of tourist "
            "depends only on the city (independence). This branching dependency is not "
            "expressible in standard first-order logic (requires Henkin/IF-logic). The "
            "formula conflates the independent choices into a single existential block."
        ),
    },
    # E3 — Cumulative quantification
    { # ok, FOL limitation
        "outcome_type": OUTCOME_AMBIGUOUS,
        "p": "Three professors taught two graduate students.",
        "formula": "∃x ∃y ∃z ∃u ∃v (Professor(x) ∧ Professor(y) ∧ Professor(z) ∧ GradStudent(u) ∧ GradStudent(v) ∧ Teaches(x, u) ∧ Teaches(y, v) ∧ Teaches(z, v))",
        "expected_q": "?",
        "note": (
            "[AMBIGUITY] E3 — Cumulative quantification. The sentence is naturally read "
            "cumulatively: there exist (exactly) three professors and (exactly) two grad "
            "students such that each professor taught at least one of the students and "
            "each student was taught by at least one of the professors. The formula "
            "approximates this but (a) lacks identity so the three professors and two "
            "students may collapse to fewer individuals, and (b) does not enforce that "
            "every named professor and student participates. Exact cumulative readings "
            "require counting and identity, unavailable in this ontology."
        ),
    },
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_example(
    ex: dict,
    with_cot: bool,
    technique: int,
    reasoning_type: str = REASONING_COT,
) -> str:
    """Render one example as a text block for inclusion in a prompt.

    Args:
        ex:             example dict from EXAMPLES.
        with_cot:       if True, include the reasoning trace.
        technique:      1 = Configuration A (no refinement shown);
                        2 = Configuration B (show refinement when q != 'yes').
        reasoning_type: which reasoning style trace to include when with_cot
                        is True.  Defaults to REASONING_COT.  Falls back to
                        REASONING_COT if the requested style is unavailable.
    """
    lines = [
        f"Sentence: {ex['p']}",
        f"Formalization: {ex['formula']}",
    ]
    if with_cot:
        reasoning_dict = ex.get("reasoning", {})
        trace = reasoning_dict.get(reasoning_type) or reasoning_dict.get(REASONING_COT, "")
        if trace:
            lines.append(f"Reasoning:\n{trace}")
        # The trace already ends with "Verdict: <q>." — the JSON below confirms it.
    # Render output as a JSON object matching the requested output structure.
    output: dict = {
        "quality_score": ex["q"],
        "comment": ex["comm"],
    }
    if technique == 2:
        output["refinement"] = ex.get("refinement") if ex["q"] != "yes" else None
    lines.append("Output:\n" + _json.dumps(output, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def build_few_shot_block(
    with_cot: bool,
    technique: int,
    include_ontology: bool = True,
    outcome_types: list[str] | None = None,
    reasoning_type: str = REASONING_COT,
    max_examples: int | None = None,
) -> str:
    """Build the full few-shot examples block for a system prompt.

    Args:
        with_cot:         include reasoning traces in examples.
        technique:        1 = config A, 2 = config B (show refinements).
        include_ontology: prepend the ontology declaration before the examples.
        outcome_types:    if given, only include examples whose 'outcome_type'
                          is in this list.  None means include all.
        reasoning_type:   which reasoning style to use for the traces
                          (REASONING_COT or REASONING_FEATURE_BASED).
        max_examples:     if given, truncate the example pool to at most this
                          many entries (applied after outcome_types filtering).
    """
    pool = (
        EXAMPLES if outcome_types is None
        else [e for e in EXAMPLES if e.get("outcome_type") in outcome_types]
    )
    if max_examples is not None:
        pool = pool[:max_examples]
    body = "\n\n---\n\n".join(
        _format_example(ex, with_cot, technique, reasoning_type)
        for ex in pool
    )
    ontology_part = "\n\n" + build_ontology_block() + "\n\n" if include_ontology else "\n\n"
    return f"### Examples{ontology_part}{body}\n\n---\n\n"
