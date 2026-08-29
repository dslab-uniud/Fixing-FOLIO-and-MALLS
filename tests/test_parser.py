"""
tests/test_parser.py — Unit tests for FOLParser.

Coverage:
  - Numeric constants as relation arguments (digit-starting tokens)
  - Core formula structures (regression)
  - Free-variable detection via FOL.free_variables()
  - Signature extraction via FOLSignature
  - Error cases: forbidden ASCII ops, unbalanced parens, free variables,
    unexpected characters
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.formalism.fol.parser import FOLParser
from core.formalism.fol.signature import FOLSignature
from core.formalism.fol.ast import RelationNode, QuantifierNode, BooleanNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(formula: str):
    return FOLParser().parse(formula)


def sig(formula: str) -> dict:
    tree = FOLParser().parse(formula)
    return FOLSignature().extract(tree)


def free_vars(formula: str, user_consts: set = frozenset()) -> set:
    """Return free variables (mirrors FOL._collect_free logic)."""
    from core.formalism.fol.ast import QuantifierNode, BooleanNode, RelationNode, BoolConstNode

    def _free(node, bound):
        if isinstance(node, BoolConstNode):
            return set()
        if isinstance(node, QuantifierNode):
            return _free(node.body, bound | {node.variable})
        if isinstance(node, BooleanNode):
            result = set()
            for child in node.children:
                result |= _free(child, bound)
            return result
        if isinstance(node, RelationNode):
            return {
                arg for arg in node.arguments
                if arg[0].islower() and arg not in bound and arg not in user_consts
            }
        return set()

    return _free(parse(formula), frozenset())


# ===========================================================================
# 1. Numeric constants
# ===========================================================================

class TestNumericConstants:
    def test_single_digit_in_relation(self):
        tree = parse("∀x (Polygon(x) ∧ Sides(x, 3) → Triangle(x))")
        # Find Sides node and check its arguments
        sides_nodes = []
        def collect(node):
            if isinstance(node, RelationNode) and node.name == "Sides":
                sides_nodes.append(node)
            if isinstance(node, (QuantifierNode, BooleanNode)):
                children = [node.body] if isinstance(node, QuantifierNode) else node.children
                for c in children:
                    collect(c)
        collect(tree)
        assert len(sides_nodes) == 1
        assert sides_nodes[0].arguments == ["x", "3"]

    def test_multi_digit_constant(self):
        tree = parse("∀x (HasAge(x, 42) → Adult(x))")
        nodes = []
        def collect(node):
            if isinstance(node, RelationNode) and node.name == "HasAge":
                nodes.append(node)
            if isinstance(node, (QuantifierNode, BooleanNode)):
                children = [node.body] if isinstance(node, QuantifierNode) else node.children
                for c in children: collect(c)
        collect(tree)
        assert nodes[0].arguments == ["x", "42"]

    def test_numeric_constant_not_free_variable(self):
        # 3 is a digit-starting token; it should NOT appear as a free variable
        assert "3" not in free_vars("∀x (Sides(x, 3) → Triangle(x))")

    def test_numeric_constant_in_signature_const(self):
        s = sig("∀x (Sides(x, 3) ∧ Angles(x, 3) → Triangle(x))")
        assert "3" in s["Const"]
        assert "Sides" in s["Rel"]
        assert s["Rel"]["Sides"] == 2
        assert s["Rel"]["Angles"] == 2

    def test_two_numeric_args(self):
        tree = parse("∀x (Between(x, 1, 2) → Bounded(x))")
        nodes = []
        def collect(node):
            if isinstance(node, RelationNode) and node.name == "Between":
                nodes.append(node)
            if isinstance(node, (QuantifierNode, BooleanNode)):
                children = [node.body] if isinstance(node, QuantifierNode) else node.children
                for c in children: collect(c)
        collect(tree)
        assert nodes[0].arguments == ["x", "1", "2"]

    def test_numeric_with_user_const(self):
        # When 3 is in user_consts, it still must not appear as a free variable
        assert "3" not in free_vars("∀x Sides(x, 3)", user_consts={"3"})

    def test_id30_malls_formula(self):
        """Regression: the exact formula from MALLS id=30 must parse cleanly."""
        f = "∀x (Polygon(x) ∧ Sides(x, 3) ∧ Angles(x, 3) → Triangle(x))"
        tree = parse(f)
        assert tree is not None
        assert "3" not in free_vars(f, user_consts={"3"})


# ===========================================================================
# 2. Core formula structures (regression)
# ===========================================================================

class TestCoreFormulas:
    def test_universal_implication(self):
        tree = parse("∀x (Human(x) → Mortal(x))")
        assert isinstance(tree, QuantifierNode)
        assert tree.quantifier == "forall"

    def test_existential_conjunction(self):
        tree = parse("∃x (Dog(x) ∧ Barks(x))")
        assert isinstance(tree, QuantifierNode)
        assert tree.quantifier == "exists"

    def test_biconditional(self):
        tree = parse("∀x (A(x) ↔ B(x))")
        assert isinstance(tree.body, BooleanNode)
        assert tree.body.operator == "iff"

    def test_xor(self):
        tree = parse("∀x (A(x) ⊕ B(x))")
        assert isinstance(tree.body, BooleanNode)
        assert tree.body.operator == "xor"

    def test_negation(self):
        tree = parse("∀x (¬A(x) → B(x))")
        assert isinstance(tree, QuantifierNode)

    def test_nested_quantifiers(self):
        tree = parse("∀x ∃y Loves(x, y)")
        assert isinstance(tree, QuantifierNode)
        assert tree.variable == "x"
        assert isinstance(tree.body, QuantifierNode)
        assert tree.body.variable == "y"

    def test_conjunction_of_universals(self):
        # Each quantifier's scope is closed by its parentheses, so the top
        # connective is the conjunction of two universally quantified formulas.
        tree = parse("∀x (A(x) → B(x)) ∧ ∀y (C(y) → D(y))")
        assert isinstance(tree, BooleanNode)
        assert tree.operator == "and"
        assert isinstance(tree.children[0], QuantifierNode)
        assert tree.children[0].variable == "x"
        assert isinstance(tree.children[1], QuantifierNode)
        assert tree.children[1].variable == "y"

    def test_nullary_relation(self):
        tree = parse("∀x (Rain → Wet(x))")
        assert tree is not None

    def test_bool_constants(self):
        assert parse("⊤") is not None
        assert parse("⊥") is not None
        assert parse("True") is not None
        assert parse("False") is not None

    def test_equality_atom(self):
        tree = parse("∀x ∀y (x = y → A(x))")
        assert tree is not None

    def test_inequality_atom(self):
        tree = parse("∀x ∀y (x ≠ y → A(x))")
        assert tree is not None

    def test_operator_precedence(self):
        # ↔ < ⊕ < → < ∨ < ∧ < ¬
        # A → B ∨ C  should parse as  A → (B ∨ C)
        tree = parse("∀x (A(x) → B(x) ∨ C(x))")
        body = tree.body  # BooleanNode implies
        assert body.operator == "implies"
        assert body.children[1].operator == "or"

    def test_right_associative_implication(self):
        # A → B → C  should parse as  A → (B → C)
        tree = parse("∀x (A(x) → B(x) → C(x))")
        body = tree.body
        assert body.operator == "implies"
        assert body.children[1].operator == "implies"


# ===========================================================================
# 3. Signature extraction
# ===========================================================================

class TestSignatureExtraction:
    def test_unary_relation(self):
        s = sig("∀x Human(x)")
        assert s["Rel"] == {"Human": 1}

    def test_binary_relation(self):
        s = sig("∀x ∀y Loves(x, y)")
        assert s["Rel"]["Loves"] == 2

    def test_ternary_relation(self):
        s = sig("∀x ∀y ∀z Between(x, y, z)")
        assert s["Rel"]["Between"] == 3

    def test_constant_uppercase(self):
        s = sig("∀x Loves(x, John)")
        assert "John" in s["Const"]

    def test_numeric_in_const(self):
        s = sig("∀x Sides(x, 3)")
        assert "3" in s["Const"]
        assert "Sides" in s["Rel"]

    def test_bound_var_in_var(self):
        s = sig("∀x Human(x)")
        assert "x" in s["Var"]

    def test_equality_excluded_from_rel(self):
        s = sig("∀x ∀y (x = y → A(x))")
        assert "=" not in s["Rel"]


# ===========================================================================
# 4. Free-variable detection
# ===========================================================================

class TestFreeVariables:
    def test_no_free_vars(self):
        assert free_vars("∀x Human(x)") == set()

    def test_free_variable_detected(self):
        # 'y' is not bound
        assert "y" in free_vars("∀x Loves(x, y)")

    def test_numeric_not_free(self):
        assert free_vars("∀x Sides(x, 3)") == set()

    def test_uppercase_not_free(self):
        assert free_vars("∀x Loves(x, John)") == set()

    def test_user_const_not_free(self):
        assert free_vars("∀x Houses(x, books, study)", user_consts={"books", "study"}) == set()

    def test_multiple_quantifiers_no_free(self):
        assert free_vars("∀x ∃y Loves(x, y)") == set()


# ===========================================================================
# 5. Error cases
# ===========================================================================

class TestErrors:
    def test_forbidden_ascii_arrow(self):
        with pytest.raises(ValueError, match="Forbidden"):
            parse("∀x (A(x) -> B(x))")

    def test_forbidden_ascii_and(self):
        with pytest.raises(ValueError, match="Forbidden"):
            parse("∀x (A(x) & B(x))")

    def test_unbalanced_open_paren(self):
        with pytest.raises(Exception):
            parse("∀x (A(x) → B(x)")

    def test_unbalanced_close_paren(self):
        with pytest.raises(Exception):
            parse("∀x A(x) → B(x))")

    def test_unexpected_character(self):
        with pytest.raises(ValueError, match="Unexpected character"):
            parse("∀x A@(x)")

    def test_lowercase_relation_name(self):
        with pytest.raises(ValueError):
            parse("∀x human(x)")

    def test_empty_formula(self):
        with pytest.raises(Exception):
            parse("")
