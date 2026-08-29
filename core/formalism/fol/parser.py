from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode


class FOLParser:
    """
    Parses a FOL string into an AST.
    Handles:
        - quantifiers (limited scope, treated as high-precedence primaries)
        - operator precedence: ↔ < ⊕ < → < ∨ < ∧ < ¬ < quantifier < primary
        - parentheses
        - relation applications

    FIXED: Quantifiers now have limited scope. ∃y(P(y)) → Q(y) parses as (∃y(P(y))) → Q(y),
    not ∃y(P(y) → Q(y)). This is achieved by treating quantifiers as part of the primary
    expression hierarchy rather than at the top level.
    """

    # Single-character Unicode operators used as tokens
    # ⊤ (U+22A4) = tautology/True,  ⊥ (U+22A5) = contradiction/False
    _OPERATORS = set('∀∃¬∧∨→↔⊕()=≠⊤⊥')

    # Boolean constant keywords and their Unicode equivalents
    _BOOL_CONSTS = {'True': True, 'False': False, '⊤': True, '⊥': False}
    # Forbidden ASCII operator sequences
    _FORBIDDEN = {'->', '&'}

    def parse(self, formula_str: str) -> Node:
        """Main entry point: string → AST."""
        for seq in self._FORBIDDEN:
            if seq in formula_str:
                raise ValueError(f"Forbidden operator {seq!r}; use Unicode symbols")
        self._tokens = self._tokenize(formula_str)
        self._pos = 0
        result = self._parse_formula()
        if self._pos != len(self._tokens):
            raise ValueError(
                f"Unexpected token {self._tokens[self._pos]!r} at position {self._pos}"
            )
        return result

    # ------------------------------------------------------------------ #
    # Tokenizer 
    # [in the formula the chars accepted are: symbols in _OPERATORS, alphanumeric chars, '-', '_'.]
    # ------------------------------------------------------------------ #

    def _tokenize(self, s: str) -> list:
        tokens = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isspace() or ch == ',':
                i += 1
            elif ch in self._OPERATORS:
                tokens.append(ch)
                i += 1
            elif ch.isalpha() or ch == '_':
                j = i
                while j < len(s) and (
                    s[j].isalnum() or s[j] == '_'
                    or (s[j] == '-' and j + 1 < len(s)
                        and (s[j + 1].isalnum() or s[j + 1] == '_'))
                ):
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif ch.isdigit():
                j = i
                while j < len(s) and s[j].isalnum():
                    j += 1
                tokens.append(s[i:j])
                i = j
            else:
                raise ValueError(f"Unexpected character {ch!r} in formula")
        return tokens

    # ------------------------------------------------------------------ #
    # Position helpers
    # ------------------------------------------------------------------ #

    def _current(self):
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected=None):
        tok = self._current()
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r} but got {tok!r}")
        if tok is None:
            raise ValueError("Unexpected end of formula")
        self._pos += 1
        return tok

    # ------------------------------------------------------------------ #
    # Grammar rules (top-down, precedence encoded by nesting)
    # ------------------------------------------------------------------ #

    def _parse_formula(self) -> Node:
        """formula → biconditional (quantifiers now handled at primary level)"""
        return self._parse_biconditional()

    def _parse_quantifier_primary(self) -> Node:
        """
        quantifier → ('∀'|'∃') var quantifier_body [LIMITED SCOPE]
        quantifier_body → negation | quantifier | primary

        This version of quantifier parsing has LIMITED scope - it applies the quantifier
        only to the negation/quantifier/primary expression immediately following it.
        This prevents quantifiers from accidentally consuming binary operators like → that appear later.

        Examples:
          ∃y(P(y)) → Q(y) parses as (∃y(P(y))) → Q(y), not ∃y(P(y) → Q(y))
          ∀x ¬P(x) parses as ∀x(¬P(x)), not ∀x(¬(...))
          ∀x∃y P(x,y) parses as ∀x(∃y(P(x,y)))
        """
        q = self._consume()
        quantifier = 'forall' if q == '∀' else 'exists'
        variable = self._consume()
        if not variable or not variable[0].isalpha() or variable[0].isupper():
            raise ValueError(f"Expected lowercase variable after quantifier, got {variable!r}")

        # The body can be:
        # - Another quantifier (for chaining like ∀x∃y P(x,y))
        # - A negation (for ∀x ¬P(x))
        # - A primary expression (for ∀x P(x) or ∀x(P(x)))
        # We use _parse_negation to handle all three cases at once
        body = self._parse_negation()
        return QuantifierNode(quantifier, variable, body)

    def _parse_biconditional(self) -> Node:
        """biconditional → xor (↔ xor)*"""
        left = self._parse_xor()
        while self._current() == '↔':
            self._consume('↔')
            right = self._parse_xor()
            left = BooleanNode('iff', [left, right])
        return left

    def _parse_xor(self) -> Node:
        """xor → implication (⊕ implication)*"""
        left = self._parse_implication()
        while self._current() == '⊕':
            self._consume('⊕')
            right = self._parse_implication()
            left = BooleanNode('xor', [left, right])
        return left

    def _parse_implication(self) -> Node:
        """implication → disjunction (→ disjunction)?   [right-associative]"""
        left = self._parse_disjunction()
        if self._current() == '→':
            self._consume('→')
            right = self._parse_implication()  # right-recursive for right-assoc
            return BooleanNode('implies', [left, right])
        return left

    def _parse_boolean(self, formula_str: str) -> Node:
        """Alias kept for API compatibility; not used internally."""
        raise NotImplementedError("Use parse() as the public entry point")

    def _parse_disjunction(self) -> Node:
        """disjunction → conjunction (∨ conjunction)*"""
        left = self._parse_conjunction()
        while self._current() == '∨':
            self._consume('∨')
            right = self._parse_conjunction()
            left = BooleanNode('or', [left, right])
        return left

    def _parse_conjunction(self) -> Node:
        """conjunction → negation (∧ negation)*"""
        left = self._parse_negation()
        while self._current() == '∧':
            self._consume('∧')
            right = self._parse_negation()
            left = BooleanNode('and', [left, right])
        return left

    def _parse_negation(self) -> Node:
        """negation → ¬ negation | primary

        Note: quantifiers are now handled in _parse_primary, so ¬∃y(...) will correctly
        parse as ¬(primary_containing_existential), not as ¬∃y(full_formula).
        """
        if self._current() == '¬':
            self._consume('¬')
            operand = self._parse_negation()  # right-recursive for right-assoc of ¬¬
            return BooleanNode('not', [operand])
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        """
        primary → quantifier | '(' formula ')' | bool_const | equality | relation
        quantifier → ('∀'|'∃') var primary   [limited scope: binds to next primary only]
        bool_const → '⊤' | '⊥' | 'True' | 'False'
        equality → term '=' term          (term is any identifier: upper or lower)
        relation → UpperName '(' args ')' | UpperName

        Key: quantifiers now have LIMITED scope. They apply to the primary immediately following them.
        This fixes the issue where ¬∃y(P(y)) → Q(y) was parsed as ¬∃y(P(y) → Q(y)) instead of
        (¬∃y(P(y))) → Q(y).
        """
        tok = self._current()

        # Handle quantifiers at primary level (limited scope)
        if tok in ('∀', '∃'):
            return self._parse_quantifier_primary()

        if tok == '(':
            self._consume('(')
            result = self._parse_formula()
            self._consume(')')
            return result
        # Boolean constants: ⊤, ⊥ (handled as single-char tokens) or True/False
        if tok in self._BOOL_CONSTS:
            self._consume()
            return BoolConstNode(self._BOOL_CONSTS[tok])
        if tok and (tok[0].isupper() or tok[0].islower() or tok[0] == '_'):
            name = self._consume()
            # Equality atom: term '=' term
            if self._current() == '=':
                self._consume('=')
                rhs = self._current()
                if not rhs or not (rhs[0].isalpha() or rhs[0] == '_'):
                    raise ValueError(f"Expected term after '=', got {rhs!r}")
                self._consume()
                return RelationNode('=', [name, rhs])
            # Inequality atom: term '≠' term  →  ¬(term = term)
            if self._current() == '≠':
                self._consume('≠')
                rhs = self._current()
                if not rhs or not (rhs[0].isalpha() or rhs[0] == '_'):
                    raise ValueError(f"Expected term after '≠', got {rhs!r}")
                self._consume()
                return BooleanNode('not', [RelationNode('=', [name, rhs])])
            # Not equality: name must be an uppercase relation name
            if name[0].islower() or name[0] == '_':
                raise ValueError(
                    f"Unexpected lowercase token {name!r} — "
                    "expected relation name or 't1 = t2'"
                )
            return self._parse_relation_body(name)
        raise ValueError(f"Unexpected token {tok!r} — expected '(' or relation name")

    def _parse_relation_body(self, name: str) -> Node:
        """Parse the optional argument list for an already-consumed relation name."""
        if self._current() == '(':
            self._consume('(')
            args = []
            while self._current() != ')':
                args.append(self._consume())
            self._consume(')')
            return RelationNode(name, args)
        return RelationNode(name, [])
