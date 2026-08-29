from .parser import FOLParser
from .signature import FOLSignature
from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode


class FOL:
    """
    Represents a First-Order Logic formula.
    Stores:
        - original string
        - AST (lazy)
        - signature (lazy)
    """

    # ASCII sequences not allowed in formulas
    _FORBIDDEN = {'->', '&'}

    def __init__(self, formula_str: str, signature: dict | None = None):
        self.formula_str = formula_str
        self._tree = None
        self._signature = None
        self._user_signature = signature  # user-provided override

    @property
    def tree(self) -> Node:
        """Lazily parse string into AST."""
        if self._tree is None:
            parser = FOLParser()
            self._tree = parser.parse(self.formula_str)
        return self._tree

    @property
    def signature(self) -> dict:
        """
        Return the logical signature.
        If the user provided one at construction time, return it as-is.
        Otherwise lazily compute it from the AST.
        """
        if self._user_signature is not None:
            return self._user_signature
        if self._signature is None:
            extractor = FOLSignature()
            self._signature = extractor.extract(self.tree)
        return self._signature

    def validate(self) -> bool:
        """
        Return True iff the formula is well-formed:
            - no forbidden ASCII operators (e.g. ->, &)
            - balanced parentheses
            - parseable by FOLParser
            - no free variables
        """
        try:
            from utils.utility import validate_parenthesis
            for seq in self._FORBIDDEN:
                if seq in self.formula_str:
                    return False
            if not validate_parenthesis(self.formula_str):
                return False
            _ = self.tree  # triggers parse; raises on syntax error
            return len(self.free_variables()) == 0
        except Exception:
            return False

    def free_variables(self) -> set:
        """Return the set of free (unbound) lowercase variable names."""
        return self._collect_free(self.tree, bound=frozenset())

    def bound_variables(self) -> set:
        """Return the set of all quantifier-bound variable names."""
        return self._collect_bound(self.tree)

    def normalize(self):
        """Optional normalization (alpha-renaming, NNF, etc.)."""
        pass

    def to_smtlib(self) -> str:
        """Return the SMT-LIB string representation of this formula."""
        from .encoder_z3 import Z3ContextBuilder, FOLZ3Encoder
        ctx = Z3ContextBuilder(self.signature)
        symbols = ctx.build_symbols()
        encoder = FOLZ3Encoder()
        z3_expr = encoder.encode(self.tree, symbols)
        return z3_expr.sexpr()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _collect_free(self, node: Node, bound: frozenset) -> set:
        # Names the user declared as constants are never free variables.
        user_consts = self._user_signature['Const'] if self._user_signature else set()
        return self._collect_free_inner(node, bound, user_consts)

    def _collect_free_inner(self, node: Node, bound: frozenset, user_consts: set) -> set:
        if isinstance(node, BoolConstNode):
            return set()
        if isinstance(node, QuantifierNode):
            return self._collect_free_inner(node.body, bound | {node.variable}, user_consts)
        if isinstance(node, BooleanNode):
            result = set()
            for child in node.children:
                result |= self._collect_free_inner(child, bound, user_consts)
            return result
        if isinstance(node, RelationNode):
            return {
                arg for arg in node.arguments
                if arg[0].islower() and arg not in bound and arg not in user_consts
            }
        return set()

    def _collect_bound(self, node: Node) -> set:
        if isinstance(node, BoolConstNode):
            return set()
        if isinstance(node, QuantifierNode):
            return {node.variable} | self._collect_bound(node.body)
        if isinstance(node, BooleanNode):
            result = set()
            for child in node.children:
                result |= self._collect_bound(child)
            return result
        return set()
