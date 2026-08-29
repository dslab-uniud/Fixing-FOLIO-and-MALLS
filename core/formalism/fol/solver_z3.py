from z3 import Solver, Not, sat, unsat
from .encoder_z3 import Z3ContextBuilder, FOLZ3Encoder


class SolverTimeoutError(Exception):
    """Raised when a Z3 solver call exceeds the specified timeout."""


class FOLSolver:
    """
    Provides semantic reasoning:
        - satisfiability
        - equivalence
        - implication

    All methods accept:
        theory  : list[str] | None — axiom formula strings added before checking
        timeout : int              — timeout in milliseconds (default: 10 s);
                                     raises SolverTimeoutError if exceeded
    """

    # Default timeout for all Z3 calls (milliseconds).
    DEFAULT_TIMEOUT = 10_000

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def is_satisfiable(self, formula, theory=None, timeout=DEFAULT_TIMEOUT) -> bool:
        """
        Return True if satisfiable, False if unsatisfiable or malformed.
        Raises SolverTimeoutError on timeout.
        """
        try:
            sig = formula.signature
            if theory:
                sig = self._merge_theory_signature(sig, theory)

            ctx = Z3ContextBuilder(sig)
            symbols = ctx.build_symbols()
            encoder = FOLZ3Encoder()

            solver = Solver()
            if timeout is not None:
                solver.set("timeout", timeout)

            theory_trees = self._parse_theory_trees(theory or [])
            if self._uses_equality(formula.tree, *theory_trees):
                self._add_equality_axioms(solver, ctx.sort)

            if theory:
                for z3_ax in self._encode_theory(theory, symbols, encoder):
                    solver.add(z3_ax)

            z3_expr = encoder.encode(formula.tree, symbols)
            solver.add(z3_expr)

            result = solver.check()
            if result == sat:
                return True
            if result == unsat:
                return False
            raise SolverTimeoutError(
                f"Z3 returned '{result}' (likely timeout after {timeout} ms)"
            )
        except SolverTimeoutError:
            raise
        except Exception:
            return False

    def are_equivalent(self, f1, f2, theory=None, timeout=DEFAULT_TIMEOUT) -> bool:
        """
        F and G equivalent iff (F XOR G) is unsatisfiable.
        Returns False if either formula is malformed.
        Raises SolverTimeoutError on timeout.
        """
        try:
            sig = self._merge_signatures(f1, f2)
            if theory:
                sig = self._merge_theory_signature(sig, theory)

            ctx = Z3ContextBuilder(sig)
            symbols = ctx.build_symbols()
            encoder = FOLZ3Encoder()

            solver = Solver()
            if timeout is not None:
                solver.set("timeout", timeout)

            theory_trees = self._parse_theory_trees(theory or [])
            if self._uses_equality(f1.tree, f2.tree, *theory_trees):
                self._add_equality_axioms(solver, ctx.sort)

            if theory:
                for z3_ax in self._encode_theory(theory, symbols, encoder):
                    solver.add(z3_ax)

            z3_f1 = encoder.encode(f1.tree, symbols)
            z3_f2 = encoder.encode(f2.tree, symbols)
            solver.add(z3_f1 != z3_f2)

            result = solver.check()
            if result == unsat:
                return True
            if result == sat:
                return False
            raise SolverTimeoutError(
                f"Z3 returned '{result}' (likely timeout after {timeout} ms)"
            )
        except SolverTimeoutError:
            raise
        except Exception:
            return False

    def implies(self, f1, f2, theory=None, timeout=DEFAULT_TIMEOUT) -> bool:
        """
        F implies G iff (F ∧ ¬G) unsatisfiable.
        Returns False if either formula is malformed.
        Raises SolverTimeoutError on timeout.
        """
        try:
            sig = self._merge_signatures(f1, f2)
            if theory:
                sig = self._merge_theory_signature(sig, theory)

            ctx = Z3ContextBuilder(sig)
            symbols = ctx.build_symbols()
            encoder = FOLZ3Encoder()

            solver = Solver()
            if timeout is not None:
                solver.set("timeout", timeout)

            theory_trees = self._parse_theory_trees(theory or [])
            if self._uses_equality(f1.tree, f2.tree, *theory_trees):
                self._add_equality_axioms(solver, ctx.sort)

            if theory:
                for z3_ax in self._encode_theory(theory, symbols, encoder):
                    solver.add(z3_ax)

            z3_f1 = encoder.encode(f1.tree, symbols)
            z3_f2 = encoder.encode(f2.tree, symbols)
            solver.add(z3_f1)
            solver.add(Not(z3_f2))

            result = solver.check()
            if result == unsat:
                return True
            if result == sat:
                return False
            raise SolverTimeoutError(
                f"Z3 returned '{result}' (likely timeout after {timeout} ms)"
            )
        except SolverTimeoutError:
            raise
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _merge_signatures(self, f1, f2):
        from .signature import FOLSignature
        extractor = FOLSignature()
        return extractor.merge(f1.signature, f2.signature)

    def _theory_signature(self, theory_strs: list[str]) -> dict:
        """Merge the signatures of all axiom formula strings."""
        from .parser import FOLParser
        from .signature import FOLSignature
        parser = FOLParser()
        extractor = FOLSignature()
        merged = {'Rel': {}, 'Var': set(), 'Const': set()}
        for s in theory_strs:
            tree = parser.parse(s)
            sig = extractor.extract(tree)
            merged = extractor.merge(merged, sig)
        return merged

    def _merge_theory_signature(self, base_sig: dict, theory_strs: list[str]) -> dict:
        from .signature import FOLSignature
        extractor = FOLSignature()
        theory_sig = self._theory_signature(theory_strs)
        return extractor.merge(base_sig, theory_sig)

    def _parse_theory_trees(self, theory_strs: list[str]) -> list:
        """Return parsed AST nodes for a list of axiom strings."""
        from .parser import FOLParser
        parser = FOLParser()
        return [parser.parse(s) for s in theory_strs]

    def _encode_theory(self, theory_strs: list[str], symbols: dict, encoder: FOLZ3Encoder):
        """Parse and encode each axiom string into a Z3 expression."""
        from .parser import FOLParser
        parser = FOLParser()
        z3_axioms = []
        for s in theory_strs:
            tree = parser.parse(s)
            z3_axioms.append(encoder.encode(tree, symbols))
        return z3_axioms

    # ------------------------------------------------------------------ #
    # Equality support
    # ------------------------------------------------------------------ #

    def _uses_equality(self, *nodes) -> bool:
        """Return True if any of the given AST nodes contains a '=' RelationNode."""
        from .ast import RelationNode, BooleanNode, QuantifierNode
        def _check(node) -> bool:
            if isinstance(node, RelationNode):
                return node.name == '='
            if isinstance(node, BooleanNode):
                return any(_check(c) for c in node.children)
            if isinstance(node, QuantifierNode):
                return _check(node.body)
            return False
        return any(_check(n) for n in nodes)

    def _add_equality_axioms(self, solver, sort) -> None:
        """
        Add reflexivity, symmetry, and transitivity axioms for equality over `sort`.

        Z3's built-in '==' already satisfies these axioms natively, so adding them
        is formally redundant.  We include them explicitly so that the equality
        theory is self-contained and visible in the solver's assertion set.
        """
        from z3 import Const, ForAll, Implies, And
        x = Const('_eq_x', sort)
        y = Const('_eq_y', sort)
        z = Const('_eq_z', sort)
        solver.add(ForAll([x], x == x))                                          # reflexivity
        solver.add(ForAll([x, y], Implies(x == y, y == x)))                      # symmetry
        solver.add(ForAll([x, y, z], Implies(And(x == y, y == z), x == z)))      # transitivity
