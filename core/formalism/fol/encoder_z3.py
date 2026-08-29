from z3 import (
    DeclareSort, Function, BoolSort, Const,
    ForAll, Exists, And, Or, Not, Implies, Xor, BoolVal,
)
from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode


class Z3ContextBuilder:
    """
    Builds Z3 symbols from a logical signature.
    Uses single domain sort 'U'.
    """

    def __init__(self, signature: dict):
        self.signature = signature
        self.sort = DeclareSort("U")

    def build_symbols(self) -> dict:
        """
        Instantiate:
            - relational symbols as Functions (arity > 0) or BoolSort Consts (arity 0)
            - individual constants as Const of sort U
        Returns dictionary name -> Z3 object.
        """
        symbols = {}
        U = self.sort
        for name, arity in self.signature['Rel'].items():
            if arity == 0:
                symbols[name] = Const(name, BoolSort())
            else:
                symbols[name] = Function(name, *([U] * arity + [BoolSort()]))
        for name in self.signature['Const']:
            symbols[name] = Const(name, U)
        return symbols


class FOLZ3Encoder:
    """
    Encodes a FOL AST into a Z3 expression (SMT-LIB).
    Requires a symbol dict from Z3ContextBuilder.build_symbols().
    """

    def __init__(self):
        self.sort = DeclareSort("U")

    def encode(self, node: Node, symbols: dict):
        """Walk the AST node and return the corresponding Z3 expression."""
        if isinstance(node, BoolConstNode):
            return BoolVal(node.value)
        if isinstance(node, QuantifierNode):
            return self._encode_quantifier(node, symbols)
        if isinstance(node, BooleanNode):
            return self._encode_boolean(node, symbols)
        if isinstance(node, RelationNode):
            return self._encode_relation(node, symbols)
        raise ValueError(f"Unknown AST node type: {type(node)}")

    def _encode_quantifier(self, node: QuantifierNode, symbols: dict):
        var = Const(node.variable, self.sort)
        new_symbols = dict(symbols)
        new_symbols[node.variable] = var
        body = self.encode(node.body, new_symbols)
        if node.quantifier == 'forall':
            return ForAll([var], body)
        return Exists([var], body)

    def _encode_boolean(self, node: BooleanNode, symbols: dict):
        children = [self.encode(c, symbols) for c in node.children]
        op = node.operator
        if op == 'not':
            return Not(children[0])
        if op == 'and':
            return And(*children)
        if op == 'or':
            return Or(*children)
        if op == 'implies':
            return Implies(children[0], children[1])
        if op == 'iff':
            return children[0] == children[1]
        if op == 'xor':
            return Xor(children[0], children[1])
        raise ValueError(f"Unknown boolean operator: {op!r}")

    def _encode_relation(self, node: RelationNode, symbols: dict):
        # Built-in equality: encode as Z3's native ==
        if node.name == '=':
            lhs, rhs = symbols[node.arguments[0]], symbols[node.arguments[1]]
            return lhs == rhs
        func = symbols[node.name]
        if not node.arguments:
            return func  # 0-arity: a Bool constant
        args = [symbols[arg] for arg in node.arguments]
        return func(*args)
