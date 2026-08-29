from .ast import Node, QuantifierNode, BooleanNode, RelationNode, BoolConstNode


class FOLSignature:
    """
    Extracts non-logical symbols from an AST.
    Output format:
        {
            'Rel': {'Human': 1, 'Loves': 2},
            'Var': {'x', 'y'},
            'Const': {'John'}
        }

    Convention:
        - Rel: relation name → arity
        - Var: lowercase argument names that are quantifier-bound
        - Const: uppercase argument names, or unbound lowercase names
    """

    def extract(self, tree: Node) -> dict:
        sig = {'Rel': {}, 'Var': set(), 'Const': set()}
        self._walk(tree, sig, bound=set())
        return sig

    def _walk(self, node: Node, sig: dict, bound: set) -> None:
        if isinstance(node, BoolConstNode):
            return  # ⊤/⊥ contribute no symbols to the signature
        if isinstance(node, QuantifierNode):
            new_bound = bound | {node.variable}
            self._walk(node.body, sig, new_bound)
        elif isinstance(node, BooleanNode):
            for child in node.children:
                self._walk(child, sig, bound)
        elif isinstance(node, RelationNode):
            # '=' is the built-in equality predicate; don't add it to user-defined Rel.
            if node.name != '=':
                sig['Rel'][node.name] = len(node.arguments)
            for arg in node.arguments:
                if arg[0].isupper():
                    sig['Const'].add(arg)
                elif arg in bound:
                    sig['Var'].add(arg)
                else:
                    # Free (unbound) lowercase name: treat as constant term
                    sig['Const'].add(arg)

    def merge(self, sig1: dict, sig2: dict) -> dict:
        """
        Merge two signatures.
        Raise ValueError if the same relation name has conflicting arities.
        """
        merged = {
            'Rel': dict(sig1['Rel']),
            'Var': sig1['Var'] | sig2['Var'],
            'Const': sig1['Const'] | sig2['Const'],
        }
        for name, arity in sig2['Rel'].items():
            if name in merged['Rel'] and merged['Rel'][name] != arity:
                raise ValueError(
                    f"Arity conflict for relation '{name}': "
                    f"{merged['Rel'][name]} vs {arity}"
                )
            merged['Rel'][name] = arity
        return merged
