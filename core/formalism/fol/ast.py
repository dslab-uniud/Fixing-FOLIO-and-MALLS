class Node:
    """Base class for AST nodes."""
    pass


class QuantifierNode(Node):
    """
    Represents ∀x φ or ∃x φ
    attributes:
        quantifier: 'forall' | 'exists'
        variable: str
        body: Node
    """

    def __init__(self, quantifier: str, variable: str, body: 'Node'):
        self.quantifier = quantifier
        self.variable = variable
        self.body = body


class BooleanNode(Node):
    """
    Represents boolean operators.
    attributes:
        operator: 'and' | 'or' | 'implies' | 'iff' | 'xor' | 'not'
        children: list[Node]
    """

    def __init__(self, operator: str, children: list):
        self.operator = operator
        self.children = children


class RelationNode(Node):
    """
    Represents R(t1,...,tn)
    attributes:
        name: str
        arguments: list[str]
    """

    def __init__(self, name: str, arguments: list):
        self.name = name
        self.arguments = arguments


class BoolConstNode(Node):
    """
    Represents a boolean constant: ⊤ / True (tautology) or ⊥ / False (contradiction).
    attributes:
        value: bool  — True for ⊤/True, False for ⊥/False
    """

    def __init__(self, value: bool):
        self.value = value
