# core/formalism — First-Order Logic Toolkit

## fol/

| File              | Purpose |
|-------------------|---------|
| `fol.py`          | `FOL` class: lazy `.tree` (AST), `.signature`, `.validate()`, `.to_smtlib()` |
| `parser.py`       | `FOLParser`: parse Unicode FOL strings → AST nodes; correct operator precedence |
| `ast.py`          | AST node definitions (`BinOp`, `UnaryOp`, `Quantifier`, `Predicate`, `Variable`, `Constant`) |
| `signature.py`    | Walk an AST to extract predicate / constant / variable signatures |
| `solver_z3.py`    | Z3-backed solver: `are_equivalent()`, `is_satisfiable()`, `implies()` (30 s timeout) |
| `encoder_z3.py`   | Encode FOL AST into Z3 SMT-LIB format |
| `translator.py`   | FOL transformations (e.g. signature renaming) |
| `perturbations.py`| Stub — LLM-based perturbations are not used in this project |

## FOL Syntax

Unicode only. ASCII alternatives (`->`, `&`, `|`, `~`) are rejected by `validate()`.

| Symbol | Meaning |
|--------|---------|
| `∀`, `∃` | Universal / existential quantifiers |
| `¬` | Negation |
| `∧`, `∨` | Conjunction, disjunction |
| `→`, `↔`, `⊕` | Implication, biconditional, XOR |
| `Predicate(x)` | Predicate (capitalized name) |
| `x`, `y` | Variables (lowercase) |
| `A`, `B` | Constants (capitalized) |

Operator precedence (tightest first): `¬` > `∧` > `∨` > `→` > `⊕` > `↔`.
