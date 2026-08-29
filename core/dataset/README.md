# core/dataset — Datasets and Loaders

## Modules

| File                   | Purpose |
|------------------------|---------|
| `schema.py`            | `RawPair` dataclass: `(p, phi_gold, phi_hat, source, ambiguity)` |
| `loaders.py`           | Functions to load FOLIO / MALLS raw pairs from JSONL |
| `signature_loader.py`  | Load FOL predicate signature (predicate name → arity) per dataset |
| `datasets/`            | Raw dataset files; see [`datasets/README.md`](datasets/README.md) |
