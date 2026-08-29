from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional


# ---------------------------------------------------------------------------
# Valid values for controlled fields
# ---------------------------------------------------------------------------

VALID_LABELS = {'yes', 'no', 'timeout', 'uncertain'}
VALID_LABEL_METHODS = {'z3', 'human'}
VALID_ERROR_TYPES = {'predicate', 'argument', 'quantifier', 'connective', 'mistranslation'}
VALID_PROMPT_IDS = {'P1', 'P2', 'manual'}


# ---------------------------------------------------------------------------
# RawPair — Stage 1 intermediate
# ---------------------------------------------------------------------------

@dataclass
class RawPair:
    """A verified (p, φ_gold) pair from a source dataset.

    Produced during Stage 1 curation, before any candidate formalization is
    generated. Ambiguous sentences are retained with ambiguity_flag=True.
    """

    p: str
    """Natural language sentence."""

    phi_gold: str
    """Ground-truth FOL formalization."""

    source_dataset: str
    """Name of the original dataset this pair was drawn from."""

    id_source_dataset: str
    """Original index or identifier within source_dataset."""

    ambiguity_flag: bool = False
    """True if p admits multiple non-equivalent FOL readings."""

    comment: Optional[str] = None
    """Free-text note; may contain keywords [AMBIGUITY], [PARTIAL]."""

    label: Optional[str] = None
    """Source-dataset quality label ('yes' / 'no') for phi_gold, when available."""


# ---------------------------------------------------------------------------
# Instance — full dataset schema
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """One fully labelled dataset instance, as defined in dataset.md.

    An Instance is produced in Stage 2 by pairing a RawPair with a candidate
    formalization φ_hat and running Z3 (or human annotation) to assign a label.
    """

    id: str
    """Unique instance identifier."""

    p: str
    """Natural language sentence."""

    phi_gold: str
    """Ground-truth FOL formalization φ."""

    phi_hat: str
    """Candidate FOL formalization φ̂."""

    label: str
    """Correctness verdict: 'yes', 'no', 'timeout', or 'uncertain'."""

    label_method: str
    """How the label was assigned: 'z3' or 'human'."""

    error_type: Optional[str]
    """Coarse error category for 'no' instances (see VALID_ERROR_TYPES); None otherwise."""

    source_dataset: str
    """Name of the original dataset p was drawn from."""

    id_source_dataset: str
    """Original index in source_dataset."""

    generator_model: Optional[str]
    """Model that produced φ̂ (or 'human' for manually constructed instances)."""

    generator_prompt_id: Optional[str]
    """Identifier of the generation prompt: 'P1', 'P2', or 'manual'."""

    ambiguity_flag: bool
    """True if p was flagged as ambiguous during curation."""

    comment: Optional[str]
    """Free-text comment; may contain keywords [AMBIGUITY], [PARTIAL], [TIMEOUT]."""

    def to_dict(self) -> dict:
        """Serialize to a plain dict for interop with JudgeMethod / Metric layers."""
        return {
            'id': self.id,
            'p': self.p,
            'phi_gold': self.phi_gold,
            'phi_hat': self.phi_hat,
            'label': self.label,
            'label_method': self.label_method,
            'error_type': self.error_type,
            'source_dataset': self.source_dataset,
            'id_source_dataset': self.id_source_dataset,
            'generator_model': self.generator_model,
            'generator_prompt_id': self.generator_prompt_id,
            'ambiguity_flag': self.ambiguity_flag,
            'comment': self.comment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Instance':
        """Deserialize from a plain dict (e.g. loaded from CSV/JSON).

        Empty strings are treated as None for nullable fields.
        The string 'True'/'False' is converted to bool for ambiguity_flag.
        """
        def _nullable(val):
            return val if val not in (None, '') else None

        def _to_bool(val):
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() == 'true'

        return cls(
            id=d['id'],
            p=d['p'],
            phi_gold=d['phi_gold'],
            phi_hat=d['phi_hat'],
            label=d['label'],
            label_method=d['label_method'],
            error_type=_nullable(d.get('error_type')),
            source_dataset=d['source_dataset'],
            id_source_dataset=d['id_source_dataset'],
            generator_model=_nullable(d.get('generator_model')),
            generator_prompt_id=_nullable(d.get('generator_prompt_id')),
            ambiguity_flag=_to_bool(d.get('ambiguity_flag', False)),
            comment=_nullable(d.get('comment')),
        )


# ---------------------------------------------------------------------------
# Dataset — container with named splits
# ---------------------------------------------------------------------------

class Dataset:
    """Container for a collection of Instance objects.

    Provides named splits (working, working_z3, …), generic filtering,
    and basic statistics.  All split properties return a new Dataset so that
    the full pipeline can be written as:

        ds.working.filter(is_negative)
    """

    def __init__(self, instances: list[Instance]) -> None:
        self._instances: list[Instance] = list(instances)

    # ------------------------------------------------------------------ #
    # Container protocol                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._instances)

    def __iter__(self) -> Iterator[Instance]:
        return iter(self._instances)

    def __getitem__(self, idx: int) -> Instance:
        return self._instances[idx]

    # ------------------------------------------------------------------ #
    # Named splits (as defined in dataset.md)                             #
    # ------------------------------------------------------------------ #

    @property
    def full(self) -> 'Dataset':
        """All instances, including ambiguous and timeout/uncertain."""
        return Dataset(self._instances)

    @property
    def working(self) -> 'Dataset':
        """ambiguity_flag=False AND label in {'yes', 'no'}."""
        return Dataset([
            inst for inst in self._instances
            if not inst.ambiguity_flag and inst.label in {'yes', 'no'}
        ])

    @property
    def working_z3(self) -> 'Dataset':
        """Subset of working where label_method='z3'."""
        return Dataset([
            inst for inst in self.working
            if inst.label_method == 'z3'
        ])

    @property
    def working_human(self) -> 'Dataset':
        """Subset of working where label_method='human'."""
        return Dataset([
            inst for inst in self.working
            if inst.label_method == 'human'
        ])

    @property
    def ambiguous(self) -> 'Dataset':
        """Instances where ambiguity_flag=True."""
        return Dataset([inst for inst in self._instances if inst.ambiguity_flag])

    # ------------------------------------------------------------------ #
    # Filtering                                                            #
    # ------------------------------------------------------------------ #

    def filter(self, predicate: Callable[[Instance], bool]) -> 'Dataset':
        """Return a new Dataset containing only instances satisfying predicate."""
        return Dataset([inst for inst in self._instances if predicate(inst)])

    # ------------------------------------------------------------------ #
    # Conversion                                                           #
    # ------------------------------------------------------------------ #

    def to_list(self) -> list[Instance]:
        """Return a copy of the underlying instance list."""
        return list(self._instances)

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Return aggregate counts useful for reporting.

        Returned dict includes at minimum:
          - 'total': int
          - 'by_label': dict[str, int]
          - 'by_label_method': dict[str, int]
          - 'by_generator_model': dict[str, int]
          - 'by_error_type': dict[str, int]
          - 'ambiguous': int
          - 'needs_annotation': int  (label in {timeout, uncertain} or missing error_type)
        """
        by_label = Counter(inst.label for inst in self._instances)
        by_label_method = Counter(inst.label_method for inst in self._instances)
        by_generator = Counter(
            inst.generator_model for inst in self._instances if inst.generator_model
        )
        by_error_type = Counter(
            inst.error_type for inst in self._instances if inst.error_type
        )
        ambiguous_count = sum(1 for inst in self._instances if inst.ambiguity_flag)
        needs_ann = sum(
            1 for inst in self._instances
            if inst.label in {'timeout', 'uncertain'}
            or (inst.label == 'no' and inst.error_type is None)
        )
        return {
            'total': len(self._instances),
            'by_label': dict(by_label),
            'by_label_method': dict(by_label_method),
            'by_generator_model': dict(by_generator),
            'by_error_type': dict(by_error_type),
            'ambiguous': ambiguous_count,
            'needs_annotation': needs_ann,
        }
