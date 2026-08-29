"""loaders.py — parsers for external source datasets.

These classes convert heterogeneous source formats (CSV, JSON, JSONL) into the
project's RawPair schema.  They are used during Stage 1 curation, before
any candidate formalization is generated.

For reading/writing *our own* standardized format, see io.py instead.
"""
from __future__ import annotations

import csv
import json
import sys
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .schema import RawPair


# Required schema field names that must be present after mapping
_REQUIRED_FIELDS = {'p', 'phi_gold', 'id_source_dataset'}


class BaseLoader(ABC):
    """Abstract base class for source-dataset loaders.

    Subclasses must implement load() and preview().  The column/key mapping
    passed at construction time insulates callers from differences in field
    naming across source datasets.
    """

    @abstractmethod
    def load(self, path: str) -> list[RawPair]:
        """Parse the file at *path* and return a list of RawPair objects.

        Args:
            path: Filesystem path to the source file.

        Returns:
            A list of RawPair instances populated from the source file.
        """
        pass

    @abstractmethod
    def preview(self, path: str, n: int = 5) -> list[dict]:
        """Return the first *n* raw records from *path* without any mapping.

        Useful for inspecting an unfamiliar source file before configuring a
        column/key map.

        Args:
            path: Filesystem path to the source file.
            n:    Number of records to return.

        Returns:
            A list of raw dicts as they appear in the file.
        """
        pass

    # ------------------------------------------------------------------ #
    # Shared helper                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_map(raw: dict, name_map: dict[str, str]) -> dict:
        """Remap keys of *raw* using *name_map* (source_key → schema_key)."""
        return {name_map[k]: v for k, v in raw.items() if k in name_map}

    @staticmethod
    def _to_bool(val) -> bool:
        """Convert a value to bool (handles 'true'/'false' strings)."""
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() == 'true'

    @staticmethod
    def _nullable(val) -> Optional[str]:
        if val is None or (isinstance(val, str) and val.strip() == ''):
            return None
        return val


class CSVLoader(BaseLoader):
    """Load a CSV source dataset into RawPair objects.

    Args:
        column_map:     Maps CSV column names → schema field names.
                        Required target keys: 'p', 'phi_gold', 'id_source_dataset'.
                        Optional target keys: 'ambiguity_flag', 'comment'.
                        Example: {"sentence": "p", "formula": "phi_gold", "idx": "id_source_dataset"}
        source_dataset: Name to record in RawPair.source_dataset for every row.
    """

    def __init__(self, column_map: dict[str, str], source_dataset: str) -> None:
        self.column_map = column_map
        self.source_dataset = source_dataset

    def load(self, path: str) -> list[RawPair]:
        """Read the CSV at *path* and return one RawPair per row.

        Rows where required fields are missing or empty are skipped with a
        warning printed to stderr.
        """
        pairs = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # start=2: row 1 is header
                mapped = self._apply_map(row, self.column_map)
                missing = _REQUIRED_FIELDS - set(mapped.keys())
                if missing:
                    print(
                        f"[CSVLoader] row {row_num}: skipping — missing required fields "
                        f"after mapping: {sorted(missing)}",
                        file=sys.stderr,
                    )
                    continue
                if any(not mapped.get(f, '').strip() for f in _REQUIRED_FIELDS):
                    print(
                        f"[CSVLoader] row {row_num}: skipping — empty required field(s)",
                        file=sys.stderr,
                    )
                    continue
                pairs.append(RawPair(
                    p=mapped['p'].strip(),
                    phi_gold=mapped['phi_gold'].strip(),
                    source_dataset=self.source_dataset,
                    id_source_dataset=mapped['id_source_dataset'].strip(),
                    ambiguity_flag=self._to_bool(mapped.get('ambiguity_flag', False)),
                    comment=self._nullable(mapped.get('comment')),
                ))
        return pairs

    def preview(self, path: str, n: int = 5) -> list[dict]:
        """Return the first *n* rows of the CSV as raw dicts (no mapping applied)."""
        rows = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(rows) >= n:
                    break
                rows.append(dict(row))
        return rows


class JSONLoader(BaseLoader):
    """Load a JSON source dataset into RawPair objects.

    Args:
        key_map:        Maps JSON key names → schema field names.
                        Required target keys: 'p', 'phi_gold', 'id_source_dataset'.
                        Optional target keys: 'ambiguity_flag', 'comment'.
        source_dataset: Name to record in RawPair.source_dataset.
        records_path:   Dot-separated path to the list of records inside the
                        JSON structure.  None means the top-level object is
                        already a list.
                        Example: "data.examples" for {"data": {"examples": [...]}}
    """

    def __init__(
        self,
        key_map: dict[str, str],
        source_dataset: str,
        records_path: Optional[str] = None,
    ) -> None:
        self.key_map = key_map
        self.source_dataset = source_dataset
        self.records_path = records_path

    def _get_records(self, data) -> list:
        """Navigate *data* using self.records_path and return the list of records."""
        if self.records_path is None:
            return data
        from utils.utility import safe_get
        records = safe_get(data, self.records_path)
        if records is None:
            raise ValueError(
                f"[JSONLoader] records_path '{self.records_path}' not found in JSON data"
            )
        return records

    def load(self, path: str) -> list[RawPair]:
        """Read the JSON file at *path* and return one RawPair per record.

        Uses safe_get() from utils.utility to navigate nested structures via
        records_path.  Records missing required keys are skipped with a warning.
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        records = self._get_records(data)

        pairs = []
        for rec_num, record in enumerate(records, start=1):
            mapped = self._apply_map(record, self.key_map)
            missing = _REQUIRED_FIELDS - set(mapped.keys())
            if missing:
                print(
                    f"[JSONLoader] record {rec_num}: skipping — missing required fields "
                    f"after mapping: {sorted(missing)}",
                    file=sys.stderr,
                )
                continue
            if any(not str(mapped.get(f, '')).strip() for f in _REQUIRED_FIELDS):
                print(
                    f"[JSONLoader] record {rec_num}: skipping — empty required field(s)",
                    file=sys.stderr,
                )
                continue
            pairs.append(RawPair(
                p=str(mapped['p']).strip(),
                phi_gold=str(mapped['phi_gold']).strip(),
                source_dataset=self.source_dataset,
                id_source_dataset=str(mapped['id_source_dataset']).strip(),
                ambiguity_flag=self._to_bool(mapped.get('ambiguity_flag', False)),
                comment=self._nullable(mapped.get('comment')),
            ))
        return pairs

    def preview(self, path: str, n: int = 5) -> list[dict]:
        """Return the first *n* raw records from the JSON (no mapping applied)."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        records = self._get_records(data)
        return list(records[:n])


class JSONLLoader(BaseLoader):
    """Load a JSONL (newline-delimited JSON) source dataset into RawPair objects.

    Each line of the file must be a self-contained JSON object.

    Args:
        key_map:           Maps JSON key names → schema field names.
                           Required target keys: 'p', 'phi_gold', 'id_source_dataset'.
                           Optional target keys: 'ambiguity_flag', 'comment', 'label'.
        source_dataset:    Name to record in RawPair.source_dataset.
        phi_gold_transform: Optional callable applied to the raw phi_gold value
                           *after* key mapping, before storing.  Use this when
                           phi_gold comes as a list (e.g. ``ground_truth: [...]``)
                           and you want to extract the first element:
                           ``phi_gold_transform=lambda v: v[0] if isinstance(v, list) else v``
    """

    def __init__(
        self,
        key_map: dict[str, str],
        source_dataset: str,
        phi_gold_transform: Optional[Callable] = None,
    ) -> None:
        self.key_map = key_map
        self.source_dataset = source_dataset
        self.phi_gold_transform = phi_gold_transform

    def load(self, path: str) -> list[RawPair]:
        """Read the JSONL file at *path* and return one RawPair per line.

        Empty lines and lines that produce records with missing required fields
        are skipped with a warning printed to stderr.
        """
        pairs = []
        with open(path, encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[JSONLLoader] line {line_num}: skipping — JSON parse error: {exc}",
                        file=sys.stderr,
                    )
                    continue

                mapped = self._apply_map(record, self.key_map)

                # Apply phi_gold_transform before required-field checks
                if 'phi_gold' in mapped and self.phi_gold_transform is not None:
                    mapped['phi_gold'] = self.phi_gold_transform(mapped['phi_gold'])

                missing = _REQUIRED_FIELDS - set(mapped.keys())
                if missing:
                    print(
                        f"[JSONLLoader] line {line_num}: skipping — missing required fields "
                        f"after mapping: {sorted(missing)}",
                        file=sys.stderr,
                    )
                    continue
                if any(not str(mapped.get(f, '')).strip() for f in _REQUIRED_FIELDS):
                    print(
                        f"[JSONLLoader] line {line_num}: skipping — empty required field(s)",
                        file=sys.stderr,
                    )
                    continue

                pairs.append(RawPair(
                    p=str(mapped['p']).strip(),
                    phi_gold=str(mapped['phi_gold']).strip(),
                    source_dataset=self.source_dataset,
                    id_source_dataset=str(mapped['id_source_dataset']).strip(),
                    ambiguity_flag=self._to_bool(mapped.get('ambiguity_flag', False)),
                    comment=self._nullable(mapped.get('comment')),
                    label=self._nullable(mapped.get('label')),
                ))
        return pairs

    def preview(self, path: str, n: int = 5) -> list[dict]:
        """Return the first *n* raw records from the JSONL (no mapping applied)."""
        records = []
        with open(path, encoding='utf-8') as f:
            for raw_line in f:
                if len(records) >= n:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
