"""signature_loader.py — Load and format FOL signatures for prompt filling.

Provides helpers to read the per-story FOLIO signature.jsonl and the
per-instance MALLS signature.jsonl, and to render their contents as the
vocabulary strings that fill the ``{relations}`` and ``{constants}``
placeholders in P1/P2 prompts.

Usage::

    from core.dataset.signature_loader import (
        get_story_id_from_instance_id,
        load_folio_signature,
        format_folio_vocabulary,
        load_malls_signature,
    )

    # FOLIO
    story_id = get_story_id_from_instance_id("concl_380_1")  # → 380
    sig = load_folio_signature(story_id)
    rels, consts = format_folio_vocabulary(sig)

    # MALLS (per-instance, keyed by integer id)
    sig = load_malls_signature(5)
    rels, consts = format_folio_vocabulary(sig)  # same Rel/Const schema as FOLIO
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_FOLIO_SIG_PATH     = _HERE / "datasets/folio/signature.jsonl"
_FOLIO_OLD_SIG_PATH = _HERE / "datasets/folio/signature_old.jsonl"
_MALLS_SIG_PATH     = _HERE / "datasets/malls/signature.jsonl"
_MALLS_OLD_SIG_PATH = _HERE / "datasets/malls/signature_old.jsonl"

# ---------------------------------------------------------------------------
# Module-level caches (lazy-loaded once per process)
# ---------------------------------------------------------------------------

_folio_cache:     dict[int, dict] = {}
_folio_old_cache: dict[int, dict] = {}
_malls_cache:     dict[int, dict] = {}
_malls_old_cache: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Instance-id helpers
# ---------------------------------------------------------------------------

def get_story_id_from_instance_id(instance_id: str) -> int:
    """Return the integer story_id embedded in a FOLIO instance id.

    FOLIO instance ids have two forms:
    * ``story_<N>``          → story_id = N
    * ``concl_<N>_<M>``     → story_id = N

    Args:
        instance_id: FOLIO instance identifier string.

    Returns:
        Integer story_id.

    Examples:
        >>> get_story_id_from_instance_id("story_380")
        380
        >>> get_story_id_from_instance_id("concl_380_1")
        380
    """
    parts = instance_id.split("_")
    return int(parts[1])


# ---------------------------------------------------------------------------
# FOLIO signature loading
# ---------------------------------------------------------------------------

def _load_folio_all() -> dict[int, dict]:
    """Load *all* FOLIO story signatures from signature.jsonl (cached).

    signature.jsonl may contain pretty-printed (multi-line) JSON objects
    concatenated without delimiters, so we use raw_decode to stream-parse.
    """
    global _folio_cache
    if _folio_cache:
        return _folio_cache
    decoder = json.JSONDecoder()
    cache: dict[int, dict] = {}
    text = _FOLIO_SIG_PATH.read_text(encoding="utf-8")
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\n\r":
            idx += 1
        if idx >= n:
            break
        rec, end_idx = decoder.raw_decode(text, idx)
        cache[rec["story_id"]] = rec
        idx = end_idx
    _folio_cache = cache
    return _folio_cache


def load_folio_signature(story_id: int) -> dict:
    """Return the signature dict for a FOLIO story.

    The returned dict has keys ``'Rel'`` and ``'Const'``.

    Args:
        story_id: Integer story identifier.

    Returns:
        Signature dict for the given story.

    Raises:
        KeyError: If *story_id* is not present in signature.jsonl.
    """
    return _load_folio_all()[story_id]


def _load_folio_old_all() -> dict[int, dict]:
    """Load *all* FOLIO old-signature entries from signature_old.jsonl (cached)."""
    global _folio_old_cache
    if _folio_old_cache:
        return _folio_old_cache
    decoder = json.JSONDecoder()
    cache: dict[int, dict] = {}
    text = _FOLIO_OLD_SIG_PATH.read_text(encoding="utf-8")
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\n\r":
            idx += 1
        if idx >= n:
            break
        rec, end_idx = decoder.raw_decode(text, idx)
        cache[rec["story_id"]] = rec
        idx = end_idx
    _folio_old_cache = cache
    return _folio_old_cache


def load_folio_old_signature(story_id: int) -> dict:
    """Return the *old* signature dict for a FOLIO story (from signature_old.jsonl).

    Identical schema to :func:`load_folio_signature` but uses the pre-curation
    predicate/constant vocabulary.

    Args:
        story_id: Integer story identifier.

    Returns:
        Old signature dict for the given story.

    Raises:
        KeyError: If *story_id* is not present in signature_old.jsonl.
    """
    return _load_folio_old_all()[story_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VAR_NAMES = ["x", "y", "z", "u", "v", "w"]


def _arg_list(arity: int) -> str:
    """Return a comma-separated variable list for the given arity."""
    return ", ".join(_VAR_NAMES[:arity])


def _fill_meaning(template: str, arity: int) -> str:
    """Fill positional placeholders {0}, {1}, ... in a meaning template.

    Uses at least *arity* variables; expands further if the template references
    higher indices (e.g. optional extra arguments noted in the signature).
    """
    import re
    indices = [int(m) for m in re.findall(r"\{(\d+)\}", template)]
    n_vars = max(arity, max(indices) + 1) if indices else arity
    return template.format(*_VAR_NAMES[:n_vars])


# ---------------------------------------------------------------------------
# FOLIO vocabulary formatting
# ---------------------------------------------------------------------------

def format_folio_vocabulary(signature: dict) -> tuple[str, str]:
    """Format a FOLIO story signature into (relations_str, constants_str).

    Relations string example::

        - InThisClub(x): x is in this club
        - PerformOftenIn(x, y): x performs often in y

    Constants string example::

        - bonnie: Bonnie
        - schoolTalentShow: the school talent show

    Args:
        signature: Dict returned by :func:`load_folio_signature`.

    Returns:
        ``(relations_str, constants_str)`` tuple ready to be injected into a
        prompt template.
    """
    rel_lines: list[str] = []
    for name, info in signature["Rel"].items():
        raw_arity = info["arity"]
        arity: int = int(raw_arity) if isinstance(raw_arity, int) else int(str(raw_arity).split()[0])
        meaning: str = _fill_meaning(info["pos_meaning"], arity)
        rel_lines.append(f"- {name}({_arg_list(arity)}): {meaning}")

    const_lines: list[str] = []
    for name, meaning in signature["Const"].items():
        const_lines.append(f"- {name}: {meaning}")

    return "\n".join(rel_lines), "\n".join(const_lines)


# ---------------------------------------------------------------------------
# MALLS signature loading (per-instance, same Rel/Const schema as FOLIO)
# ---------------------------------------------------------------------------

def _load_malls_all() -> dict[int, dict]:
    """Load *all* MALLS per-instance signatures from signature.jsonl (cached).

    signature.jsonl uses a multi-line concatenated JSON format (same as FOLIO).
    The file may also contain commas or array brackets between objects; the
    decoder skips those.
    """
    global _malls_cache
    if _malls_cache:
        return _malls_cache
    decoder = json.JSONDecoder(strict=False)
    cache: dict[int, dict] = {}
    content = _MALLS_SIG_PATH.read_text(encoding="utf-8").strip()
    idx, n = 0, len(content)
    while idx < n:
        while idx < n and (content[idx].isspace() or content[idx] in ",[]"):
            idx += 1
        if idx >= n:
            break
        rec, end_idx = decoder.raw_decode(content, idx)
        cache[rec["id"]] = rec
        idx = end_idx
    _malls_cache = cache
    return _malls_cache


def load_malls_signature(instance_id: int) -> dict:
    """Return the signature dict for a MALLS instance.

    The returned dict has keys ``'Rel'`` and ``'Const'`` — identical schema to
    FOLIO, so :func:`format_folio_vocabulary` can be used directly.

    Args:
        instance_id: Integer MALLS instance id (1–100).

    Returns:
        Signature dict for the given instance.

    Raises:
        KeyError: If *instance_id* is not present in signature.jsonl.
    """
    return _load_malls_all()[instance_id]


def _load_malls_old_all() -> dict[int, dict]:
    """Load *all* MALLS old-signature entries from signature_old.jsonl (cached)."""
    global _malls_old_cache
    if _malls_old_cache:
        return _malls_old_cache
    decoder = json.JSONDecoder(strict=False)
    cache: dict[int, dict] = {}
    content = _MALLS_OLD_SIG_PATH.read_text(encoding="utf-8").strip()
    idx, n = 0, len(content)
    while idx < n:
        while idx < n and (content[idx].isspace() or content[idx] in ",[]"):
            idx += 1
        if idx >= n:
            break
        rec, end_idx = decoder.raw_decode(content, idx)
        cache[rec["id"]] = rec
        idx = end_idx
    _malls_old_cache = cache
    return _malls_old_cache


def load_malls_old_signature(instance_id: int) -> dict:
    """Return the *old* signature dict for a MALLS instance (from signature_old.jsonl).

    Identical schema to :func:`load_malls_signature` but uses the pre-curation
    predicate/constant vocabulary.

    Args:
        instance_id: Integer MALLS instance id (1–100).

    Returns:
        Old signature dict for the given instance.

    Raises:
        KeyError: If *instance_id* is not present in signature_old.jsonl.
    """
    return _load_malls_old_all()[instance_id]


