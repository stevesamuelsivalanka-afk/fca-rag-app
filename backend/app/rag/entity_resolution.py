from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable


LEGAL_SUFFIXES = {
    "plc",
    "limited",
    "ltd",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
}

GENERIC_PREFIXES = {"the", "a", "an", "bank", "financial", "company", "group"}


def normalize_entity_name(value: object) -> str:
    """Normalize punctuation and accents without discarding entity words."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", text))


def resolve_entity_candidates(
    question: str,
    canonical_names: Iterable[str],
) -> list[str]:
    """Return exact names or exact leading-name candidates, never fuzzy filters."""
    query = f" {normalize_entity_name(question)} "
    names = sorted(
        {str(name).strip() for name in canonical_names if str(name).strip()},
        key=lambda name: (-len(normalize_entity_name(name)), name.casefold()),
    )

    exact = [
        name
        for name in names
        if f" {normalize_entity_name(name)} " in query
    ]
    if exact:
        return exact

    candidates: set[str] = set()
    for name in names:
        tokens = normalize_entity_name(name).split()
        core_tokens = [
            token
            for token in tokens
            if token not in LEGAL_SUFFIXES
        ]
        for size in range(1, len(core_tokens) + 1):
            prefix = " ".join(core_tokens[:size])
            if (
                len(prefix) >= 3
                and not (size == 1 and prefix in GENERIC_PREFIXES)
                and f" {prefix} " in query
            ):
                candidates.add(name)

    return sorted(
        candidates,
        key=lambda name: (-len(normalize_entity_name(name)), name.casefold()),
    )


def fuzzy_entity_candidates(
    query: str,
    canonical_names: Iterable[str],
    limit: int = 5,
) -> list[str]:
    """Produce suggestions only; callers must not use these as retrieval filters."""
    normalized_query = normalize_entity_name(query)
    if len(normalized_query) < 4:
        return []

    ranked = sorted(
        (
            SequenceMatcher(
                None,
                normalized_query,
                normalize_entity_name(name),
            ).ratio(),
            name,
        )
        for name in set(canonical_names)
    )
    return [
        name
        for score, name in reversed(ranked)
        if score >= 0.72
    ][:limit]