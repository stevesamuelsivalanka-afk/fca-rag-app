from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.rag.entity_resolution import (
    fuzzy_entity_candidates,
    normalize_entity_name,
    resolve_entity_candidates,
)
from app.rag.store import VectorStore


# ============================================================
# CONSTANTS
# ============================================================

YEAR_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\b"
)

STOP_WORDS = {
    "the", "a", "an", "and", "or", "was", "were",
    "is", "are", "to", "of", "in", "on", "for",
    "with", "what", "why", "how", "did", "does",
    "do", "this", "that", "these", "those",
    "which", "who", "when", "where", "from",
    "into", "their", "his", "her", "its", "about",
    "against", "between", "over", "under", "during",
    "than", "have", "has", "had", "been", "being",
    "can", "could", "would", "should", "tell",
    "me", "please",
}

ENFORCEMENT_TERMS = {
    "fine",
    "fined",
    "penalty",
    "penalties",
    "penalised",
    "penalized",
    "breach",
    "breaches",
    "breached",
    "failings",
    "failure",
    "failures",
    "conduct",
    "rule",
    "rules",
    "principle",
    "principles",
    "listing",
    "market abuse",
    "systems and controls",
    "controls",
    "governance",
    "monitoring",
    "risk",
    "requirements",
    "decision notice",
    "final notice",
    "tribunal",
    "upper tribunal",
    "senior manager",
    "misconduct",
    "enforcement",
    "regulatory",
}

COMPARISON_TERMS = {
    "compare",
    "comparison",
    "versus",
    "vs",
    "difference",
    "differences",
    "between",
    "both",
    "differ",
    "compared",
}

COMMON_ISSUE_TERMS = {
    "common",
    "commonly",
    "most common",
    "recurring",
    "frequent",
    "frequently",
    "typical",
    "usual",
    "issues",
    "problems",
    "themes",
    "patterns",
}

LARGEST_TERMS = {
    "largest fine",
    "largest penalty",
    "biggest fine",
    "biggest penalty",
    "highest fine",
    "highest penalty",
    "largest fines",
    "biggest fines",
    "highest fines",
}

SMALLEST_TERMS = {
    "smallest fine",
    "smallest penalty",
    "lowest fine",
    "lowest penalty",
}

COUNT_TERMS = {
    "how many",
    "number of fines",
    "number of penalties",
    "count of fines",
    "count of penalties",
    "how many firms",
    "how many companies",
}

TOTAL_TERMS = {
    "total fine",
    "total fines",
    "total penalty",
    "total penalties",
    "combined fine",
    "combined fines",
    "sum of fines",
}

AVERAGE_TERMS = {"average fine", "average fines", "mean fine", "mean fines"}

WHY_TERMS = {
    "why",
    "reason",
    "reasons",
    "what was the reason",
    "why was",
    "why were",
}

TRIBUNAL_TERMS = {
    "tribunal",
    "upper tribunal",
    "decision notice",
    "decision notices",
}

RULE_TERMS = {
    "rule",
    "rules",
    "principle",
    "principles",
    "conduct rule",
    "conduct rules",
    "breach",
    "breaches",
    "breached",
    "requirements",
    "depp",
    "smcr",
    "icr",
}

DATE_TERMS = {
    "when",
    "date",
    "dated",
    "issued",
    "publication date",
}

PERSON_TERMS = {
    "person",
    "individual",
    "director",
    "senior manager",
    "chief executive",
    "ceo",
    "manager",
}


# ============================================================
# QUERY ANALYSIS
# ============================================================

@dataclass
class QueryAnalysis:
    question: str
    entities: list[str]
    ambiguous_entities: list[str]
    years: list[int]
    comparison: bool
    common_issues: bool
    intent: str
    entity_type: str = "unknown"
    comparison_entities: list[str] = field(default_factory=list)
    comparison_years: list[int] = field(default_factory=list)
    aggregation: str | None = None
    topic: str = ""
    requested_fields: list[str] = field(default_factory=list)
    unresolved_entities: list[str] = field(default_factory=list)
    top_n: int | None = None


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value: Any) -> str:
    return normalize_entity_name(value)


def tokenize(value: Any) -> set[str]:

    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            normalize(value),
        )
        if token not in STOP_WORDS
    }


# ============================================================
# ENTITY CATALOGUE
# ============================================================

def build_entity_catalogue(
    metadata: list[dict],
) -> list[str]:

    entities: dict[str, str] = {}

    for item in metadata:
        firm = str(item.get("firm") or item.get("firm_normalized") or "").strip()
        normalized = normalize(firm)
        if len(normalized) >= 3:
            entities.setdefault(normalized, firm)

    return sorted(
        entities.values(),
        key=lambda value: (
            -len(tokenize(value)),
            -len(value),
        ),
    )


# ============================================================
# YEARS
# ============================================================

def extract_years(
    question: str,
) -> list[int]:

    years = []

    for value in YEAR_RE.findall(
        question
    ):

        year = int(value)

        if 1900 <= year <= 2100:
            years.append(year)

    return sorted(
        set(years)
    )


# ============================================================
# ENTITIES
# ============================================================

def exact_entity_matches(
    question: str,
    catalogue: list[str],
) -> list[str]:

    return resolve_entity_candidates(
        question,
        catalogue,
    )


def fuzzy_entity_matches(
    question: str,
    catalogue: list[str],
) -> list[str]:

    return fuzzy_entity_candidates(
        question,
        catalogue,
    )


def extract_entities(
    question: str,
    catalogue: list[str],
) -> list[str]:

    exact = exact_entity_matches(
        question,
        catalogue,
    )

    if exact:
        return exact[:5]

    return []


# ============================================================
# ENTITY MATCHING
# ============================================================

def metadata_entity_matches(
    metadata: dict,
    entity: str,
) -> bool:

    actual = normalize_entity_name(
        metadata.get(
            "firm_normalized",
            metadata.get(
                "firm",
                "",
            ),
        )
    )

    return bool(actual) and actual == normalize_entity_name(entity)


# ============================================================
# INTENT
# ============================================================

def detect_comparison(
    question: str,
    entities: list[str],
) -> bool:

    normalized = normalize(
        question
    )

    return any(
        re.search(rf"\b{re.escape(term)}\b", normalized)
        for term in COMPARISON_TERMS
    )


def detect_common_issues(
    question: str,
) -> bool:

    normalized = normalize(
        question
    )
    corpus_terms = {
        "common", "commonly", "most common", "recurring", "frequent",
        "frequently", "typical", "usual", "themes", "patterns",
    }
    return any(term in normalized for term in corpus_terms)


def detect_intent(
    question: str,
) -> str:

    normalized = normalize(
        question
    )

    if re.search(r"\b(?:total|sum)\b", normalized) and re.search(
        r"\b(?:largest|biggest|highest|top)\b", normalized
    ):
        return "TOP_N_TOTAL"

    if any(term in normalized for term in AVERAGE_TERMS):
        return "AVERAGE"

    if re.search(r"\btop\s+\d+\b", normalized) or re.search(
        r"\b(?:list|show|give)\s+(?:the\s+)?(?:\d+|five|three|ten)\s+(?:largest|biggest|highest)",
        normalized,
    ):
        return "TOP_N"

    if any(term in normalized for term in LARGEST_TERMS):
        return "LARGEST_FINE"

    if any(term in normalized for term in SMALLEST_TERMS):
        return "SMALLEST_FINE"

    if any(term in normalized for term in TOTAL_TERMS):
        return "TOTAL"

    if any(term in normalized for term in COUNT_TERMS):
        return "COUNT"

    if detect_common_issues(question):
        return "COMMON_ISSUES"

    if detect_comparison(question, []):
        return "COMPARISON"

    if any(
        term in normalized
        for term in TRIBUNAL_TERMS
    ):
        return "FACT"

    if "principle" in normalized or "principles" in normalized:
        return "PRINCIPLES"

    if any(term in normalized for term in WHY_TERMS) or re.search(
        r"\b(caused|cause|led to|resulted in)\b",
        normalized,
    ):
        return "WHY_FINE"

    if any(term in normalized for term in DATE_TERMS):
        return "DATE"

    if any(term in normalized for term in ("how much", "amount", "what fine", "fine amount")):
        return "FINE_AMOUNT"

    if any(term in normalized for term in RULE_TERMS):
        return "BREACH"

    if any(term in normalized for term in ("history", "happened", "what happened")):
        return "FIRM_HISTORY"

    if any(term in normalized for term in ("source", "document", "notice", "pdf")):
        return "SOURCE_LOOKUP"

    if any(
        term in normalized
        for term in PERSON_TERMS
    ):
        return "FACT"

    return "FACT" if any(term in normalized for term in ("fine", "fined", "penalty")) else "GENERAL_SEARCH"


def analyze_question(
    question: str,
    catalogue: list[str],
) -> QueryAnalysis:

    question = str(
        question or ""
    ).strip()

    years = extract_years(
        question
    )
    range_match = re.search(
        r"\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b",
        question,
        re.IGNORECASE,
    )
    if range_match:
        first_year, last_year = map(int, range_match.groups())
        if first_year <= last_year:
            years = list(range(first_year, last_year + 1))

    entities = extract_entities(
        question,
        catalogue,
    )

    comparison = detect_comparison(
        question,
        entities,
    )

    common_issues = detect_common_issues(
        question
    )

    ambiguous_entities = []

    ambiguous_entities = entities if len(entities) > 1 and not comparison else []
    ignored_capitalized = {
        "compare", "which", "what", "why", "when", "where", "how",
        "tell", "does", "did", "can", "could", "would", "should",
        "fca", "final", "notice", "year", "firm", "bank", "banks",
    }
    capitalized_runs = re.findall(
        r"\b(?:[A-Z][a-z][A-Za-z0-9&'.-]*|[A-Z]{2,})(?:\s+(?:[A-Z][a-z][A-Za-z0-9&'.-]*|[A-Z]{2,}|&))*\b",
        question,
    )
    known_normalized = [normalize_entity_name(entity) for entity in entities]
    unresolved_entities = []
    for mention in capitalized_runs:
        normalized_mention = normalize_entity_name(mention)
        if not normalized_mention or normalized_mention.split()[0] in ignored_capitalized:
            continue
        if any(
            known == normalized_mention or known.startswith(f"{normalized_mention} ")
            for known in known_normalized
        ):
            continue
        if len(normalized_mention.split()) > 1 or mention.isupper():
            unresolved_entities.append(mention)

    if not entities:
        entity_led_patterns = (
            r"\bwhy\s+(?:was|were)\s+(.+?)\s+(?:fined|penalised|penalized|sanctioned)\b",
            r"\bhow\s+much\s+(?:was|were)\s+(.+?)\s+(?:fined|penalised|penalized)\b",
            r"\bwhat\s+happened\s+to\s+(.+?)(?:\s+in\s+20\d{2}|[?.!]|$)",
            r"\bwhich\s+principles\s+did\s+(.+?)\s+breach\b",
        )
        for pattern in entity_led_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if not match:
                continue
            mention = match.group(1).strip(" .,!?:;")
            normalized_mention = normalize_entity_name(mention)
            generic_mentions = {
                "a firm", "the firm", "banks", "a bank", "the bank",
                "any firm", "any bank", "firms", "companies",
            }
            if normalized_mention and normalized_mention not in generic_mentions:
                unresolved_entities.append(mention)
            break

    intent = detect_intent(question)
    aggregation = {
        "LARGEST_FINE": "largest",
        "SMALLEST_FINE": "smallest",
        "COUNT": "count",
        "TOTAL": "total",
        "TOP_N": "top_n",
        "TOP_N_TOTAL": "top_n_total",
        "AVERAGE": "average",
    }.get(intent)
    normalized_question = normalize(question)
    top_match = re.search(r"\btop\s+(\d+)\b", normalized_question)
    if top_match:
        top_n = int(top_match.group(1))
    else:
        word_counts = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        word_match = re.search(
            r"\b(" + "|".join(word_counts) + r")\b",
            normalized_question,
        )
        top_n = word_counts[word_match.group(1)] if word_match else None
    lowered = normalize(question)
    entity_type = (
        "bank" if re.search(r"\bbanks\b", lowered)
        else "person" if any(term in lowered for term in PERSON_TERMS)
        else "firm" if entities
        else "unknown"
    )
    requested_fields = {
        "WHY_FINE": ["fine", "reason", "breaches", "principles", "findings"],
        "FINE_AMOUNT": ["fine", "amount", "year", "source"],
        "DATE": ["date", "source"],
        "BREACH": ["breaches", "rules", "findings"],
        "PRINCIPLES": ["principles", "source"],
        "COMPARISON": ["firm", "year", "fine", "reason", "breaches"],
        "COMMON_ISSUES": ["reason", "breaches", "findings"],
        "COUNT": ["firm", "year", "amount", "source"],
        "LARGEST_FINE": ["firm", "year", "amount", "source"],
        "SMALLEST_FINE": ["firm", "year", "amount", "source"],
        "TOTAL": ["year", "amount", "source"],
    }.get(intent, ["firm", "year", "source"])

    return QueryAnalysis(
        question=question,
        entities=entities,
        ambiguous_entities=ambiguous_entities,
        years=years,
        comparison=comparison,
        common_issues=common_issues,
        intent=intent,
        entity_type=entity_type,
        comparison_entities=entities if comparison else [],
        comparison_years=years if comparison else [],
        aggregation=aggregation,
        topic=" ".join(sorted(tokenize(question))),
        requested_fields=requested_fields,
        unresolved_entities=sorted(set(unresolved_entities)),
        top_n=top_n,
    )


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_question(
    question: str,
) -> str:

    normalized = normalize(
        question
    )

    enforcement_question = any(
        term in normalized
        for term in (
            "why",
            "breach",
            "breached",
            "failings",
            "failure",
            "fine",
            "fined",
            "penalty",
            "rules",
            "principles",
            "conduct",
            "tribunal",
            "enforcement",
            "misconduct",
        )
    )

    if not enforcement_question:
        return question

    return (
        f"{question} "
        "FCA enforcement regulatory "
        "breaches failings conduct rules "
        "systems controls requirements "
        "decision notice final notice "
        "regulatory findings"
    )


# ============================================================
# SCORING
# ============================================================

def lexical_score(
    question: str,
    text: str,
) -> float:

    question_tokens = tokenize(
        question
    )

    text_tokens = tokenize(
        text
    )

    if not question_tokens:
        return 0.0

    overlap = (
        question_tokens
        & text_tokens
    )

    return (
        len(overlap)
        / len(question_tokens)
    )


def enforcement_score(
    text: str,
) -> float:

    normalized = normalize(
        text
    )

    matches = sum(
        1
        for term in ENFORCEMENT_TERMS
        if term in normalized
    )

    return min(
        matches / 8.0,
        1.0,
    )


def evidence_score(
    question: str,
    hit: dict,
) -> float:

    semantic = float(
        hit.get(
            "score",
            0.0,
        )
        or 0.0
    )

    lexical = lexical_score(
        question,
        hit.get(
            "text",
            "",
        ),
    )

    enforcement = enforcement_score(
        hit.get(
            "text",
            "",
        )
    )

    return (
        semantic * 0.65
        + lexical * 0.20
        + enforcement * 0.15
    )


# ============================================================
# RELEVANCE
# ============================================================

def passes_relevance_gate(
    hit: dict,
) -> bool:

    score = float(
        hit.get(
            "score",
            0.0,
        )
        or 0.0
    )

    return score >= float(
        settings.min_relevance
    )


def apply_relevance_gate(
    hits: list[dict],
) -> list[dict]:

    return [
        hit
        for hit in hits
        if passes_relevance_gate(
            hit
        )
    ]


# ============================================================
# DIVERSITY
# ============================================================

def select_diverse_hits(
    question: str,
    hits: list[dict],
    target_k: int,
) -> list[dict]:

    if not hits:
        return []

    scored = []

    for hit in hits:

        enriched = dict(hit)

        enriched[
            "_evidence_score"
        ] = evidence_score(
            question,
            hit,
        )

        scored.append(
            enriched
        )

    scored.sort(
        key=lambda hit: float(
            hit.get(
                "_evidence_score",
                0.0,
            )
            or 0.0
        ),
        reverse=True,
    )

    selected = []

    seen_documents = set()

    # Prefer different source documents.
    for hit in scored:

        if len(selected) >= target_k:
            break

        url = str(
            hit.get(
                "url",
                "",
            )
            or hit.get(
                "source_url",
                "",
            )
            or ""
        )

        if url and url in seen_documents:
            continue

        selected.append(
            hit
        )

        if url:
            seen_documents.add(
                url
            )

    # Fill if fewer than target.
    if len(selected) < target_k:

        for hit in scored:

            if hit in selected:
                continue

            selected.append(
                hit
            )

            if len(selected) >= target_k:
                break

    return selected[
        :target_k
    ]


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        store: VectorStore,
    ):

        self.store = store

        self.metadata = getattr(
            store,
            "metadata",
            [],
        )

        self.entity_catalogue = (
            build_entity_catalogue(
                self.metadata
            )
        )

        print(
            "Dynamic entity catalogue:",
            len(
                self.entity_catalogue
            ),
        )

        print(
            "Relevance threshold:",
            settings.min_relevance,
        )

        print(
            "TOP_K:",
            settings.top_k,
        )

        print(
            "RETRIEVAL_K:",
            settings.retrieval_k,
        )

    # --------------------------------------------------------
    # Compatibility
    # --------------------------------------------------------

    @staticmethod
    def tokenize(value: Any):
        return tokenize(value)

    def analyze_question(
        self,
        question: str,
    ):
        requested_years = extract_years(question)
        source_metadata = self.metadata
        if requested_years:
            requested = set(requested_years)
            source_metadata = [
                item
                for item in self.metadata
                if str(item.get("year")) in {str(year) for year in requested}
            ]
        return analyze_question(
            question,
            build_entity_catalogue(source_metadata),
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    @staticmethod
    def _target_k():

        return max(
            1,
            int(settings.top_k),
        )

    @staticmethod
    def _retrieval_k(
        target_k: int,
    ):

        return max(
            int(settings.retrieval_k),
            target_k * 4,
        )

    # --------------------------------------------------------
    # Entity search
    # --------------------------------------------------------

    def _search_entity(
        self,
        question: str,
        search_question: str,
        entity: str,
        year: int | None = None,
        k: int = 15,
    ) -> list[dict]:

        filters = {
            "firm_entity": entity
        }

        if year is not None:
            filters["year"] = year

        candidates = self.store.search(
            search_question,
            filters=filters,
            k=k,
        )

        candidates = (
            apply_relevance_gate(
                candidates
            )
        )

        return select_diverse_hits(
            question,
            candidates,
            k,
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    def _comparison_search(
        self,
        question: str,
        entities: list[str],
        years: list[int],
        search_question: str,
        target_k: int,
    ) -> list[dict]:

        groups = []

        # ----------------------------------------------------
        # If years are present, search every entity/year pair.
        # This guarantees year coverage.
        # ----------------------------------------------------

        if years:

            for entity in entities:

                for year in years:

                    results = self._search_entity(
                        question=question,
                        search_question=search_question,
                        entity=entity,
                        year=year,
                        k=max(
                            4,
                            target_k * 2,
                        ),
                    )

                    groups.append(
                        (
                            entity,
                            year,
                            results,
                        )
                    )

        else:

            for entity in entities:

                results = self._search_entity(
                    question=question,
                    search_question=search_question,
                    entity=entity,
                    k=max(
                        5,
                        target_k * 2,
                    ),
                )

                groups.append(
                    (
                        entity,
                        None,
                        results,
                    )
                )

        selected = []

        # ----------------------------------------------------
        # First guarantee one hit from each available group.
        # ----------------------------------------------------

        for (
            entity,
            year,
            results,
        ) in groups:

            if results:

                best = results[0]

                if best not in selected:
                    selected.append(
                        best
                    )

        # ----------------------------------------------------
        # Then fill remaining slots with strongest evidence.
        # ----------------------------------------------------

        remaining = []

        for (
            _entity,
            _year,
            results,
        ) in groups:

            remaining.extend(
                results
            )

        remaining.sort(
            key=lambda hit: float(
                hit.get(
                    "_evidence_score",
                    0.0,
                )
                or 0.0
            ),
            reverse=True,
        )

        for hit in remaining:

            if len(selected) >= target_k:
                break

            if hit not in selected:
                selected.append(
                    hit
                )

        return selected[:max(target_k, len(groups))]

    def _independent_entity_search(
        self,
        question: str,
        entities: list[str],
        years: list[int],
        search_question: str,
        target_k: int,
    ) -> list[dict]:
        groups = []
        requested_years: list[int | None] = years or [None]
        for entity in entities:
            for year in requested_years:
                results = self._search_entity(
                    question,
                    search_question,
                    entity,
                    year,
                    k=max(4, target_k),
                )
                groups.append((entity, year, results))

        selected = []
        for entity, year, results in groups:
            for hit in results:
                item = dict(hit)
                item["entity_group"] = entity
                item["year_group"] = year
                selected.append(item)
                if len([entry for entry in selected if entry["entity_group"] == entity and entry["year_group"] == year]) >= 2:
                    break
        return selected

    # --------------------------------------------------------
    # General search
    # --------------------------------------------------------

    def _general_search(
        self,
        question: str,
        analysis: QueryAnalysis,
        search_question: str,
        target_k: int,
        retrieval_k: int,
    ) -> list[dict]:

        filters = None

        if len(
            analysis.entities
        ) == 1:

            filters = {
                "firm_entity":
                analysis.entities[0]
            }

        if analysis.entity_type == "bank":
            filters = filters or {}
            filters["entity_type"] = "bank"

        if len(
            analysis.years
        ) == 1:

            filters = filters or {}

            filters["year"] = (
                analysis.years[0]
            )

        elif len(
            analysis.years
        ) > 1:

            # For a non-comparison multi-year
            # question, retrieve across all requested years.
            filters = filters or {}

            filters["years"] = (
                analysis.years
            )

        candidates = self.store.search(
            search_question,
            filters=filters,
            k=retrieval_k,
        )

        candidates = (
            apply_relevance_gate(
                candidates
            )
        )

        return select_diverse_hits(
            question,
            candidates,
            target_k,
        )

    # --------------------------------------------------------
    # Main search
    # --------------------------------------------------------

    def search(
        self,
        question: str,
    ) -> list[dict]:

        question = str(
            question or ""
        ).strip()

        if not question:
            return []

        analysis = self.analyze_question(
            question
        )

        target_k = self._target_k()

        retrieval_k = self._retrieval_k(
            target_k
        )

        search_question = expand_question(
            question
        )

        print(
            "\n========== FCA RAG RETRIEVER =========="
        )

        print(
            "Question:",
            question,
        )

        print(
            "Entities:",
            analysis.entities,
        )

        print(
            "Years:",
            analysis.years,
        )

        print(
            "Intent:",
            analysis.intent,
        )

        print(
            "Comparison:",
            analysis.comparison,
        )

        print(
            "Common issues:",
            analysis.common_issues,
        )

        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        result_limit = target_k

        if (
            analysis.comparison
            and analysis.entities
        ):

            hits = self._comparison_search(
                question=question,
                entities=analysis.entities,
                years=analysis.years,
                search_question=search_question,
                target_k=target_k,
            )
            result_limit = max(target_k, len(analysis.entities) * max(1, len(analysis.years)))

        elif len(analysis.entities) > 1:
            hits = self._independent_entity_search(
                question,
                analysis.entities,
                analysis.years,
                search_question,
                target_k,
            )
            result_limit = len(hits)

        # ----------------------------------------------------
        # Common issues
        # ----------------------------------------------------

        elif analysis.common_issues:

            # Wider retrieval is intentional here because
            # "common" is a cross-document question.
            candidates = self.store.search(
                search_question,
                filters={"entity_type": "bank"} if analysis.entity_type == "bank" else None,
                k=max(
                    retrieval_k,
                    min(len(self.metadata), 100),
                ),
            )

            candidates = (
                apply_relevance_gate(
                    candidates
                )
            )

            hits = select_diverse_hits(
                question,
                candidates,
                min(12, len(self.entity_catalogue)),
            )
            result_limit = len(hits)

        # ----------------------------------------------------
        # Normal/entity/year search
        # ----------------------------------------------------

        else:

            hits = self._general_search(
                question=question,
                analysis=analysis,
                search_question=search_question,
                target_k=target_k,
                retrieval_k=retrieval_k,
            )

        final_hits = [
            hit
            for hit in hits
            if passes_relevance_gate(
                hit
            )
        ][:result_limit]

        print(
            "\n========== FINAL FCA EVIDENCE =========="
        )

        print(
            "Final evidence:",
            len(final_hits),
        )

        for index, hit in enumerate(
            final_hits,
            start=1,
        ):

            print(
                f"{index}. "
                f"firm={hit.get('firm', '')} | "
                f"year={hit.get('year', '')} | "
                f"page={hit.get('page', '')} | "
                f"score={float(hit.get('score', 0.0) or 0.0):.4f} | "
                f"evidence={float(hit.get('_evidence_score', 0.0) or 0.0):.4f}"
            )

        print(
            "========================================\n"
        )

        return final_hits


__all__ = [
    "QueryAnalysis",
    "Retriever",
    "normalize",
    "tokenize",
    "build_entity_catalogue",
    "extract_years",
    "extract_entities",
    "detect_comparison",
    "detect_common_issues",
    "detect_intent",
    "expand_question",
    "lexical_score",
    "enforcement_score",
    "evidence_score",
    "passes_relevance_gate",
    "apply_relevance_gate",
    "select_diverse_hits",
    "metadata_entity_matches",
    "analyze_question",
]