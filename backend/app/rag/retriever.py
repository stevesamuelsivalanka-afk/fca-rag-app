from __future__ import annotations

import re

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.rag.store import (
    VectorStore,
    canonical_firm_name,
)


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
    "would",
}


# ============================================================
# LEGAL / CORPORATE SUFFIXES
# ============================================================

LEGAL_SUFFIXES = {
    "bank",
    "banks",
    "limited",
    "ltd",
    "plc",
    "llp",
    "lp",
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "group",
    "holdings",
    "holding",
    "services",
    "service",
    "uk",
    "international",
    "securities",
    "capital",
    "asset",
    "assets",
    "management",
    "investment",
    "investments",
    "markets",
    "financial",
    "finance",
    "life",
    "insurance",
    "assurance",
}


GENERIC_TERMS = {
    "fca",
    "fine",
    "fines",
    "fined",
    "penalty",
    "penalties",
    "enforcement",
    "enforcement action",
    "regulatory",
    "regulation",
    "regulations",
    "breach",
    "breaches",
    "breached",
    "fail",
    "failed",
    "failure",
    "failures",
    "failing",
    "failings",
    "misconduct",
    "final notice",
    "final notices",
    "decision notice",
    "decision notices",
    "authority",
    "principle",
    "principles",
    "depp",
    "smcr",
    "conduct rules",
    "financial crime",
    "money laundering",
    "aml",
    "sanctions",
    "systems",
    "controls",
    "governance",
    "compliance",
    "risk",
    "risks",
}


# ============================================================
# INTENT TERMS
# ============================================================

INTENT_TERMS = {
    "why": {
        "why",
        "reason",
        "reasons",
        "because",
        "cause",
        "caused",
        "failure",
        "failures",
        "failings",
        "breach",
        "breached",
        "misconduct",
    },
    "rules": {
        "principle",
        "principles",
        "rule",
        "rules",
        "conduct",
        "icr",
        "smcr",
        "senior management",
        "individual conduct",
        "depp",
        "guideline",
        "guidelines",
    },
    "penalty": {
        "fine",
        "fines",
        "fined",
        "penalty",
        "penalties",
        "amount",
        "amounts",
        "£",
        "million",
        "thousand",
    },
    "comparison": {
        "compare",
        "comparison",
        "versus",
        "vs",
        "between",
        "difference",
        "different",
        "each",
    },
    "common_issues": {
        "common",
        "commonly",
        "issues",
        "issue",
        "problems",
        "themes",
        "reasons",
        "failings",
        "failures",
        "banks",
        "bank",
        "firms",
        "firm",
        "insurance",
        "insurers",
    },
    "tribunal": {
        "tribunal",
        "upper tribunal",
        "decision notice",
        "decision",
        "appeal",
        "appealed",
    },
    "date": {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "early",
        "mid",
        "late",
    },
}


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


# ============================================================
# QUERY ANALYSIS
# ============================================================

@dataclass
class QueryAnalysis:
    question: str
    search_question: str
    entities: list[str]
    ambiguous_entities: list[str]
    years: list[int]
    months: list[int]
    intent: str
    comparison: bool
    common_issues: bool
    asks_why: bool
    asks_rules: bool
    asks_penalty: bool
    asks_tribunal: bool
    asks_date: bool


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 3,
        retrieval_k: int = 15,
        min_relevance: float = 0.20,
    ):
        self.store = store

        self.top_k = max(
            1,
            int(top_k),
        )

        self.retrieval_k = max(
            5,
            int(retrieval_k),
        )

        self.min_relevance = float(
            min_relevance
        )

        self.entity_catalogue = (
            self._build_entity_catalogue()
        )

        print(
            "Dynamic entity catalogue:",
            len(self.entity_catalogue),
            "entities",
        )

    # ========================================================
    # TEXT
    # ========================================================

    @staticmethod
    def tokenize(text: str) -> set[str]:
        text = str(text or "").lower()

        text = re.sub(
            r"[^a-z0-9£]+",
            " ",
            text,
        )

        return {
            token
            for token in text.split()
            if token
            and token not in STOP_WORDS
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(text or "").lower().strip(),
        )

    # ========================================================
    # ENTITY CATALOGUE
    # ========================================================

    def _build_entity_catalogue(self) -> list[str]:
        values = set()

        for metadata in self.store.metadata or []:
            firm = (
                metadata.get("firm")
                or metadata.get("firm_normalized")
                or ""
            )

            firm = str(firm).strip()

            if firm:
                values.add(firm)

        return sorted(
            values,
            key=lambda value: (
                -len(self.tokenize(value)),
                -len(value),
                value.lower(),
            ),
        )

    def _entity_root_tokens(
        self,
        entity: str,
    ) -> set[str]:
        tokens = self.tokenize(entity)

        return {
            token
            for token in tokens
            if token not in LEGAL_SUFFIXES
            and token not in GENERIC_TERMS
        }

    def _entity_root(
        self,
        entity: str,
    ) -> str:
        tokens = self._entity_root_tokens(entity)

        return " ".join(
            sorted(tokens)
        )

    # ========================================================
    # EXACT ENTITY MATCH
    # ========================================================

    def _exact_entity_matches(
        self,
        question: str,
    ) -> list[str]:

        normalized_question = (
            self.normalize_text(question)
        )

        matches = []

        for entity in self.entity_catalogue:
            normalized_entity = (
                self.normalize_text(entity)
            )

            if not normalized_entity:
                continue

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(normalized_entity)
                + r"(?![a-z0-9])"
            )

            if re.search(
                pattern,
                normalized_question,
            ):
                matches.append(entity)

        return matches

    # ========================================================
    # BRAND / ROOT ENTITY MATCHING
    # ========================================================

    def _root_entity_candidates(
        self,
        question: str,
    ) -> dict[str, list[str]]:

        question_tokens = self.tokenize(
            question
        )

        grouped = defaultdict(list)

        if not question_tokens:
            return {}

        for entity in self.entity_catalogue:

            root_tokens = (
                self._entity_root_tokens(entity)
            )

            if not root_tokens:
                continue

            overlap = (
                question_tokens
                & root_tokens
            )

            # Every meaningful root token must be present
            # for multi-token entities.
            if not overlap:
                continue

            if len(root_tokens) == 1:
                if overlap != root_tokens:
                    continue
            else:
                if len(overlap) != len(root_tokens):
                    continue

            root = self._entity_root(entity)

            if root:
                grouped[root].append(entity)

        return dict(grouped)

    def _resolve_root_entities(
        self,
        question: str,
        years: list[int],
        comparison: bool,
    ) -> tuple[list[str], list[str]]:

        grouped = self._root_entity_candidates(
            question
        )

        if not grouped:
            return [], []

        resolved = []
        ambiguous = []

        for root, entities in grouped.items():

            # Deduplicate legal entity names.
            entities = list(dict.fromkeys(entities))

            if len(entities) == 1:
                resolved.append(
                    entities[0]
                )
                continue

            # ------------------------------------------------
            # For a multi-year comparison, establish whether
            # one legal entity exists across all requested
            # years. This is what allows "Barclays" to resolve
            # dynamically without hardcoding Barclays.
            # ------------------------------------------------

            if comparison and years:

                coverage = []

                for entity in entities:

                    entity_years = set()

                    for metadata in (
                        self.store.metadata or []
                    ):
                        if not self._metadata_entity_matches(
                            metadata,
                            entity,
                        ):
                            continue

                        try:
                            year = int(
                                metadata.get("year")
                            )
                        except (
                            TypeError,
                            ValueError,
                        ):
                            continue

                        entity_years.add(year)

                    requested_coverage = (
                        entity_years
                        & set(years)
                    )

                    coverage.append(
                        (
                            len(requested_coverage),
                            entity,
                            requested_coverage,
                        )
                    )

                coverage.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                if coverage:

                    best_count = coverage[0][0]

                    best = [
                        item
                        for item in coverage
                        if item[0] == best_count
                    ]

                    # Exactly one legal entity covers
                    # every requested year.
                    if (
                        best_count == len(years)
                        and len(best) == 1
                    ):
                        resolved.append(
                            best[0][1]
                        )
                        continue

            # Multiple legal entities remain possible.
            ambiguous.extend(
                entities
            )

        return (
            list(dict.fromkeys(resolved)),
            list(dict.fromkeys(ambiguous)),
        )

    
    def _basic_query_analysis(
        self,
        question: str,
    ) -> QueryAnalysis:

        years = self.extract_years(
            question
        )

        months = self.extract_months(
            question
        )

        normalized = self.normalize_text(
            question
        )

        intent = self.detect_intent(
            question
        )

        comparison = bool(
            re.search(
                r"\bcompare\b"
                r"|\bversus\b"
                r"|\bvs\.?\b"
                r"|\bbetween\b",
                normalized,
            )
        )

        return QueryAnalysis(
            question=question,
            search_question=question,
            entities=[],
            ambiguous_entities=[],
            years=years,
            months=months,
            intent=intent,
            comparison=comparison,
            common_issues=(
                intent == "common_issues"
            ),
            asks_why=(
                normalized.startswith("why")
                or intent == "why"
            ),
            asks_rules=(
                intent == "rules"
            ),
            asks_penalty=(
                intent == "penalty"
            ),
            asks_tribunal=(
                intent == "tribunal"
            ),
            asks_date=(
                bool(months)
                or intent == "date"
            ),
        )


    def _basic_query_analysis(
        self,
        question: str,
    ) -> QueryAnalysis:

        years = self.extract_years(
            question
        )

        months = self.extract_months(
            question
        )

        normalized = self.normalize_text(
            question
        )

        intent = self.detect_intent(
            question
        )

        comparison = bool(
            re.search(
                r"\bcompare\b"
                r"|\bversus\b"
                r"|\bvs\.?\b"
                r"|\bbetween\b",
                normalized,
            )
        )

        return QueryAnalysis(
            question=question,
            search_question=question,
            entities=[],
            ambiguous_entities=[],
            years=years,
            months=months,
            intent=intent,
            comparison=comparison,
            common_issues=(
                intent == "common_issues"
            ),
            asks_why=(
                normalized.startswith("why")
                or intent == "why"
            ),
            asks_rules=(
                intent == "rules"
            ),
            asks_penalty=(
                intent == "penalty"
            ),
            asks_tribunal=(
                intent == "tribunal"
            ),
            asks_date=(
                bool(months)
                or intent == "date"
            ),
        )


    def _resolve_ambiguous_root_entity(
        self,
        question: str,
        candidates: list[str],
        years: list[int],
    ) -> tuple[list[str], list[str]]:

        if not candidates:
            return [], []

        analysis = self._basic_query_analysis(
            question
        )

        scored = []

        for entity in candidates:

            entity_metadata = []

            for metadata in self.store.metadata or []:

                if not self._metadata_entity_matches(
                    metadata,
                    entity,
                ):
                    continue

                if years and not self._metadata_year_matches(
                    metadata,
                    years,
                ):
                    continue

                entity_metadata.append(
                    metadata
                )

            if not entity_metadata:
                continue

            best_score = 0.0
            total_score = 0.0
            enforcement_hits = 0

            for metadata in entity_metadata:

                text = str(
                    metadata.get("text") or ""
                )

                firm = str(
                    metadata.get("firm")
                    or ""
                )

                searchable = (
                    f"{firm} {text}"
                )

                lexical = self.lexical_score(
                    question,
                    searchable,
                )

                phrase = self.phrase_score(
                    question,
                    searchable,
                )

                enforcement = self.enforcement_score(
                    searchable,
                    analysis,
                )

                score = (
                    lexical * 0.35
                    + phrase * 0.20
                    + enforcement * 0.45
                )

                best_score = max(
                    best_score,
                    score,
                )

                total_score += score

                lower = text.lower()

                if any(
                    term in lower
                    for term in (
                        "breached",
                        "breach",
                        "failed to",
                        "failings",
                        "misconduct",
                        "financial penalty",
                        "fined",
                        "penalty",
                    )
                ):
                    enforcement_hits += 1

            average_score = (
                total_score
                / len(entity_metadata)
            )

            final_score = (
                best_score * 0.65
                + average_score * 0.20
                + min(
                    0.15,
                    enforcement_hits * 0.03,
                )
            )

            scored.append(
                (
                    final_score,
                    best_score,
                    entity,
                    enforcement_hits,
                )
            )

        if not scored:
            return [], candidates

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[3],
            ),
            reverse=True,
        )

        if len(scored) == 1:
            return [scored[0][2]], []

        best = scored[0]
        second = scored[1]

        margin = (
            best[0] - second[0]
        )

        if (
            best[0] >= 0.20
            and margin >= 0.06
        ):
            return [best[2]], []

        return [], candidates

    def resolve_entities(
        self,
        question: str,
        years: list[int] | None = None,
        comparison: bool = False,
    ) -> tuple[list[str], list[str]]:

        years = years or []

        exact = self._exact_entity_matches(
            question
        )

        if exact:
            return (
                exact,
                [],
            )

        resolved, ambiguous = (
            self._resolve_root_entities(
                question,
                years,
                comparison,
            )
        )

        if resolved:
            return (
                resolved,
                [],
            )

        if ambiguous:
            return self._resolve_ambiguous_root_entity(
                question,
                ambiguous,
                years,
            )

        return [], []
    # ========================================================
    # YEAR / MONTH
    # ========================================================

    @staticmethod
    def extract_years(
        question: str,
    ) -> list[int]:

        return sorted(
            {
                int(value)
                for value in re.findall(
                    r"\b(?:2024|2025|2026)\b",
                    question,
                )
            }
        )

    @staticmethod
    def extract_months(
        question: str,
    ) -> list[int]:

        question_lower = question.lower()

        months = []

        for name, number in MONTHS.items():

            if re.search(
                rf"\b{name}\b",
                question_lower,
            ):
                months.append(number)

        return sorted(set(months))

    # ========================================================
    # INTENT
    # ========================================================

    def detect_intent(
        self,
        question: str,
    ) -> str:

        text = self.normalize_text(
            question
        )

        scores = {}

        for intent, terms in INTENT_TERMS.items():

            score = 0

            for term in terms:
                if term in text:
                    score += 1

            scores[intent] = score

        if text.startswith("why "):
            return "why"

        if (
            "what happened" in text
            or "reason" in text
            or "why was" in text
            or "why were" in text
        ):
            return "why"

        best = max(
            scores,
            key=scores.get,
        )

        if scores[best] <= 0:
            return "general"

        return best
    # ========================================================
    # QUERY ANALYSIS
    # ========================================================

    def analyze_question(
        self,
        question: str,
    ) -> QueryAnalysis:

        question = str(
            question or ""
        ).strip()

        years = self.extract_years(
            question
        )

        months = self.extract_months(
            question
        )

        intent = self.detect_intent(
            question
        )

        normalized = self.normalize_text(
            question
        )

        comparison = (
            intent == "comparison"
            or bool(
                re.search(
                    r"\bcompare\b"
                    r"|\bversus\b"
                    r"|\bvs\.?\b"
                    r"|\bbetween\b",
                    normalized,
                )
            )
        )

        common_issues = (
            intent == "common_issues"
            or (
                "common" in normalized
                and (
                    "issues" in normalized
                    or "failings" in normalized
                    or "failures" in normalized
                )
            )
        )

        entities, ambiguous = (
            self.resolve_entities(
                question,
                years=years,
                comparison=comparison,
            )
        )

        asks_why = (
            normalized.startswith("why")
            or intent == "why"
        )

        asks_rules = (
            intent == "rules"
            or any(
                term in normalized
                for term in (
                    "principle",
                    "principles",
                    "conduct rule",
                    "conduct rules",
                    "smcr",
                    "icr",
                    "depp",
                )
            )
        )

        asks_penalty = (
            intent == "penalty"
            or any(
                term in normalized
                for term in (
                    "fine",
                    "fines",
                    "fined",
                    "penalty",
                    "penalties",
                )
            )
        )

        asks_tribunal = (
            intent == "tribunal"
            or "tribunal" in normalized
            or "appeal" in normalized
            or "decision notice" in normalized
        )

        asks_date = (
            bool(months)
            or intent == "date"
            or "late " in normalized
            or "early " in normalized
            or "mid " in normalized
        )

        search_terms = [
            question,
            "FCA enforcement",
            "Final Notice",
        ]

        if asks_why:
            search_terms.extend(
                [
                    "breach",
                    "failed",
                    "failings",
                    "misconduct",
                    "reason",
                ]
            )

        if asks_rules:
            search_terms.extend(
                [
                    "Principle",
                    "Conduct Rules",
                    "SMCR",
                    "ICR",
                    "DEPP",
                ]
            )

        if asks_penalty:
            search_terms.extend(
                [
                    "financial penalty",
                    "fine",
                    "penalty",
                    "Step 1",
                    "Step 2",
                    "Step 3",
                ]
            )

        if asks_tribunal:
            search_terms.extend(
                [
                    "Upper Tribunal",
                    "Decision Notice",
                    "Final Notice",
                    "appeal",
                ]
            )

        return QueryAnalysis(
            question=question,
            search_question=" ".join(
                search_terms
            ),
            entities=entities,
            ambiguous_entities=ambiguous,
            years=years,
            months=months,
            intent=intent,
            comparison=comparison,
            common_issues=common_issues,
            asks_why=asks_why,
            asks_rules=asks_rules,
            asks_penalty=asks_penalty,
            asks_tribunal=asks_tribunal,
            asks_date=asks_date,
        )

    # ========================================================
    # LEXICAL SCORING
    # ========================================================

    def lexical_score(
        self,
        question: str,
        text: str,
    ) -> float:

        q = self.tokenize(question)
        t = self.tokenize(text)

        if not q or not t:
            return 0.0

        overlap = (
            len(q & t)
            / max(1, len(q))
        )

        return min(
            1.0,
            overlap * 2.5,
        )

    def phrase_score(
        self,
        question: str,
        text: str,
    ) -> float:

        question_lower = self.normalize_text(
            question
        )

        text_lower = self.normalize_text(
            text
        )

        phrases = [
            "financial penalty",
            "final notice",
            "decision notice",
            "upper tribunal",
            "conduct rules",
            "individual conduct rules",
            "senior manager conduct rules",
            "principle 1",
            "principle 2",
            "principle 3",
            "principle 6",
            "money laundering",
            "financial crime",
            "sanctions screening",
            "systems and controls",
        ]

        score = 0.0

        for phrase in phrases:
            if (
                phrase in question_lower
                and phrase in text_lower
            ):
                score += 0.20

        return min(
            1.0,
            score,
        )

    # ========================================================
    # ENFORCEMENT SCORE
    # ========================================================

    def enforcement_score(
        self,
        text: str,
        analysis: QueryAnalysis,
    ) -> float:

        text_lower = self.normalize_text(
            text
        )

        score = 0.0

        strong_terms = {
            "breached": 0.18,
            "breach": 0.15,
            "misconduct": 0.15,
            "failed to": 0.15,
            "failings": 0.14,
            "financial penalty": 0.14,
            "fined": 0.14,
            "penalty": 0.12,
            "the authority considers": 0.10,
            "principle": 0.10,
            "final notice": 0.10,
            "decision notice": 0.10,
            "step 1": 0.08,
            "step 2": 0.08,
            "step 3": 0.08,
        }

        for term, weight in strong_terms.items():
            if term in text_lower:
                score += weight

        if analysis.asks_why:
            if any(
                term in text_lower
                for term in (
                    "breached",
                    "breach",
                    "failed to",
                    "failings",
                    "misconduct",
                )
            ):
                score += 0.25

        if analysis.asks_rules:
            if any(
                term in text_lower
                for term in (
                    "principle",
                    "conduct rule",
                    "smcr",
                    "icr",
                )
            ):
                score += 0.25

        if analysis.asks_penalty:
            if any(
                term in text_lower
                for term in (
                    "penalty",
                    "fined",
                    "fine",
                    "step 1",
                    "step 2",
                    "step 3",
                )
            ):
                score += 0.25

        if analysis.asks_tribunal:
            if any(
                term in text_lower
                for term in (
                    "tribunal",
                    "decision notice",
                    "appeal",
                )
            ):
                score += 0.35

        return min(
            1.0,
            score,
        )

    # ========================================================
    # DATE
    # ========================================================

    def date_score(
        self,
        metadata: dict[str, Any],
        analysis: QueryAnalysis,
    ) -> float:

        if not (
            analysis.years
            or analysis.months
        ):
            return 0.0

        score = 0.0

        try:
            metadata_year = int(
                metadata.get("year")
            )
        except (
            TypeError,
            ValueError,
        ):
            metadata_year = None

        if (
            metadata_year is not None
            and metadata_year in analysis.years
        ):
            score += 0.35

        text = self.normalize_text(
            metadata.get("text") or ""
        )

        for month_number in analysis.months:

            month_name = next(
                (
                    name
                    for name, number
                    in MONTHS.items()
                    if number == month_number
                ),
                "",
            )

            if month_name in text:
                score += 0.10

        return min(
            1.0,
            score,
        )

    # ========================================================
    # KEYS
    # ========================================================

    @staticmethod
    def case_key(
        metadata: dict[str, Any],
    ) -> str:

        url = (
            metadata.get("url")
            or metadata.get("source_url")
            or ""
        )

        if url:
            return str(url)

        return "|".join(
            [
                str(
                    metadata.get("firm")
                    or ""
                ),
                str(
                    metadata.get("year")
                    or ""
                ),
            ]
        )

    @staticmethod
    def chunk_key(
        metadata: dict[str, Any],
    ) -> str:

        return "|".join(
            [
                str(
                    metadata.get("url")
                    or metadata.get("source_url")
                    or ""
                ),
                str(
                    metadata.get("page")
                    or ""
                ),
                str(
                    metadata.get("text")
                    or ""
                )[:160],
            ]
        )

    # ========================================================
    # METADATA MATCHING
    # ========================================================

    def _metadata_entity_matches(
        self,
        metadata: dict[str, Any],
        entity: str,
    ) -> bool:

        actual = canonical_firm_name(
            metadata.get(
                "firm_normalized"
            )
            or metadata.get("firm")
            or ""
        )

        requested = canonical_firm_name(
            entity
        )

        return bool(
            requested
        ) and actual == requested

    def _metadata_year_matches(
        self,
        metadata: dict[str, Any],
        years: list[int],
    ) -> bool:

        if not years:
            return True

        try:
            year = int(
                metadata.get("year")
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        return year in years

    # ========================================================
    # LEXICAL SEARCH
    # ========================================================

    def _lexical_search(
        self,
        analysis: QueryAnalysis,
        limit: int = 80,
    ) -> list[dict]:

        candidates = []

        question = analysis.search_question

        for metadata_index, metadata in enumerate(
            self.store.metadata or []
        ):

            if not self._metadata_year_matches(
                metadata,
                analysis.years,
            ):
                continue

            text = str(
                metadata.get("text") or ""
            )

            firm = str(
                metadata.get("firm")
                or metadata.get(
                    "firm_normalized"
                )
                or ""
            )

            searchable = (
                f"{firm} {text}"
            )

            lexical = self.lexical_score(
                question,
                searchable,
            )

            phrase = self.phrase_score(
                analysis.question,
                searchable,
            )

            enforcement = self.enforcement_score(
                searchable,
                analysis,
            )

            date = self.date_score(
                metadata,
                analysis,
            )

            entity_bonus = 0.0

            if analysis.entities:
                if any(
                    self._metadata_entity_matches(
                        metadata,
                        entity,
                    )
                    for entity in analysis.entities
                ):
                    entity_bonus = 0.45

            score = (
                lexical * 0.35
                + phrase * 0.20
                + enforcement * 0.25
                + date * 0.10
                + entity_bonus
            )

            if score <= 0:
                continue

            result = {
                **metadata,
                "score": float(
                    min(1.0, score)
                ),
                "_metadata_index": metadata_index,
                "_retrieval_type": "lexical",
            }

            candidates.append(result)

        candidates.sort(
            key=lambda item: float(
                item.get("score", 0.0)
            ),
            reverse=True,
        )

        return candidates[:limit]

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def _semantic_search(
        self,
        query: str,
        filters: dict | None = None,
        k: int = 20,
    ) -> list[dict]:

        try:
            results = self.store.search(
                query=query,
                filters=filters,
                k=k,
                firm_match_mode="exact",
            )
        except TypeError:
            results = self.store.search(
                query=query,
                filters=filters,
                k=k,
            )

        for item in results:
            item["_retrieval_type"] = "semantic"

        return results

    # ========================================================
    # CANDIDATE MERGE / RERANK
    # ========================================================

    def _merge_candidates(
        self,
        analysis: QueryAnalysis,
        candidates: list[dict],
    ) -> list[dict]:

        merged = {}

        for candidate in candidates:

            key = self.chunk_key(candidate)

            if key not in merged:
                merged[key] = {
                    **candidate
                }
                continue

            current = merged[key]

            current["score"] = max(
                float(
                    current.get(
                        "score",
                        0.0,
                    )
                ),
                float(
                    candidate.get(
                        "score",
                        0.0,
                    )
                ),
            )

            current["_retrieval_type"] = "hybrid"

        reranked = []

        for candidate in merged.values():

            text = str(
                candidate.get("text") or ""
            )

            semantic = float(
                candidate.get(
                    "score",
                    0.0,
                )
            )

            searchable = (
                str(
                    candidate.get("firm")
                    or ""
                )
                + " "
                + text
            )

            lexical = self.lexical_score(
                analysis.question,
                searchable,
            )

            phrase = self.phrase_score(
                analysis.question,
                text,
            )

            enforcement = self.enforcement_score(
                text,
                analysis,
            )

            date = self.date_score(
                candidate,
                analysis,
            )

            entity_bonus = 0.0

            if analysis.entities:
                if any(
                    self._metadata_entity_matches(
                        candidate,
                        entity,
                    )
                    for entity in analysis.entities
                ):
                    entity_bonus = 0.25

            final_score = (
                semantic * 0.30
                + lexical * 0.18
                + phrase * 0.12
                + enforcement * 0.30
                + date * 0.05
                + entity_bonus
            )

            candidate["_evidence_score"] = float(
                min(1.0, final_score)
            )

            reranked.append(candidate)

        reranked.sort(
            key=lambda item: float(
                item.get(
                    "_evidence_score",
                    0.0,
                )
            ),
            reverse=True,
        )

        return reranked

    # ========================================================
    # DIVERSITY
    # ========================================================

    def _select_diverse(
        self,
        candidates: list[dict],
        limit: int,
        analysis: QueryAnalysis,
    ) -> list[dict]:

        if not candidates:
            return []

        selected = []

        seen_chunks = set()
        seen_cases = set()

        for candidate in candidates:

            chunk = self.chunk_key(
                candidate
            )

            case = self.case_key(
                candidate
            )

            if chunk in seen_chunks:
                continue

            if case in seen_cases:
                continue

            selected.append(candidate)

            seen_chunks.add(chunk)
            seen_cases.add(case)

            if len(selected) >= limit:
                return selected

        for candidate in candidates:

            chunk = self.chunk_key(
                candidate
            )

            if chunk in seen_chunks:
                continue

            selected.append(candidate)
            seen_chunks.add(chunk)

            if len(selected) >= limit:
                break

        return selected

    # ========================================================
    # ENTITY GROUPING
    # ========================================================

    def _group_by_entity(
        self,
        hits: list[dict],
    ) -> dict[str, list[dict]]:

        grouped = defaultdict(list)

        for hit in hits:

            key = canonical_firm_name(
                hit.get(
                    "firm_normalized"
                )
                or hit.get("firm")
                or ""
            )

            if key:
                grouped[key].append(hit)

        return dict(grouped)

    # ========================================================
    # COMPARISON SEARCH
    # ========================================================

    def _comparison_search(
        self,
        analysis: QueryAnalysis,
    ) -> list[dict]:

        all_candidates = []

        # ----------------------------------------------------
        # Resolved legal entities
        # ----------------------------------------------------

        if analysis.entities:

            for entity in analysis.entities:

                if analysis.years:

                    for year in analysis.years:

                        filters = {
                            "firm_normalized": entity,
                            "year": year,
                        }

                        query = (
                            f"{analysis.question} "
                            f"{entity} "
                            f"{year} "
                            "FCA Final Notice "
                            "financial penalty fine "
                            "breach failings misconduct"
                        )

                        all_candidates.extend(
                            self._semantic_search(
                                query,
                                filters=filters,
                                k=20,
                            )
                        )

                        # Exact metadata fallback.
                        for metadata_index, metadata in enumerate(
                            self.store.metadata or []
                        ):

                            if not self._metadata_entity_matches(
                                metadata,
                                entity,
                            ):
                                continue

                            if not self._metadata_year_matches(
                                metadata,
                                [year],
                            ):
                                continue

                            item = {
                                **metadata,
                                "score": 0.0,
                                "_metadata_index": metadata_index,
                                "_retrieval_type":
                                    "lexical-comparison",
                            }

                            all_candidates.append(item)

                else:

                    all_candidates.extend(
                        self._semantic_search(
                            analysis.search_question,
                            filters={
                                "firm_normalized": entity
                            },
                            k=30,
                        )
                    )

            return self._merge_candidates(
                analysis,
                all_candidates,
            )

        # ----------------------------------------------------
        # Ambiguous legal entities
        # ----------------------------------------------------

        if analysis.ambiguous_entities:

            for entity in analysis.ambiguous_entities:

                for year in (
                    analysis.years or [None]
                ):

                    filters = {
                        "firm_normalized": entity
                    }

                    if year is not None:
                        filters["year"] = year

                    query = (
                        f"{analysis.question} "
                        f"{entity} "
                        f"{year or ''} "
                        "FCA Final Notice "
                        "financial penalty fine breach"
                    )

                    all_candidates.extend(
                        self._semantic_search(
                            query,
                            filters=filters,
                            k=12,
                        )
                    )

            return self._merge_candidates(
                analysis,
                all_candidates,
            )

        # ----------------------------------------------------
        # Generic comparison
        # ----------------------------------------------------

        if analysis.years:

            all_candidates.extend(
                self._semantic_search(
                    analysis.search_question,
                    filters={
                        "years": analysis.years
                    },
                    k=40,
                )
            )

        else:

            all_candidates.extend(
                self._semantic_search(
                    analysis.search_question,
                    k=40,
                )
            )

        all_candidates.extend(
            self._lexical_search(
                analysis,
                limit=100,
            )
        )

        return self._merge_candidates(
            analysis,
            all_candidates,
        )

    # ========================================================
    # COMMON ISSUES
    # ========================================================

    def _common_issue_search(
        self,
        analysis: QueryAnalysis,
    ) -> list[dict]:

        candidates = []

        candidates.extend(
            self._semantic_search(
                analysis.search_question
                + " financial crime "
                + "systems controls "
                + "failings breaches",
                filters=(
                    {"years": analysis.years}
                    if analysis.years
                    else None
                ),
                k=40,
            )
        )

        candidates.extend(
            self._lexical_search(
                analysis,
                limit=120,
            )
        )

        merged = self._merge_candidates(
            analysis,
            candidates,
        )

        return self._select_diverse(
            merged,
            max(
                8,
                self.top_k * 3,
            ),
            analysis,
        )

    # ========================================================
    # NORMAL SEARCH
    # ========================================================

    def _normal_search(
        self,
        analysis: QueryAnalysis,
    ) -> list[dict]:

        candidates = []

        if analysis.entities:

            for entity in analysis.entities:

                filters = {
                    "firm_normalized": entity
                }

                if analysis.years:

                    if len(analysis.years) == 1:
                        filters["year"] = (
                            analysis.years[0]
                        )
                    else:
                        filters["years"] = (
                            analysis.years
                        )

                candidates.extend(
                    self._semantic_search(
                        analysis.search_question,
                        filters=filters,
                        k=30,
                    )
                )

            candidates.extend(
                self._lexical_search(
                    analysis,
                    limit=100,
                )
            )

        elif analysis.ambiguous_entities:

            for entity in analysis.ambiguous_entities:

                filters = {
                    "firm_normalized": entity
                }

                if analysis.years:

                    if len(analysis.years) == 1:
                        filters["year"] = (
                            analysis.years[0]
                        )
                    else:
                        filters["years"] = (
                            analysis.years
                        )

                candidates.extend(
                    self._semantic_search(
                        analysis.search_question,
                        filters=filters,
                        k=25,
                    )
                )

        else:

            candidates.extend(
                self._semantic_search(
                    analysis.search_question,
                    filters=(
                        {
                            "years": analysis.years
                        }
                        if analysis.years
                        else None
                    ),
                    k=40,
                )
            )

            candidates.extend(
                self._lexical_search(
                    analysis,
                    limit=120,
                )
            )

        return self._merge_candidates(
            analysis,
            candidates,
        )

    # ========================================================
    # PUBLIC SEARCH
    # ========================================================

    def search(
        self,
        question: str,
    ) -> list[dict]:

        analysis = self.analyze_question(
            question
        )

        print(
            "\n========== RETRIEVER DEBUG =========="
        )

        print(
            "Question:",
            analysis.question,
        )

        print(
            "Search question:",
            analysis.search_question,
        )

        print(
            "Entities:",
            analysis.entities,
        )

        print(
            "Ambiguous entities:",
            analysis.ambiguous_entities,
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

        print(
            "======================================"
        )

        if analysis.common_issues:

            candidates = self._common_issue_search(
                analysis
            )

        elif analysis.comparison:

            candidates = self._comparison_search(
                analysis
            )

        else:

            candidates = self._normal_search(
                analysis
            )

        if analysis.common_issues:
            limit = max(
                8,
                self.top_k * 3,
            )
        else:
            limit = max(
                3,
                self.top_k,
            )

        selected = self._select_diverse(
            candidates,
            limit,
            analysis,
        )

        print(
            "\n========== RETRIEVER RESULTS =========="
        )

        print(
            "Candidates:",
            len(candidates),
        )

        print(
            "Selected:",
            len(selected),
        )

        for index, hit in enumerate(
            selected,
            start=1,
        ):

            print(
                f"[{index}]",
                hit.get("firm"),
                hit.get("year"),
                "page=",
                hit.get("page"),
                "score=",
                round(
                    float(
                        hit.get(
                            "_evidence_score",
                            hit.get(
                                "score",
                                0.0,
                            ),
                        )
                    ),
                    4,
                ),
            )

        print(
            "========================================\n"
        )

        return selected