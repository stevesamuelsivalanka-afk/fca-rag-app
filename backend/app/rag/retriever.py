import json
import re
from pathlib import Path

from app.core.config import settings
from app.rag.store import VectorStore


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "was",
    "were",
    "is",
    "are",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "what",
    "why",
    "how",
    "did",
    "does",
    "do",
    "this",
    "that",
    "these",
    "those",
    "which",
    "who",
    "when",
    "where",
    "from",
    "into",
    "their",
    "his",
    "her",
    "its",
}


ENFORCEMENT_TERMS = {
    "fine",
    "fined",
    "penalty",
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
}


def normalize(value):
    value = str(value or "").lower().strip()

    value = value.replace("’", "'")
    value = value.replace("–", "-")
    value = value.replace("—", "-")

    value = re.sub(r"\s+", " ", value)

    return value


def tokenize(value):
    return set(
        word
        for word in re.findall(
            r"[a-z0-9]+",
            normalize(value),
        )
        if word not in STOP_WORDS
    )


def load_metadata():
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "metadata.json"
    )

    if not path.exists():
        print("Metadata not found:", path)
        return []

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        print("Metadata error:", exc)
        return []


def build_entity_catalogue(metadata):
    """
    Build entities directly from the indexed data.

    Nothing here is specific to Barclays, Staley,
    John Wood Group, etc.
    """

    entities = set()

    for item in metadata:

        for field in (
            "firm_normalized",
            "firm",
        ):

            value = normalize(
                item.get(field)
            )

            if len(value) >= 3:
                entities.add(value)

    return sorted(
        entities,
        key=lambda x: (
            -len(tokenize(x)),
            -len(x),
        ),
    )


def extract_years(question):
    years = []

    for value in YEAR_RE.findall(question):

        year = int(value)

        if 1900 <= year <= 2100:
            years.append(year)

    return sorted(set(years))


def exact_entity_matches(
    question,
    catalogue,
):
    """
    Match complete metadata entity phrases.
    """

    question_normalized = normalize(
        question
    )

    matches = []

    for entity in catalogue:

        if entity in question_normalized:

            matches.append(entity)

    return sorted(
        set(matches),
        key=len,
        reverse=True,
    )


def fuzzy_entity_matches(
    question,
    catalogue,
):
    """
    Handle natural shortened names.

    Example:

        "James Staley"

    can match:

        "james edward staley"

    without hardcoding Staley.
    """

    question_tokens = tokenize(
        question
    )

    candidates = []

    for entity in catalogue:

        entity_tokens = tokenize(
            entity
        )

        if not entity_tokens:
            continue

        overlap = (
            question_tokens
            & entity_tokens
        )

        if not overlap:
            continue

        coverage = (
            len(overlap)
            / len(entity_tokens)
        )

        # For multi-word entities, require
        # meaningful overlap.
        if (
            len(entity_tokens) >= 2
            and coverage >= 0.50
        ):
            candidates.append(
                (
                    coverage,
                    len(overlap),
                    len(entity),
                    entity,
                )
            )

        # Single-word entities must be exact
        # token matches.
        elif (
            len(entity_tokens) == 1
            and entity_tokens <= question_tokens
        ):
            candidates.append(
                (
                    1.0,
                    1,
                    len(entity),
                    entity,
                )
            )

    candidates.sort(
        reverse=True
    )

    return [
        item[3]
        for item in candidates[:5]
    ]


def extract_entities(
    question,
    catalogue,
):
    exact = exact_entity_matches(
        question,
        catalogue,
    )

    if exact:
        return exact[:5]

    return fuzzy_entity_matches(
        question,
        catalogue,
    )


def detect_comparison(
    question,
    entities,
):
    lower = normalize(question)

    comparison_words = (
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference",
        "differences",
        "between",
        "both",
        "differ",
    )

    return (
        len(entities) >= 2
        or any(
            word in lower
            for word in comparison_words
        )
    )


def expand_question(question):
    """
    Add generic FCA vocabulary for enforcement questions.

    This is not tied to any specific firm.
    """

    lower = normalize(question)

    enforcement_question = any(
        term in lower
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


def lexical_score(
    question,
    text,
):
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


def enforcement_score(text):
    text = normalize(text)

    matches = 0

    for term in ENFORCEMENT_TERMS:

        if term in text:
            matches += 1

    return min(
        matches / 8.0,
        1.0,
    )


def evidence_score(
    question,
    hit,
):
    text = hit.get("text") or ""

    semantic = float(
        hit.get("score", 0)
    )

    lexical = lexical_score(
        question,
        text,
    )

    enforcement = enforcement_score(
        text
    )

    # Semantic remains the strongest signal.
    return (
        semantic * 0.65
        + lexical * 0.20
        + enforcement * 0.15
    )


def select_diverse_hits(
    question,
    hits,
    target_k,
):
    scored = []

    for hit in hits:

        score = evidence_score(
            question,
            hit,
        )

        scored.append(
            (
                score,
                hit,
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    selected = []

    # First pass:
    # avoid returning the same page repeatedly.
    for score, hit in scored:

        page = hit.get("page")

        duplicate_page = False

        for existing in selected:

            existing_page = (
                existing.get("page")
            )

            if (
                page is not None
                and existing_page is not None
                and str(page) == str(existing_page)
            ):
                duplicate_page = True
                break

        if duplicate_page:
            continue

        hit["_evidence_score"] = score

        selected.append(hit)

        if len(selected) >= target_k:
            break

    # Second pass if page diversity wasn't enough.
    if len(selected) < target_k:

        for score, hit in scored:

            if hit in selected:
                continue

            hit["_evidence_score"] = score

            selected.append(hit)

            if len(selected) >= target_k:
                break

    return selected


class Retriever:

    def __init__(
        self,
        store: VectorStore,
    ):
        self.store = store

        self.metadata = load_metadata()

        self.entity_catalogue = (
            build_entity_catalogue(
                self.metadata
            )
        )

        print(
            "Dynamic entity catalogue:",
            len(self.entity_catalogue),
            "entities",
        )

    def search(
        self,
        question: str,
    ):

        target_k = max(
            1,
            settings.top_k,
        )

        retrieval_k = max(
            settings.retrieval_k,
            target_k * 5,
        )

        entities = extract_entities(
            question,
            self.entity_catalogue,
        )

        years = extract_years(
            question
        )

        comparison = detect_comparison(
            question,
            entities,
        )

        search_question = expand_question(
            question
        )

        print(
            "\n========== RETRIEVER DEBUG =========="
        )

        print(
            "Question:",
            question,
        )

        print(
            "Search question:",
            search_question,
        )

        print(
            "Entities:",
            entities,
        )

        print(
            "Years:",
            years,
        )

        print(
            "Comparison:",
            comparison,
        )

        print(
            "Internal retrieval:",
            retrieval_k,
        )

        # =====================================================
        # MULTI-ENTITY QUESTION
        # =====================================================

        if (
            comparison
            and len(entities) >= 2
        ):

            per_entity_k = max(
                6,
                retrieval_k // len(entities),
            )

            entity_results = {}

            for entity in entities:

                filters = {
                    "firm_normalized": entity,
                }

                if len(years) == 1:
                    filters["year"] = years[0]

                elif len(years) > 1:
                    filters["years"] = years

                print(
                    "\nSearching entity:",
                    entity,
                )

                results = self.store.search(
                    search_question,
                    filters=filters,
                    k=per_entity_k,
                )

                results = select_diverse_hits(
                    question,
                    results,
                    per_entity_k,
                )

                entity_results[
                    entity
                ] = results

            hits = []

            # Guarantee at least one result from
            # every detected entity.
            for entity in entities:

                results = entity_results.get(
                    entity,
                    [],
                )

                if results:
                    hits.append(
                        results[0]
                    )

            # Fill remaining slots globally.
            remaining = []

            for results in entity_results.values():

                remaining.extend(
                    results[1:]
                )

            remaining.sort(
                key=lambda h: h.get(
                    "_evidence_score",
                    h.get("score", 0),
                ),
                reverse=True,
            )

            for hit in remaining:

                if len(hits) >= target_k:
                    break

                if hit not in hits:
                    hits.append(hit)

        # =====================================================
        # NORMAL QUESTION
        # =====================================================

        else:

            filters = {}

            if len(years) == 1:
                filters["year"] = years[0]

            elif len(years) > 1:
                filters["years"] = years

            if len(entities) == 1:
                filters[
                    "firm_normalized"
                ] = entities[0]

            hits = self.store.search(
                search_question,
                filters=(
                    filters
                    if filters
                    else None
                ),
                k=retrieval_k,
            )

            hits = select_diverse_hits(
                question,
                hits,
                target_k,
            )

        print(
            "\n========== FINAL RETRIEVER HITS =========="
        )

        print(
            "Hits:",
            len(hits),
        )

        for index, hit in enumerate(
            hits,
            1,
        ):

            print(
                f"{index}. "
                f"{hit.get('firm')} | "
                f"year={hit.get('year')} | "
                f"page={hit.get('page')} | "
                f"amount={hit.get('amount_text')} | "
                f"semantic={hit.get('score')} | "
                f"evidence={hit.get('_evidence_score')} "
            )

        print(
            "==========================================\n"
        )

        return hits

# import re
# from app.core.config import settings
# from app.rag.store import VectorStore


# YEAR_RE = re.compile(r"\b(2024|2025|2026)\b")


# # Map different ways users may mention a firm
# # to the normalized value used by the metadata.
# FIRM_ALIASES = {
#     "barclays bank plc": "barclays",
#     "barclays bank": "barclays",
#     "barclays": "barclays",

#     "nationwide building society": "nationwide",
#     "nationwide": "nationwide",

#     "monzo bank limited": "monzo",
#     "monzo bank": "monzo",
#     "monzo": "monzo",

#     "metro bank plc": "metro bank",
#     "metro bank": "metro bank",
#     "metro": "metro bank",

#     "starling bank limited": "starling",
#     "starling bank": "starling",
#     "starling": "starling",

#     "hsbc uk bank plc": "hsbc",
#     "hsbc uk": "hsbc",
#     "hsbc bank": "hsbc",
#     "hsbc": "hsbc",

#     "tsb bank plc": "tsb",
#     "tsb bank": "tsb",
#     "tsb": "tsb",
# }


# COMPARISON_TERMS = [
#     "compare",
#     "comparison",
#     "versus",
#     "vs",
#     "difference",
#     "differences",
#     "between",
#     "both",
#     "consecutive",
# ]


# def extract_firms(question: str):
#     """
#     Find ALL firms mentioned in the question.
#     """

#     lower = question.lower()
#     found = []

#     # Check longer names first
#     aliases = sorted(
#         FIRM_ALIASES.items(),
#         key=lambda x: len(x[0]),
#         reverse=True
#     )

#     for alias, canonical in aliases:
#         if alias in lower and canonical not in found:
#             found.append(canonical)

#     return found


# def extract_years(question: str):
#     """
#     Extract all years from the question.
#     """

#     return sorted(
#         set(
#             int(year)
#             for year in YEAR_RE.findall(question)
#         )
#     )


# def is_comparison_question(question: str, firms):
#     """
#     Detect comparison-style questions.
#     """

#     lower = question.lower()

#     if len(firms) >= 2:
#         return True

#     return any(
#         term in lower
#         for term in COMPARISON_TERMS
#     )


# def extract_filters(question: str):
#     """
#     Build metadata filters for normal searches.
#     """

#     filters = {}

#     years = extract_years(question)
#     firms = extract_firms(question)

#     # Year filter
#     if len(years) == 1:
#         filters["year"] = years[0]

#     elif len(years) > 1:
#         filters["years"] = years

#     # Firm filter only when there is one firm.
#     #
#     # For multiple firms we search each firm separately
#     # inside Retriever.search().
#     if len(firms) == 1:
#         filters["firm_normalized"] = firms[0]

#     return filters


# class Retriever:

#     def __init__(self, store: VectorStore):
#         self.store = store

#     def search(self, question: str):

#         filters = extract_filters(question)

#         firms = extract_firms(question)
#         years = extract_years(question)

#         comparison = is_comparison_question(
#             question,
#             firms
#         )

#         # We want 3 final hits.
#         target_k = settings.top_k

#         print("\n========== RETRIEVER DEBUG ==========")
#         print("Question:", question)
#         print("Firms:", firms)
#         print("Years:", years)
#         print("Comparison:", comparison)
#         print("Filters:", filters)
#         print("Target hits:", target_k)

#         # =====================================================
#         # COMPARISON QUERY
#         # =====================================================

#         if comparison and len(firms) >= 2:

#             # For 2 firms and target_k=3:
#             #
#             # Starling -> 2 internal hits
#             # Monzo    -> 2 internal hits
#             #
#             # Then select exactly 3 final hits.
#             internal_k = max(
#                 1,
#                 (target_k + len(firms) - 1) // len(firms)
#             )

#             by_firm = {}

#             for firm in firms:

#                 firm_filters = {}

#                 # Preserve year filters
#                 if len(years) == 1:
#                     firm_filters["year"] = years[0]

#                 elif len(years) > 1:
#                     firm_filters["years"] = years

#                 # Search this firm separately
#                 firm_filters["firm_normalized"] = firm

#                 print(
#                     f"\n--- Searching firm: {firm} ---"
#                 )

#                 firm_hits = self.store.search(
#                     question,
#                     filters=firm_filters,
#                     k=internal_k
#                 )

#                 print(
#                     f"{firm}: {len(firm_hits)} hits"
#                 )

#                 by_firm[firm] = firm_hits

#             # =================================================
#             # BUILD FINAL 3 HITS
#             # =================================================

#             hits = []

#             # First guarantee at least ONE result
#             # from every requested firm.
#             for firm in firms:

#                 firm_hits = by_firm.get(
#                     firm,
#                     []
#                 )

#                 if firm_hits:
#                     hits.append(
#                         firm_hits[0]
#                     )

#             # Fill remaining slots.
#             for firm in firms:

#                 firm_hits = by_firm.get(
#                     firm,
#                     []
#                 )

#                 for candidate in firm_hits[1:]:

#                     if len(hits) >= target_k:
#                         break

#                     hits.append(candidate)

#                 if len(hits) >= target_k:
#                     break

#         # =====================================================
#         # NORMAL QUERY
#         # =====================================================

#         else:

#             hits = self.store.search(
#                 question,
#                 filters=filters or None,
#                 k=target_k
#             )

#         # =====================================================
#         # FINAL DEBUG
#         # =====================================================

#         print(
#             "\n========== FINAL RETRIEVER HITS =========="
#         )

#         print(
#             "Hits:",
#             len(hits)
#         )

#         for i, h in enumerate(hits, 1):

#             print(
#                 f"{i}. "
#                 f"{h.get('firm')} | "
#                 f"year={h.get('year')} | "
#                 f"amount={h.get('amount_text')} | "
#                 f"page={h.get('page')} | "
#                 f"url={h.get('url')}"
#             )

#         print(
#             "==========================================\n"
#         )

#         return hits

# # import re
# # from app.core.config import settings
# # from app.rag.store import VectorStore

# # YEAR_RE = re.compile(r"\b(2024|2025|2026)\b")

# # # def extract_filters(question: str) -> dict:
# # #     filters = {}
# # #     year = YEAR_RE.search(question)
# # #     if year:
# # #         filters["year"] = int(year.group(1))
# # #     # Conservative company extraction: only use it for a small set of known examples.
# # #     companies = ["Barclays", "Nationwide Building Society", "Monzo", "Metro Bank", "Starling Bank", "HSBC", "TSB Bank"]
# # #     lower = question.lower()
# # #     for c in companies:
# # #         if c.lower() in lower:
# # #             filters["firm_normalized"] = c.lower()
# # #             break
# # #     return filters

# # def extract_filters(question: str) -> dict:
# #     filters = {}

# #     years = [int(y) for y in re.findall(r"\b(2024|2025|2026)\b", question)]

# #     if len(years) == 1:
# #         filters["year"] = years[0]
# #     elif len(years) > 1:
# #         filters["years"] = sorted(set(years))

# #     companies = [
# #         "Barclays",
# #         "Nationwide Building Society",
# #         "Monzo",
# #         "Metro Bank",
# #         "Starling Bank",
# #         "HSBC",
# #         "TSB Bank"
# #     ]

# #     lower = question.lower()

# #     for c in companies:
# #         if c.lower() in lower:
# #             filters["firm_normalized"] = c.lower()
# #             break

# #     return filters

# # # class Retriever:
# # #     def __init__(self, store: VectorStore):
# # #         self.store = store

# # #     def search(self, question: str):
# # #         filters = extract_filters(question)
# # #         hits = self.store.search(question, filters=filters or None, k=settings.top_k)
# # #         # Previous behavior searched without metadata filters when fewer than
# # #         # two filtered results were found:
# # #         # if len(hits) < 2 and filters:
# # #         #     hits = self.store.search(question, filters=None, k=settings.top_k)
# # #         return hits
# # class Retriever:
# #     def __init__(self, store: VectorStore):
# #         self.store = store

# #     def search(self, question: str):
# #         filters = extract_filters(question)

# #         print("\n========== RETRIEVER DEBUG ==========")
# #         print("Question:", question)
# #         print("Filters:", filters)

# #         hits = self.store.search(
# #             question,
# #             filters=filters or None,
# #             k=settings.top_k
# #         )

# #         print("Hits:", len(hits))

# #         for i, h in enumerate(hits, 1):
# #             print(
# #                 f"{i}. {h.get('firm')} | "
# #                 f"year={h.get('year')} | "
# #                 f"amount={h.get('amount_text')} | "
# #                 f"page={h.get('page')} | "
# #                 f"url={h.get('url')}"
# #             )

# #         print("=====================================\n")

# #         return hits

    