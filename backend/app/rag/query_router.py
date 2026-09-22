import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    query_type: str
    needs_llm: bool
    needs_multiple_cases: bool
    needs_multiple_entities: bool


def _contains(
    text: str,
    terms: tuple[str, ...],
) -> bool:

    return any(
        re.search(
            rf"\b{re.escape(term)}\b",
            text,
        )
        for term in terms
    )


class QueryRouter:

    def classify(
        self,
        question: str,
        entities: list[str],
        years: list[int],
    ) -> QueryPlan:

        text = question.lower().strip()

        if _contains(
            text,
            (
                "largest",
                "biggest",
                "highest",
                "maximum",
            ),
        ) and "fine" in text:

            return QueryPlan(
                query_type="AGGREGATION",
                needs_llm=False,
                needs_multiple_cases=True,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "smallest",
                "lowest",
                "minimum",
            ),
        ) and "fine" in text:

            return QueryPlan(
                query_type="AGGREGATION",
                needs_llm=False,
                needs_multiple_cases=True,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "how many",
                "count",
                "number of",
            ),
        ):

            return QueryPlan(
                query_type="AGGREGATION",
                needs_llm=False,
                needs_multiple_cases=True,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "total",
                "sum",
            ),
        ):

            return QueryPlan(
                query_type="AGGREGATION",
                needs_llm=False,
                needs_multiple_cases=True,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "average",
                "mean",
            ),
        ):

            return QueryPlan(
                query_type="AGGREGATION",
                needs_llm=False,
                needs_multiple_cases=True,
                needs_multiple_entities=False,
            )

        if (
            len(entities) >= 2
            or _contains(
                text,
                (
                    "compare",
                    "comparison",
                    "versus",
                    "vs",
                    "difference",
                    "between",
                ),
            )
        ):

            return QueryPlan(
                query_type="COMPARISON",
                needs_llm=True,
                needs_multiple_cases=True,
                needs_multiple_entities=True,
            )

        if _contains(
            text,
            (
                "common issues",
                "common reasons",
                "most common",
                "typically fined",
                "usually fined",
            ),
        ):

            return QueryPlan(
                query_type="COMMON_ISSUES",
                needs_llm=True,
                needs_multiple_cases=True,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "upper tribunal",
                "tribunal",
            ),
        ):

            return QueryPlan(
                query_type="TRIBUNAL",
                needs_llm=True,
                needs_multiple_cases=True,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "conduct rule",
                "conduct rules",
                "which rule",
                "which rules",
                "principle",
                "principles",
            ),
        ):

            return QueryPlan(
                query_type="RULES",
                needs_llm=True,
                needs_multiple_cases=False,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "why",
                "reason",
                "reasons",
            ),
        ):

            return QueryPlan(
                query_type="WHY",
                needs_llm=True,
                needs_multiple_cases=False,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "when",
                "date",
                "dated",
            ),
        ):

            return QueryPlan(
                query_type="DATE",
                needs_llm=True,
                needs_multiple_cases=False,
                needs_multiple_entities=False,
            )

        if _contains(
            text,
            (
                "source",
                "document",
                "notice",
                "pdf",
            ),
        ):

            return QueryPlan(
                query_type="SOURCE",
                needs_llm=True,
                needs_multiple_cases=False,
                needs_multiple_entities=False,
            )

        if entities:

            return QueryPlan(
                query_type="ENTITY",
                needs_llm=True,
                needs_multiple_cases=False,
                needs_multiple_entities=False,
            )

        return QueryPlan(
            query_type="GENERAL",
            needs_llm=True,
            needs_multiple_cases=False,
            needs_multiple_entities=False,
        )