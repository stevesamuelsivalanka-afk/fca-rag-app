from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

import requests

from app.core.config import settings


class RAGService:

    def __init__(
        self,
        retriever,
    ):
        self.retriever = retriever

    # ============================================================
    # TEXT HELPERS
    # ============================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        text = str(
            text or ""
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _sentences(
        text: str,
    ) -> list[str]:

        text = (
            RAGService._clean_text(
                text
            )
        )

        if not text:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ============================================================
    # EVIDENCE SENTENCE RELEVANCE
    # ============================================================

    def _sentence_score(
        self,
        question: str,
        sentence: str,
    ) -> float:

        q = self.retriever.tokenize(
            question
        )

        s = self.retriever.tokenize(
            sentence
        )

        if not q or not s:
            return 0.0

        overlap = len(
            q & s
        )

        score = (
            overlap
            / max(
                1,
                len(q),
            )
        )

        lower = sentence.lower()

        # Enforcement language gets priority.

        for term in (
            "breached",
            "breach",
            "failed to",
            "failings",
            "misconduct",
            "financial penalty",
            "fined",
            "principle",
            "conduct rule",
            "senior manager",
            "individual conduct",
            "decision notice",
            "final notice",
            "upper tribunal",
            "step 1",
            "step 2",
            "step 3",
        ):

            if term in lower:
                score += 0.12

        return score

    # ============================================================
    # BUILD EVIDENCE
    # ============================================================

    def _build_evidence(
        self,
        question: str,
        hit: dict,
        max_chars: int = 1000,
    ) -> str:

        text = self._clean_text(
            hit.get(
                "text"
            )
            or ""
        )

        if not text:
            return ""

        sentences = (
            self._sentences(
                text
            )
        )

        ranked = sorted(
            (
                (
                    self._sentence_score(
                        question,
                        sentence,
                    ),
                    index,
                    sentence,
                )
                for index, sentence
                in enumerate(sentences)
            ),
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        selected = []

        length = 0

        for _, _, sentence in ranked:

            if (
                length
                + len(sentence)
                > max_chars
            ):
                continue

            selected.append(
                sentence
            )

            length += (
                len(sentence)
                + 1
            )

            if length >= (
                max_chars * 0.85
            ):
                break

        if not selected:

            return text[
                :max_chars
            ]

        return " ".join(
            selected
        )

    # ============================================================
    # CONTEXT
    # ============================================================

    def _build_context(
        self,
        question: str,
        hits: list[dict],
    ) -> str:

        blocks = []

        for number, hit in enumerate(
            hits,
            start=1,
        ):

            firm = (
                hit.get(
                    "firm"
                )
                or hit.get(
                    "firm_normalized"
                )
                or "Unknown firm"
            )

            year = (
                hit.get(
                    "year"
                )
                or ""
            )

            page = (
                hit.get(
                    "page"
                )
                or ""
            )

            evidence = (
                self._build_evidence(
                    question,
                    hit,
                    max_chars=950,
                )
            )

            if not evidence:
                continue

            blocks.append(
                (
                    f"[S{number}] "
                    f"{firm} "
                    f"({year}, page {page})\n"
                    f"{evidence}"
                )
            )

        return "\n\n".join(
            blocks
        )

    # ============================================================
    # DETERMINISTIC LARGEST FINE
    # ============================================================

    @staticmethod
    def _extract_amount(
        text: str,
    ) -> float | None:

        if not text:
            return None

        # Handles examples such as:
        #
        # £44,078,500
        # £44m
        # £1.1 million
        # £46,803,329.51

        matches = re.findall(
            r"£\s*"
            r"([0-9][0-9,]*(?:\.[0-9]+)?)"
            r"\s*"
            r"(million|m|thousand|k)?",
            text.lower(),
        )

        values = []

        for number, multiplier in matches:

            try:
                value = float(
                    number.replace(
                        ",",
                        "",
                    )
                )
            except ValueError:
                continue

            if multiplier in (
                "million",
                "m",
            ):
                value *= 1_000_000

            elif multiplier in (
                "thousand",
                "k",
            ):
                value *= 1_000

            values.append(
                value
            )

        if not values:
            return None

        return max(
            values
        )

    def _largest_fine_answer(
        self,
        hits: list[dict],
        year: int | None,
    ) -> str | None:

        candidates = []

        for hit in hits:

            text = self._clean_text(
                hit.get(
                    "text"
                )
                or ""
            )

            amount = (
                self._extract_amount(
                    text
                )
            )

            if amount is None:
                continue

            candidates.append(
                (
                    amount,
                    hit,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        amount, hit = (
            candidates[0]
        )

        firm = (
            hit.get(
                "firm"
            )
            or hit.get(
                "firm_normalized"
            )
            or "The firm"
        )

        display_amount = (
            f"£{amount:,.2f}"
            if amount % 1
            else f"£{amount:,.0f}"
        )

        year_text = (
            str(year)
            if year
            else str(
                hit.get(
                    "year"
                )
                or ""
            )
        )

        return (
            f"{firm} received the largest "
            f"fine identified in the retrieved "
            f"FCA evidence for {year_text}, "
            f"with a fine of {display_amount}. "
            f"[S1]"
        )

    # ============================================================
    # CITATION VALIDATION
    # ============================================================

    @staticmethod
    def _citation_numbers(
        answer: str,
    ) -> list[int]:

        return [
            int(value)
            for value in re.findall(
                r"\[S(\d+)\]",
                answer or "",
            )
        ]

    def _citations_valid(
        self,
        answer: str,
        source_count: int,
    ) -> bool:

        numbers = (
            self._citation_numbers(
                answer
            )
        )

        if not numbers:
            return False

        return all(
            1 <= number <= source_count
            for number in numbers
        )

    # ============================================================
    # EVIDENCE QUALITY
    # ============================================================

    def _best_score(
        self,
        hits: list[dict],
    ) -> float:

        if not hits:
            return 0.0

        return max(
            float(
                hit.get(
                    "_evidence_score",
                    hit.get(
                        "score",
                        0.0,
                    ),
                )
            )
            for hit in hits
        )

    def _distinct_cases(
        self,
        hits: list[dict],
    ) -> int:

        cases = set()

        for hit in hits:

            cases.add(
                (
                    str(
                        hit.get(
                            "url"
                        )
                        or hit.get(
                            "source_url"
                        )
                        or ""
                    ),
                    str(
                        hit.get(
                            "firm"
                        )
                        or ""
                    ),
                    str(
                        hit.get(
                            "year"
                        )
                        or ""
                    ),
                )
            )

        return len(
            cases
        )

    def _coverage_score(
        self,
        question: str,
        hits: list[dict],
        analysis,
    ) -> float:

        if not hits:
            return 0.0

        score = 0.0

        # Strong evidence.

        best = self._best_score(
            hits
        )

        score += min(
            0.45,
            best * 0.60,
        )

        # Multiple independent cases.

        cases = (
            self._distinct_cases(
                hits
            )
        )

        if cases >= 3:
            score += 0.25

        elif cases >= 2:
            score += 0.18

        elif cases >= 1:
            score += 0.08

        # Entity coverage.

        if analysis.entities:

            matched = 0

            for entity in (
                analysis.entities
            ):

                if any(
                    self.retriever._metadata_entity_matches(
                        hit,
                        entity,
                    )
                    for hit in hits
                ):
                    matched += 1

            score += (
                0.25
                * (
                    matched
                    / len(
                        analysis.entities
                    )
                )
            )

        # Year coverage.

        if analysis.years:

            found_years = {
                int(
                    hit.get(
                        "year"
                    )
                )
                for hit in hits
                if str(
                    hit.get(
                        "year"
                    )
                    or ""
                ).isdigit()
            }

            coverage = (
                len(
                    found_years
                    & set(
                        analysis.years
                    )
                )
                / len(
                    analysis.years
                )
            )

            score += (
                0.25
                * coverage
            )

        return min(
            1.0,
            score,
        )

    # ============================================================
    # GROUNDED FALLBACK
    # ============================================================

    def _fallback_answer(
        self,
        hits: list[dict],
        reason: str | None = None,
    ) -> str:

        if not hits:

            return (
                "I couldn't find enough relevant "
                "evidence in the indexed FCA documents "
                "to answer that reliably."
            )

        lines = []

        if reason:
            lines.append(
                reason
            )

        lines.append(
            "The closest relevant FCA evidence I found is:"
        )

        for index, hit in enumerate(
            hits[:3],
            start=1,
        ):

            firm = (
                hit.get(
                    "firm"
                )
                or hit.get(
                    "firm_normalized"
                )
                or "Unknown firm"
            )

            year = (
                hit.get(
                    "year"
                )
                or ""
            )

            page = (
                hit.get(
                    "page"
                )
                or ""
            )

            evidence = (
                self._build_evidence(
                    hit.get(
                        "text"
                    )
                    or "",
                    hit,
                    max_chars=500,
                )
            )

            lines.append(
                (
                    f"[S{index}] "
                    f"{firm} "
                    f"({year}, page {page}): "
                    f"{evidence}"
                )
            )

        return "\n\n".join(
            lines
        )

    # ============================================================
    # OLLAMA
    # ============================================================

    def _ollama_generate(
        self,
        prompt: str,
    ) -> str:

        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": settings.llm_num_predict,
                "num_ctx": settings.llm_num_ctx,
            },
            "keep_alive": settings.ollama_keep_alive,
        }

        response = requests.post(
            f"{settings.ollama_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=settings.llm_timeout_seconds,
        )

        response.raise_for_status()

        data = response.json()

        return str(
            data.get(
                "response"
            )
            or ""
        ).strip()

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:

        return f"""
You are an FCA enforcement research assistant.

Answer the user's question ONLY from the supplied FCA evidence.

USER QUESTION:
{question}

FCA EVIDENCE:
{context}

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not infer a fact that is not supported by the evidence.
4. Every factual claim must have one or more citations such as [S1].
5. Use only the supplied source numbers.
6. Do not create citations that do not exist.
7. Keep separate legal entities separate.
8. Do not merge firms merely because their names contain the same group or brand.
9. If comparing years, explicitly distinguish the years.
10. If comparing firms, explicitly distinguish the firms.
11. For Principles, Conduct Rules, SMCR, ICR, DEPP, Decision Notices,
    Final Notices and Tribunal matters, use the exact terminology
    appearing in the evidence.
12. If the evidence does not establish part of the requested answer,
    explicitly say that the supplied FCA evidence does not establish it.
13. Do not fill missing information with general knowledge.
14. Prefer precise FCA wording over generic explanations.
15. For "why" questions, explain the actual misconduct/failure,
    not merely the regulatory guidance surrounding it.
16. For "most common" questions, identify recurring issues only when
    they are supported by multiple independent FCA cases.
17. For numerical questions, preserve the exact amount when available.
18. Keep the answer concise but complete.

Return only the final answer.
""".strip()

    # ============================================================
    # ANSWER
    # ============================================================

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:

        started = time.perf_counter()

        question = str(
            question or ""
        ).strip()

        if not question:

            return {
                "answer": (
                    "Please provide a question."
                ),
                "sources": [],
            }

        # --------------------------------------------------------
        # Analyze
        # --------------------------------------------------------

        analysis = (
            self.retriever.analyze_question(
                question
            )
        )

        print(
            "\n========== QUERY PLAN =========="
        )

        print(
            "Type:",
            analysis.intent.upper(),
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
            "Common issues:",
            analysis.common_issues,
        )

        print(
            "Comparison:",
            analysis.comparison,
        )

        print(
            "================================"
        )

        # --------------------------------------------------------
        # Retrieve
        # --------------------------------------------------------

        retrieval_started = (
            time.perf_counter()
        )

        hits = (
            self.retriever.search(
                question
            )
        )

        retrieval_ms = (
            time.perf_counter()
            - retrieval_started
        ) * 1000

        print(
            f"Retrieval: {retrieval_ms:.0f} ms"
        )

        # --------------------------------------------------------
        # No evidence
        # --------------------------------------------------------

        if not hits:

            answer = (
                self._fallback_answer(
                    [],
                )
            )

            total_ms = (
                time.perf_counter()
                - started
            ) * 1000

            return {
                "answer": answer,
                "sources": [],
                "backend_ms": round(
                    total_ms
                ),
                "generation_ms": 0,
                "total_ms": round(
                    total_ms
                ),
            }

        # --------------------------------------------------------
        # Deterministic largest-fine question
        # --------------------------------------------------------

        normalized_question = (
            question.lower()
        )

        asks_largest = (
            (
                "largest fine"
                in normalized_question
            )
            or (
                "largest penalty"
                in normalized_question
            )
            or (
                "biggest fine"
                in normalized_question
            )
        )

        if asks_largest:

            year = (
                analysis.years[0]
                if len(
                    analysis.years
                ) == 1
                else None
            )

            deterministic = (
                self._largest_fine_answer(
                    hits,
                    year,
                )
            )

            if deterministic:

                sources = (
                    self._build_sources(
                        hits[:1]
                    )
                )

                total_ms = (
                    time.perf_counter()
                    - started
                ) * 1000

                return {
                    "answer": deterministic,
                    "sources": sources,
                    "backend_ms": round(
                        total_ms
                    ),
                    "generation_ms": 0,
                    "total_ms": round(
                        total_ms
                    ),
                }

        # --------------------------------------------------------
        # Evidence coverage
        # --------------------------------------------------------

        coverage = (
            self._coverage_score(
                question,
                hits,
                analysis,
            )
        )

        best_score = (
            self._best_score(
                hits
            )
        )

        print(
            "Evidence coverage:",
            round(
                coverage,
                3,
            ),
        )

        print(
            "Best evidence score:",
            round(
                best_score,
                3,
            ),
        )

        # --------------------------------------------------------
        # Ambiguous comparison
        # --------------------------------------------------------

        if (
            analysis.comparison
            and analysis.ambiguous_entities
            and not analysis.entities
        ):

            answer = (
                self._fallback_answer(
                    hits,
                    reason=(
                        "The question refers to an entity "
                        "name that maps to multiple FCA legal "
                        "entities in the indexed documents. "
                        "I have kept those entities separate "
                        "rather than combining their enforcement "
                        "records."
                    ),
                )
            )

            total_ms = (
                time.perf_counter()
                - started
            ) * 1000

            return {
                "answer": answer,
                "sources": self._build_sources(
                    hits[:3]
                ),
                "backend_ms": round(
                    total_ms
                ),
                "generation_ms": 0,
                "total_ms": round(
                    total_ms
                ),
            }

        # --------------------------------------------------------
        # Evidence too weak
        # --------------------------------------------------------

        if (
            best_score < 0.24
            or coverage < 0.20
        ):

            answer = (
                self._fallback_answer(
                    hits,
                    reason=(
                        "The indexed FCA evidence is "
                        "not sufficiently specific to "
                        "support a reliable synthesized answer."
                    ),
                )
            )

            total_ms = (
                time.perf_counter()
                - started
            ) * 1000

            return {
                "answer": answer,
                "sources": self._build_sources(
                    hits[:3]
                ),
                "backend_ms": round(
                    total_ms
                ),
                "generation_ms": 0,
                "total_ms": round(
                    total_ms
                ),
            }

        # --------------------------------------------------------
        # Context
        # --------------------------------------------------------

        context_hits = hits

        # Common questions need several cases.

        if analysis.common_issues:

            context_hits = hits[
                :min(
                    len(hits),
                    8,
                )
            ]

        else:

            context_hits = hits[
                :min(
                    len(hits),
                    5,
                )
            ]

        context = (
            self._build_context(
                question,
                context_hits,
            )
        )

        if not context:

            answer = (
                self._fallback_answer(
                    hits
                )
            )

            total_ms = (
                time.perf_counter()
                - started
            ) * 1000

            return {
                "answer": answer,
                "sources": self._build_sources(
                    hits[:3]
                ),
                "backend_ms": round(
                    total_ms
                ),
                "generation_ms": 0,
                "total_ms": round(
                    total_ms
                ),
            }

        # --------------------------------------------------------
        # LLM
        # --------------------------------------------------------

        prompt = self._build_prompt(
            question,
            context,
        )

        generation_started = (
            time.perf_counter()
        )

        try:

            generated = (
                self._ollama_generate(
                    prompt
                )
            )

        except Exception as exc:

            print(
                "Ollama error:",
                exc,
            )

            generated = ""

        generation_ms = (
            time.perf_counter()
            - generation_started
        ) * 1000

        # --------------------------------------------------------
        # Validate generated answer
        # --------------------------------------------------------

        if not generated:

            generated = (
                self._fallback_answer(
                    hits
                )
            )

        elif not self._citations_valid(
            generated,
            len(
                context_hits
            ),
        ):

            print(
                "Citation validation failed. "
                "Using grounded fallback."
            )

            generated = (
                self._fallback_answer(
                    hits
                )
            )

        # --------------------------------------------------------
        # Sources
        # --------------------------------------------------------

        sources = (
            self._build_sources(
                context_hits
            )
        )

        total_ms = (
            time.perf_counter()
            - started
        ) * 1000

        print(
            f"Generation: {generation_ms:.0f} ms"
        )

        print(
            f"Total: {total_ms:.0f} ms"
        )

        return {
            "answer": generated,
            "sources": sources,
            "backend_ms": round(
                retrieval_ms
            ),
            "generation_ms": round(
                generation_ms
            ),
            "total_ms": round(
                total_ms
            ),
        }

    # ============================================================
    # SOURCES
    # ============================================================

    @staticmethod
    def _build_sources(
                hits: list[dict],
            ) -> list[dict]:

                sources = []

                for index, hit in enumerate(
                    hits,
                    start=1,
                ):
                    firm = (
                        hit.get("firm")
                        or hit.get("firm_normalized")
                        or "Unknown firm"
                    )

                    year = hit.get("year") or ""
                    page = hit.get("page") or ""

                    url = (
                        hit.get("url")
                        or hit.get("source_url")
                        or ""
                    )

                    # Use an existing title when available.
                    # Otherwise create a deterministic title from
                    # the indexed FCA metadata.
                    title = (
                        hit.get("title")
                        or hit.get("document_title")
                        or hit.get("name")
                        or ""
                    )

                    if not title:
                        if page:
                            title = (
                                f"{firm} - FCA Final Notice "
                                f"({year}, page {page})"
                            )
                        else:
                            title = (
                                f"{firm} - FCA Final Notice "
                                f"({year})"
                            )

                    sources.append(
                        {
                            "id": f"S{index}",
                            "title": str(title),
                            "firm": str(firm),
                            "year": year,
                            "page": page,
                            "url": url,
                        }
                    )

                return sources