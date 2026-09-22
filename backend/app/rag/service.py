import json
import re
import threading
import time
from pathlib import Path

import requests

from app.core.config import settings
from app.rag.retriever import Retriever


FALLBACK_MESSAGE = (
    "I couldn't find enough relevant evidence in the indexed "
    "FCA documents to answer that reliably."
)


SYSTEM_PROMPT = """You are an FCA enforcement research assistant.

Use ONLY the FCA excerpts supplied below.

Do not use outside knowledge.

Do not invent:
- fines
- dates
- firms
- breaches
- regulatory rules
- regulatory findings
- tribunal decisions

Answer the user's question directly.

For each factual claim, cite the supporting source using [S1], [S2], or [S3].

For comparisons, clearly identify which evidence relates to which entity.

If the supplied excerpts do not support an answer, say:

"I couldn't find enough relevant evidence in the indexed FCA documents to answer that reliably."

Keep the answer concise and under 100 words.
"""


OLLAMA_LOCK = threading.Lock()


def load_metadata():
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "metadata.json"
    )

    if not path.exists():
        return []

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        print(
            "Metadata load error:",
            exc,
        )

        return []


def parse_amount(value):
    if not value:
        return None

    cleaned = re.sub(
        r"[^0-9.]",
        "",
        str(value),
    )

    try:
        return float(cleaned)

    except ValueError:
        return None


def find_largest_fine(
    metadata,
    year,
):
    rows = []

    for item in metadata:

        try:
            item_year = int(
                item.get("year")
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if item_year != year:
            continue

        amount = parse_amount(
            item.get("amount_text")
        )

        if amount is None:
            continue

        rows.append(
            {
                "firm": item.get("firm"),
                "amount": amount,
                "amount_text": item.get(
                    "amount_text"
                ),
                "url": item.get("url"),
                "page": item.get("page"),
                "title": item.get(
                    "title",
                    "FCA Final Notice",
                ),
            }
        )

    unique = {}

    for row in rows:

        key = (
            row["firm"],
            row["amount_text"],
        )

        unique[key] = row

    if not unique:
        return None

    return max(
        unique.values(),
        key=lambda x: x["amount"],
    )


def is_largest_question(question):
    return bool(
        re.search(
            r"\b("
            r"largest|"
            r"biggest|"
            r"highest|"
            r"maximum"
            r")\b.*\bfine\b",
            question.lower(),
        )
    )


def extract_year(question):
    match = re.search(
        r"\b(19\d{2}|20\d{2})\b",
        question,
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def build_context(hits):
    if not hits:
        return ""

    per_hit = max(
        400,
        settings.max_context_chars
        // len(hits),
    )

    parts = []

    for index, hit in enumerate(
        hits,
        1,
    ):

        text = (
            hit.get("text")
            or ""
        ).strip()

        text = text[:per_hit]

        parts.append(
            f"[S{index}]\n"
            f"Firm: {hit.get('firm')}\n"
            f"Year: {hit.get('year')}\n"
            f"Page: {hit.get('page')}\n"
            f"Source: {hit.get('url')}\n"
            f"Evidence:\n{text}"
        )

    context = "\n\n".join(
        parts
    )

    return context[
        :settings.max_context_chars
    ]


def fallback_answer(
    question,
    hits,
):
    """
    Deterministic fallback.

    If Ollama times out, the application still returns
    the actual retrieved FCA evidence instead of HTTP 500.
    """

    if not hits:
        return FALLBACK_MESSAGE

    lines = [
        "The following FCA evidence was retrieved. "
        "The local language model did not complete "
        "the synthesis within the configured time limit:"
    ]

    for index, hit in enumerate(
        hits,
        1,
    ):

        text = (
            hit.get("text")
            or ""
        ).strip()

        # Keep fallback readable.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = text[:500]

        lines.append(
            f"\n[S{index}] "
            f"{hit.get('firm')} "
            f"({hit.get('year')}, "
            f"page {hit.get('page')}): "
            f"{text}"
        )

    return "\n".join(
        lines
    )


class RAGService:

    def __init__(
        self,
        retriever: Retriever,
    ):
        self.retriever = retriever

        # Load once rather than reading the JSON
        # for every question.
        self.metadata = load_metadata()

    def answer(
        self,
        question: str,
    ):

        started = time.perf_counter()

        question = question.strip()

        if not question:
            return (
                "Please enter a question.",
                [],
                0,
            )

        # =====================================================
        # DETERMINISTIC AGGREGATION
        # =====================================================

        if is_largest_question(
            question
        ):

            year = extract_year(
                question
            )

            if year is not None:

                largest = find_largest_fine(
                    self.metadata,
                    year,
                )

                if largest:

                    answer = (
                        f"{largest['firm']} "
                        f"received the largest fine "
                        f"in {year}, with a fine of "
                        f"{largest['amount_text']}. [S1]"
                    )

                    sources = [
                        {
                            "title": largest[
                                "title"
                            ],
                            "firm": largest[
                                "firm"
                            ],
                            "year": year,
                            "page": largest[
                                "page"
                            ],
                            "url": largest[
                                "url"
                            ],
                            "score": 1.0,
                        }
                    ]

                    elapsed = (
                        time.perf_counter()
                        - started
                    ) * 1000

                    print(
                        "\nSTRUCTURED RESULT:",
                        answer,
                    )

                    return (
                        answer,
                        sources,
                        elapsed,
                    )

        # =====================================================
        # RETRIEVAL
        # =====================================================

        hits = self.retriever.search(
            question
        )

        # Never throw away all evidence just because
        # FAISS scores are slightly below a threshold.
        #
        # Retriever already performs candidate selection.
        if not hits:

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000

            return (
                FALLBACK_MESSAGE,
                [],
                elapsed,
            )

        # =====================================================
        # CONTEXT
        # =====================================================

        context = build_context(
            hits
        )

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"USER QUESTION:\n"
            f"{question}\n\n"
            f"FCA EVIDENCE:\n"
            f"{context}\n\n"
            f"ANSWER:"
        )

        print(
            "\n========== OLLAMA DEBUG =========="
        )

        print(
            "Model:",
            settings.llm_model,
        )

        print(
            "Context characters:",
            len(context),
        )

        print(
            "Prompt characters:",
            len(prompt),
        )

        print(
            "Evidence chunks:",
            len(hits),
        )

        print(
            "=================================="
        )

        # =====================================================
        # LLM
        # =====================================================

        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": (
                    settings.llm_num_predict
                ),
                "num_ctx": (
                    settings.llm_num_ctx
                ),
            },
            "keep_alive": (
                settings.ollama_keep_alive
            ),
        }

        llm_started = time.perf_counter()

        try:

            with OLLAMA_LOCK:

                response = requests.post(
                    (
                        f"{settings.ollama_url}"
                        "/api/generate"
                    ),
                    json=payload,
                    timeout=(
                        settings.llm_timeout_seconds
                    ),
                )

            response.raise_for_status()

            data = response.json()

            answer = (
                data.get("response")
                or ""
            ).strip()

            llm_elapsed = (
                time.perf_counter()
                - llm_started
            ) * 1000

            print(
                f"Ollama generation: "
                f"{llm_elapsed:.0f} ms"
            )

            # Empty response = fallback.
            if not answer:

                print(
                    "Ollama returned an empty answer."
                )

                answer = fallback_answer(
                    question,
                    hits,
                )

        except requests.exceptions.Timeout:

            llm_elapsed = (
                time.perf_counter()
                - llm_started
            ) * 1000

            print(
                f"Ollama timeout after "
                f"{llm_elapsed:.0f} ms"
            )

            answer = fallback_answer(
                question,
                hits,
            )

        except requests.exceptions.RequestException as exc:

            print(
                "Ollama request error:",
                exc,
            )

            answer = fallback_answer(
                question,
                hits,
            )

        # =====================================================
        # SOURCES
        # =====================================================

        sources = []

        for hit in hits:

            sources.append(
                {
                    "title": hit.get(
                        "title",
                        "FCA Final Notice",
                    ),
                    "firm": hit.get(
                        "firm"
                    ),
                    "year": hit.get(
                        "year"
                    ),
                    "page": hit.get(
                        "page"
                    ),
                    "url": hit.get(
                        "url"
                    ),
                    "score": hit.get(
                        "score"
                    ),
                }
            )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000

        print(
            f"Total response: "
            f"{elapsed:.0f} ms"
        )

        return (
            answer,
            sources,
            elapsed,
        )


# import json
# import re
# import time

# import requests
# import threading

# from app.core.config import settings
# from app.rag.retriever import Retriever


# SYSTEM_PROMPT = """You are an FCA enforcement research assistant.

# Answer using ONLY the supplied FCA document excerpts.

# Do not use outside knowledge.

# If the excerpts do not contain enough evidence, say:
# "I couldn't find enough relevant evidence in the indexed FCA documents to answer that reliably."

# Never invent a fine amount, date, firm, breach, or regulatory finding.

# Cite factual claims using [S1], [S2], etc.

# Keep answers concise but explain the reason for a fine when asked.

# If the question asks for a comparison or common issues, synthesize only from
# the supplied excerpts and make clear when the evidence is limited.
# """

# OLLAMA_LOCK = threading.Lock()


# def find_largest_fine(metadata, year):
#     """Find the largest fine for a specific year from structured metadata."""

#     rows = []

#     for item in metadata:
#         try:
#             if int(item.get("year")) != year:
#                 continue
#         except (TypeError, ValueError):
#             continue

#         amount_text = item.get("amount_text")

#         if not amount_text:
#             continue

#         # Convert values such as:
#         # £44,078,500
#         # £1,107,306.92
#         # into numeric values.
#         cleaned = re.sub(r"[^0-9.]", "", str(amount_text))

#         try:
#             amount = float(cleaned)
#         except ValueError:
#             continue

#         rows.append({
#             "firm": item.get("firm"),
#             "amount": amount,
#             "amount_text": amount_text,
#             "url": item.get("url"),
#             "page": item.get("page"),
#             "title": item.get("title", "FCA Final Notice"),
#         })

#     # Multiple chunks/pages can represent the same fine.
#     # Keep only one record for each firm + fine amount.
#     unique = {}

#     for row in rows:
#         key = (
#             row["firm"],
#             row["amount_text"]
#         )
#         unique[key] = row

#     if not unique:
#         return None

#     return max(
#         unique.values(),
#         key=lambda x: x["amount"]
#     )

# def load_metadata():
#     """Load the processed FCA metadata used by the RAG index."""

#     from pathlib import Path

#     metadata_path = (
#         Path(__file__).resolve().parents[2]
#         / "data"
#         / "processed"
#         / "metadata.json"
#     )

#     if not metadata_path.exists():
#         print("Metadata file not found:", metadata_path)
#         return []

#     try:
#         return json.loads(
#             metadata_path.read_text(encoding="utf-8")
#         )
#     except Exception as exc:
#         print("Metadata load error:", exc)
#         return []

# # def load_metadata():
# #     """Load the processed FCA metadata used by the RAG index."""

# #     metadata_path = settings.data_dir / "metadata.json"

# #     if not metadata_path.exists():
# #         return []

# #     try:
# #         return json.loads(
# #             metadata_path.read_text(encoding="utf-8")
# #         )
# #     except Exception as exc:
# #         print("Metadata load error:", exc)
# #         return []


# class RAGService:
#     def __init__(self, retriever: Retriever):
#         self.retriever = retriever

#     def answer(self, question: str):
#         response_started = time.perf_counter()

#         # ---------------------------------------------------------
#         # STRUCTURED AGGREGATION
#         # ---------------------------------------------------------
#         # Questions asking for the largest fine should be answered
#         # from all indexed metadata, not semantic top-k retrieval.
#         # ---------------------------------------------------------

#         largest_match = re.search(
#             r"\b(largest|biggest|highest|max(?:imum)?)\b.*\bfine\b",
#             question.lower()
#         )

#         year_match = re.search(
#             r"\b(2024|2025|2026)\b",
#             question
#         )

#         if largest_match and year_match:
#             year = int(year_match.group(1))

#             metadata = load_metadata()

#             largest = find_largest_fine(
#                 metadata,
#                 year
#             )

#             if largest:
#                 answer = (
#                     f"{largest['firm']} received the largest fine in "
#                     f"{year}, with a fine of {largest['amount_text']}. [S1]"
#                 )

#                 sources = [{
#                     "title": largest["title"],
#                     "firm": largest["firm"],
#                     "year": year,
#                     "page": largest["page"],
#                     "url": largest["url"],
#                     "score": 1.0
#                 }]

#                 response_duration_ms = (
#                     time.perf_counter() - response_started
#                 ) * 1000

#                 print("\n========== STRUCTURED AGGREGATION ==========")
#                 print("Question:", question)
#                 print("Year:", year)
#                 print("Largest firm:", largest["firm"])
#                 print("Largest fine:", largest["amount_text"])
#                 print("============================================\n")

#                 return (
#                     answer,
#                     sources,
#                     response_duration_ms
#                 )

#         # ---------------------------------------------------------
#         # NORMAL RAG FLOW
#         # ---------------------------------------------------------

#         hits = self.retriever.search(question)

#         hits = [
#             h for h in hits
#             if h["score"] >= settings.min_relevance
#         ]

#         if not hits:
#             response_duration_ms = (
#                 time.perf_counter() - response_started
#             ) * 1000

#             return (
#                 "I couldn't find enough relevant evidence in the indexed "
#                 "FCA documents to answer that reliably.",
#                 [],
#                 response_duration_ms
#             )

#                 # ---------------------------------------------------------
#         # BUILD COMPACT CONTEXT
#         # ---------------------------------------------------------

#         # Divide the total context budget across retrieved sources.
#         per_hit_chars = max(
#             500,
#             settings.max_context_chars // len(hits)
#         )

#         context_parts = []

#         for i, h in enumerate(hits, 1):
#             text = (h.get("text") or "").strip()

#             text = text[:per_hit_chars]

#             context_parts.append(
#                 f"[S{i}] Firm: {h.get('firm')} | "
#                 f"Year: {h.get('year')} | "
#                 f"Page: {h.get('page')}\n"
#                 f"{text}"
#             )

#         context = "\n\n".join(context_parts)

#         # Absolute safety limit.
#         context = context[:settings.max_context_chars]

#         # context_parts = []

#         # per_hit_chars = max(
#         #     settings.max_context_chars // len(hits),
#         #     1200
#         # )

#         # for i, h in enumerate(hits, 1):
#         #     text = h["text"][:per_hit_chars]

#         #     context_parts.append(
#         #         f"[S{i}] Firm: {h.get('firm')} | "
#         #         f"Year: {h.get('year')} | "
#         #         f"Page: {h.get('page')} | "
#         #         f"URL: {h.get('url')}\n"
#         #         f"{text}"
#         #     )

#         # context = "\n\n".join(context_parts)

#         prompt = (
#             f"{SYSTEM_PROMPT}\n\n"
#             f"Question: {question}\n\n"
#             f"FCA excerpts:\n{context}"
#         )

#         print("\n========== OLLAMA DEBUG ==========")
#         print("Model:", settings.llm_model)
#         print("Context characters:", len(context))
#         print("Prompt characters:", len(prompt))
#         print("Number of hits:", len(hits))
#         print("==================================\n")

#         llm_started = time.perf_counter()

#         # response = requests.post(
#         #     f"{settings.ollama_url}/api/generate",
#         #     json={
#         #         "model": settings.llm_model,
#         #         "prompt": prompt,
#         #         "stream": False,
#         #         "options": {
#         #             "temperature": 0.1,
#         #             "num_predict": settings.llm_num_predict,
#         #             "num_ctx": settings.llm_num_ctx
#         #         },
#         #         "keep_alive": settings.ollama_keep_alive,
#         #     },
#         #     timeout=settings.llm_timeout_seconds
#         # )
#         payload = {
#             "model": settings.llm_model,
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "temperature": 0.1,
#                 "num_predict": settings.llm_num_predict,
#                 "num_ctx": settings.llm_num_ctx,
#             },
#             "keep_alive": settings.ollama_keep_alive,
#         }

#         with OLLAMA_LOCK:
#             response = requests.post(
#                 f"{settings.ollama_url}/api/generate",
#                 json=payload,
#                 timeout=settings.llm_timeout_seconds,
#             )
#         response.raise_for_status()

#         llm_duration_ms = (time.perf_counter() - llm_started) * 1000

#         print(
#             f"Ollama generation: {llm_duration_ms:.0f} ms"
#         )
#         # response = requests.post(
#         #     f"{settings.ollama_url}/api/generate",
#         #     json={
#         #         "model": settings.llm_model,
#         #         "prompt": prompt,
#         #         "stream": False,
#         #         "options": {
#         #             "temperature": 0,
#         #             "num_predict": settings.llm_num_predict,
#         #             "num_ctx": settings.llm_num_ctx
#         #         },
#         #         "keep_alive": settings.ollama_keep_alive,
#         #     },
#         #     timeout=settings.llm_timeout_seconds
#         # )

#         # response.raise_for_status()

#         data = response.json()

#         answer = data.get(
#             "response",
#             "No answer generated."
#         )

#         sources = []

#         for h in hits:
#             sources.append({
#                 "title": h.get("title", "FCA Final Notice"),
#                 "firm": h.get("firm"),
#                 "year": h.get("year"),
#                 "page": h.get("page"),
#                 "url": h.get("url"),
#                 "score": h.get("score")
#             })

#         response_duration_ms = (
#             time.perf_counter() - response_started
#         ) * 1000

#         return (
#             answer,
#             sources,
#             response_duration_ms
#         )
