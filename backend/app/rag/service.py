import json
import re
import threading
import time
from collections import defaultdict
from typing import Callable

import requests

from app.core.config import settings
from app.rag.entity_resolution import normalize_entity_name
from app.rag.evidence import Evidence
from app.rag.retriever import Retriever
from app.rag.answer_validator import AnswerValidator


FALLBACK_MESSAGE = (
    "I couldn't find enough relevant evidence in the indexed "
    "FCA documents to answer that reliably."
)
AMBIGUOUS_ENTITY_MESSAGE = (
    "I found multiple FCA legal entities matching that name. Please specify one: "
)

CASE_TOPICS = {
    "financial-crime controls": (
        r"financial crime|money laundering|\baml\b|sanctions screening|"
        r"transaction monitoring|customer due diligence",
    ),
    "market abuse and insider dealing": (
        r"market abuse|insider dealing|inside information|market abuse regulation",
    ),
    "transaction reporting": (
        r"transaction reporting|transaction reports|reporting transactions|report transactions|"
        r"failure to report transactions|failed to report transactions",
    ),
    "customer treatment": (
        r"unfair treatment of customers|treatment of customers|customers in financial difficulty|"
        r"customer(s)? in financial difficulty",
    ),
    "systems, controls and governance": (
        r"systems and controls|systems or controls|governance|risk management systems|"
        r"organise and control|organize and control",
    ),
    "prudential requirements": (
        r"prudential|capital requirements|capital adequacy|liquidity requirements",
    ),
}


SYSTEM_PROMPT = """
You are an FCA enforcement research assistant.

Use ONLY the FCA evidence supplied in the prompt.

Rules:

1. Never use outside knowledge.
2. Never invent facts.
3. Never invent fines, dates, firms, breaches, rules,
   regulatory findings or tribunal decisions.
4. Every factual claim must cite one or more supplied source labels such as [S1].
5. Never change a firm's legal name or merge distinct legal entities.
6. Never treat two years as the same event.
7. Cite only source labels present in the evidence.
8. If evidence is insufficient, say exactly:

"I couldn't find enough relevant evidence in the indexed "
"FCA documents to answer that reliably."

9. For comparisons, keep the labelled evidence sections separate.
10. Keep the answer concise.
11. Do not mention information that is not supported by
   the supplied evidence.
"""


OLLAMA_LOCK = threading.Lock()


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


def extract_year(question: str):

    match = re.search(
        r"\b(19\d{2}|20\d{2})\b",
        question,
    )

    return (
        int(match.group(1))
        if match
        else None
    )


def is_largest_question(
    question: str,
) -> bool:

    return bool(
        re.search(
            r"\b("
            r"largest|"
            r"biggest|"
            r"highest|"
            r"maximum"
            r")\b",
            question.lower(),
        )
        and re.search(
            r"\b("
            r"fine|"
            r"fines|"
            r"penalty|"
            r"penalties"
            r")\b",
            question.lower(),
        )
    )


def find_largest_fine(
    metadata: list[dict],
    year: int,
):

    candidates = {}

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

        amount_text = item.get(
            "amount_text"
        )

        amount = parse_amount(
            amount_text
        )

        if amount is None:
            continue

        key = (
            item.get("firm"),
            amount_text,
        )

        candidates[key] = {
            "firm": item.get("firm"),
            "amount": amount,
            "amount_text": amount_text,
            "url": item.get("url"),
            "page": item.get("page"),
            "year": year,
            "title": item.get(
                "title",
                "FCA Final Notice",
            ),
        }

    if not candidates:
        return None

    return max(
        candidates.values(),
        key=lambda item: item["amount"],
    )


def build_context(
    evidence: list[Evidence],
) -> tuple[str, list[Evidence]]:

    if not evidence:
        return "", []

    parts = []
    selected: list[Evidence] = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        text = item.text[:settings.max_chars_per_hit]
        labelled = Evidence(
            source_id=item.source_id,
            firm=item.firm,
            canonical_firm=item.canonical_firm,
            year=item.year,
            title=item.title,
            url=item.url,
            page=item.page,
            text=text,
            date=item.date,
            amount=item.amount,
            amount_text=item.amount_text,
            score=item.score,
        )
        section = labelled.prompt_section(f"S{len(selected) + 1}")
        candidate = "\n\n".join([*parts, section])
        if len(candidate) > settings.max_context_chars:
            remaining = settings.max_context_chars - len("\n\n".join(parts)) - (2 if parts else 0)
            header = section.split("Evidence: ", 1)[0] + "Evidence: "
            if remaining <= len(header):
                break
            labelled = Evidence(
                source_id=item.source_id,
                firm=item.firm,
                canonical_firm=item.canonical_firm,
                year=item.year,
                title=item.title,
                url=item.url,
                page=item.page,
                text=item.text[:remaining - len(header)],
                date=item.date,
                amount=item.amount,
                amount_text=item.amount_text,
                score=item.score,
            )
            section = labelled.prompt_section(f"S{len(selected) + 1}")
            if len("\n\n".join([*parts, section])) > settings.max_context_chars:
                break
        parts.append(section)
        selected.append(labelled)

    return "\n\n".join(parts), selected


def fallback_answer(
    evidence: list[Evidence],
) -> str:

    if not evidence:
        return FALLBACK_MESSAGE

    lines = ["Relevant indexed FCA evidence:"]

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        text = " ".join(item.text.split())[:400]

        lines.append(
            (
                f"[S{index}] {item.firm} "
                f"({item.year}, page {item.page}): "
                f"{text}"
            )
        )

    return "\n".join(lines)


class RAGService:

    def __init__(
        self,
        retriever: Retriever,
    ):

        self.retriever = retriever
        self.store = getattr(retriever, "store", None)
        self.metadata = getattr(self.store, "metadata", [])
        self.validator = AnswerValidator()
        self._chunks_by_case: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
        for row in self.metadata:
            try:
                case_key = (
                    int(row.get("year")),
                    str(row.get("url") or ""),
                    normalize_entity_name(row.get("firm") or row.get("firm_normalized")),
                )
            except (TypeError, ValueError):
                continue
            self._chunks_by_case[case_key].append(row)

    def _source_records(self) -> list[dict]:
        unique: dict[tuple[int | None, str, str], dict] = {}
        for row in self.metadata:
            try:
                year = int(row.get("year"))
            except (TypeError, ValueError):
                continue
            firm = str(row.get("firm") or row.get("firm_normalized") or "")
            url = str(row.get("url") or "")
            if not firm or not url:
                continue
            key = (year, normalize_entity_name(firm), url)
            current = unique.get(key)
            if current is None or int(row.get("page") or 0) < int(current.get("page") or 0):
                unique[key] = row
        return list(unique.values())

    @staticmethod
    def _record_amount(record: dict) -> float | None:
        value = record.get("amount")
        if value is not None:
            try:
                parsed = float(value)
                return parsed if parsed >= 0 else None
            except (TypeError, ValueError):
                return None
        return parse_amount(record.get("amount_text"))

    def _filtered_records(self, plan) -> list[dict]:
        entities = {normalize_entity_name(name) for name in plan.entities}
        years = set(plan.years)
        records = self._source_records()
        selected = []
        for record in records:
            firm = normalize_entity_name(record.get("firm") or record.get("firm_normalized"))
            if entities and firm not in entities:
                continue
            if years and int(record.get("year")) not in years:
                continue
            if plan.entity_type == "bank" and not re.search(
                r"\bbank\b",
                str(record.get("firm") or ""),
                re.IGNORECASE,
            ):
                continue
            selected.append(record)
        return selected

    @staticmethod
    def _evidence(records: list[dict]) -> list[Evidence]:
        return [Evidence.from_record(record) for record in records]

    def _case_text(self, record: dict) -> str:
        key = (
            int(record.get("year")),
            str(record.get("url") or ""),
            normalize_entity_name(record.get("firm") or record.get("firm_normalized")),
        )
        return "\n".join(
            str(chunk.get("text") or "")
            for chunk in self._chunks_by_case.get(key, [])
        )

    def _case_summary_text(self, record: dict) -> str:
        key = (
            int(record.get("year")),
            str(record.get("url") or ""),
            normalize_entity_name(record.get("firm") or record.get("firm_normalized")),
        )
        summary_chunks = []
        for chunk in self._chunks_by_case.get(key, []):
            try:
                page = int(chunk.get("page") or 0)
            except (TypeError, ValueError):
                page = 0
            if 0 < page <= 5:
                summary_chunks.append(str(chunk.get("text") or ""))
        return "\n".join((str(record.get("reason") or ""), *summary_chunks))

    def _topic_matches(self, record: dict, topic: str) -> bool:
        normalized_topic = normalize_entity_name(topic)
        aliases = {
            "financial crime": "financial-crime controls",
            "aml": "financial-crime controls",
            "money laundering": "financial-crime controls",
            "market abuse": "market abuse and insider dealing",
            "insider dealing": "market abuse and insider dealing",
            "transaction reporting": "transaction reporting",
            "customer treatment": "customer treatment",
            "systems and controls": "systems, controls and governance",
            "governance": "systems, controls and governance",
        }
        category = aliases.get(normalized_topic, topic)
        patterns = CASE_TOPICS.get(category)
        case_summary = str(record.get("reason") or "")
        if patterns:
            return bool(re.search(patterns[0], case_summary, re.IGNORECASE))
        return normalized_topic in normalize_entity_name(case_summary)

    def _record_evidence(self, record: dict, *, text: str | None = None) -> Evidence:
        source_record = dict(record)
        if text is not None:
            source_record["text"] = text
        return Evidence.from_record(source_record)

    def _structured_answer(self, plan, records: list[dict]) -> tuple[str, list[Evidence]] | None:
        intent = plan.intent

        if intent == "WHY_FINE":
            if not records:
                return FALLBACK_MESSAGE, []
            if len(records) > 1:
                return None
            record = records[0]
            evidence = [self._record_evidence(
                record,
                text=str(record.get("reason") or "").strip(),
            )]
            if not evidence[0].text:
                return FALLBACK_MESSAGE, []
            firm = record.get("firm") or "The firm"
            year = record.get("year")
            amount = record.get("amount_text")
            fine = f" It was fined {amount}." if amount else ""
            explanation = " ".join(str(record.get("reason") or "").split())
            return (
                f"{firm} was fined in {year}.{fine} The FCA summary states: "
                f"{explanation} [S1]",
                evidence,
            )

        if intent == "PRINCIPLES":
            if not records:
                return FALLBACK_MESSAGE, []
            principles: dict[str, list[dict]] = defaultdict(list)
            for record in records:
                for chunk in self._chunks_by_case.get((
                    int(record.get("year")),
                    str(record.get("url") or ""),
                    normalize_entity_name(record.get("firm") or record.get("firm_normalized")),
                ), []):
                    text = str(chunk.get("text") or "")
                    try:
                        page_number = int(chunk.get("page") or 0)
                    except (TypeError, ValueError):
                        page_number = 0
                    if page_number > 5:
                        continue
                    matches = re.findall(
                        r"\b(?:breached|breaching|breach(?:es)?\s+of)\s+"
                        r"(?:the\s+)?(?:FCA['’]s\s+)?Principles?\s+"
                        r"((?:\d{1,2}[A-Z]?(?:\s*(?:,|and|&)\s*)?)+)",
                        text,
                        re.IGNORECASE,
                    )
                    if matches:
                        for match in matches:
                            codes = re.findall(r"\d{1,2}[A-Z]?", match)
                            for code in codes:
                                principles[code].append(chunk)
            if not principles:
                return FALLBACK_MESSAGE, []
            evidence: list[Evidence] = []
            lines = []
            for code, chunks in sorted(principles.items()):
                chunk = chunks[0]
                evidence.append(self._record_evidence(chunk))
                lines.append(f"Principle {code} is identified in the indexed FCA evidence. [S{len(evidence)}]")
            return "\n".join(lines), evidence

        topic_match = re.search(
            r"\b(?:market abuse|insider dealing|financial crime|money laundering|"
            r"transaction reporting|customer treatment|systems and controls|governance)\b",
            plan.question,
            re.IGNORECASE,
        )
        if topic_match and intent in {"FACT", "BREACH", "GENERAL_SEARCH", "SOURCE_LOOKUP"}:
            topic = topic_match.group(0)
            matches = [record for record in records if self._topic_matches(record, topic)]
            if not matches:
                return FALLBACK_MESSAGE, []
            evidence = [self._record_evidence(
                record,
                text=str(record.get("reason") or self._case_text(record)[:1200]),
            ) for record in matches]
            lines = [
                f"{record.get('firm')} ({record.get('year')}): "
                f"{' '.join(str(record.get('reason') or '').split())} [S{index}]"
                for index, record in enumerate(matches, start=1)
            ]
            return "\n".join(lines), evidence

        firm_list = re.search(
            r"\bwhich\s+(?:firms?|companies|banks?)\s+(?:were\s+)?fined\s+for\s+(.+?)[?.!]*$",
            plan.question,
            re.IGNORECASE,
        )
        if firm_list:
            topic = normalize_entity_name(firm_list.group(1))
            known_topics = {
                "financial crime", "money laundering", "aml", "market abuse",
                "insider dealing", "transaction reporting", "customer treatment",
                "systems and controls", "governance", "prudential requirements",
            }
            business_suffix = re.compile(
                r"\b(plc|limited|ltd|llp|inc|incorporated|corp|corporation|company|bank|society)\b",
                re.IGNORECASE,
            )
            matches = []
            for record in records:
                firm = str(record.get("firm") or "")
                if not business_suffix.search(firm):
                    continue
                reason = normalize_entity_name(record.get("reason") or "")
                if topic in known_topics:
                    matched = self._topic_matches(record, topic)
                else:
                    matched = bool(topic and topic in reason)
                if matched:
                    matches.append(record)
            if not matches:
                return FALLBACK_MESSAGE, []
            evidence = self._evidence(matches)
            lines = [
                f"{record.get('firm')} ({record.get('year')}): "
                f"{str(record.get('reason') or '').strip()} [S{index}]"
                for index, record in enumerate(matches, start=1)
            ]
            return "\n".join(lines), evidence

        if intent == "COMMON_ISSUES":
            bank_records = [
                record for record in records
                if re.search(r"\bbank\b", str(record.get("firm") or ""), re.IGNORECASE)
            ]
            themes: list[tuple[str, list[int]]] = []
            for label in CASE_TOPICS:
                support = [
                    index
                    for index, record in enumerate(bank_records)
                    if self._topic_matches(record, label)
                ]
                if len(support) >= 2:
                    themes.append((label, support))
            if not themes:
                return FALLBACK_MESSAGE, []
            supporting_indexes = sorted({index for _, indexes in themes for index in indexes})
            evidence = self._evidence([bank_records[index] for index in supporting_indexes])
            evidence_index = {record_index: index + 1 for index, record_index in enumerate(supporting_indexes)}
            theme_lines = []
            for phrase, indexes in themes:
                citations = " ".join(
                    f"[S{evidence_index[index]}]"
                    for index in sorted(indexes)
                )
                theme_lines.append(
                    f"{phrase.capitalize()} is present in {len(indexes)} distinct indexed bank cases. {citations}"
                )
            answer = (
                f"Across {len(bank_records)} indexed bank cases, recurring wording in FCA case "
                "summaries includes:\n" + "\n".join(theme_lines)
            )
            return answer, evidence

        candidates = [
            record
            for record in records
            if self._record_amount(record) is not None
        ]
        if intent in {"LARGEST_FINE", "SMALLEST_FINE"}:
            if not candidates:
                return FALLBACK_MESSAGE, []
            reverse = intent == "LARGEST_FINE"
            winner = sorted(
                candidates,
                key=lambda record: self._record_amount(record) or 0,
                reverse=reverse,
            )[0]
            evidence = self._evidence([winner])
            label = "largest" if reverse else "smallest"
            amount = self._record_amount(winner)
            amount_text = winner.get("amount_text") or f"£{amount:,.2f}"
            year_label = f" in {winner.get('year')}"
            answer = (
                f"{winner.get('firm')} received the {label} fine{year_label}, "
                f"{amount_text}. [S1]"
            )
            return answer, evidence

        if intent in {"TOP_N", "TOP_N_TOTAL"}:
            if not candidates:
                return FALLBACK_MESSAGE, []
            count = plan.top_n or 5
            ranked = sorted(
                candidates,
                key=lambda record: self._record_amount(record) or 0,
                reverse=True,
            )[:count]
            if len(ranked) < count:
                return FALLBACK_MESSAGE, []
            evidence = self._evidence(ranked)
            if intent == "TOP_N_TOTAL":
                total = sum(self._record_amount(record) or 0 for record in ranked)
                years = sorted({int(record.get("year")) for record in ranked})
                period = str(years[0]) if len(years) == 1 else "the indexed period"
                lines = [f"The total of the {count} largest indexed fines in {period} is £{total:,.2f}."]
                lines.extend(
                    f"{index}. {record.get('firm')} — {record.get('amount_text') or f'£{self._record_amount(record):,.2f}'} [S{index}]"
                    for index, record in enumerate(ranked, start=1)
                )
            else:
                years = sorted({int(record.get("year")) for record in ranked})
                period = str(years[0]) if len(years) == 1 else "the indexed period"
                lines = [f"The {count} largest indexed fines in {period} are:"]
                lines.extend(
                    f"{index}. {record.get('firm')} — {record.get('amount_text') or f'£{self._record_amount(record):,.2f}'} [S{index}]"
                    for index, record in enumerate(ranked, start=1)
                )
            return "\n".join(lines), evidence

        if intent == "COUNT":
            distinct = {(record.get("year"), record.get("url"), normalize_entity_name(record.get("firm"))) for record in records}
            evidence = self._evidence(records)
            citation = " ".join(f"[S{index}]" for index in range(1, len(evidence) + 1))
            scope = f" in {plan.years[0]}" if len(plan.years) == 1 else ""
            subject = "bank enforcement cases" if plan.entity_type == "bank" else "FCA fine records"
            return f"The indexed data contains {len(distinct)} {subject}{scope}.{(' ' + citation) if citation else ''}", evidence

        if intent == "TOTAL":
            if not candidates:
                return FALLBACK_MESSAGE, []
            total = sum(self._record_amount(record) or 0 for record in candidates)
            evidence = self._evidence(candidates)
            citations = " ".join(f"[S{index}]" for index in range(1, len(evidence) + 1))
            scope = f" for {', '.join(str(year) for year in plan.years)}" if plan.years else ""
            return f"The indexed fines total £{total:,.2f}{scope}.{(' ' + citations) if citations else ''}", evidence

        if intent == "AVERAGE":
            if not candidates:
                return FALLBACK_MESSAGE, []
            average = sum(self._record_amount(record) or 0 for record in candidates) / len(candidates)
            evidence = self._evidence(candidates)
            citations = " ".join(f"[S{index}]" for index in range(1, len(evidence) + 1))
            return f"The average indexed fine is £{average:,.2f} across {len(candidates)} cases. {citations}", evidence

        if intent == "FINE_AMOUNT" and plan.entities:
            if not candidates:
                return FALLBACK_MESSAGE, []
            evidence = self._evidence(candidates)
            lines = []
            for index, record in enumerate(candidates, start=1):
                amount_text = record.get("amount_text") or f"£{self._record_amount(record):,.2f}"
                lines.append(f"{record.get('firm')} ({record.get('year')}): {amount_text}. [S{index}]")
            return "\n".join(lines), evidence

        if intent == "DATE" and plan.entities:
            if not records:
                return FALLBACK_MESSAGE, []
            evidence = self._evidence(records)
            lines = []
            for index, record in enumerate(records, start=1):
                date = record.get("date")
                if date:
                    lines.append(
                        f"{record.get('firm')}: FCA notice dated {date} "
                        f"({record.get('year')}). [S{index}]"
                    )
            if not lines:
                return FALLBACK_MESSAGE, []
            return "\n".join(lines), evidence

        if intent == "FIRM_HISTORY" and plan.entities:
            if not records:
                return FALLBACK_MESSAGE, []
            evidence = self._evidence(records)
            lines = []
            for index, record in enumerate(records, start=1):
                reason = " ".join(str(record.get("reason") or "").split())
                amount = record.get("amount_text")
                summary = f"{record.get('firm')} ({record.get('year')})"
                if amount:
                    summary += f", fine {amount}"
                if reason:
                    summary += f": {reason}"
                lines.append(f"{summary}. [S{index}]")
            return "\n".join(lines), evidence

        if intent == "COMPARISON" and plan.years and any(word in plan.question.casefold() for word in ("fine", "fines", "amount", "penalty")):
            if not candidates or any(
                not any(int(record.get("year")) == year for record in candidates)
                for year in plan.years
            ):
                return FALLBACK_MESSAGE, []
            evidence = self._evidence(candidates)
            lines = []
            for index, record in enumerate(candidates, start=1):
                amount_text = record.get("amount_text") or f"£{self._record_amount(record):,.2f}"
                lines.append(f"{record.get('firm')} ({record.get('year')}): {amount_text}. [S{index}]")
            return "\n".join(lines), evidence

        return None

    def _generate(
        self,
        question: str,
        plan,
        evidence: list[Evidence],
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, float, list[Evidence]]:

        context, evidence = build_context(evidence)
        if plan.intent == "COMPARISON" or len(plan.entities) > 1:
            grouped: dict[str, list[Evidence]] = defaultdict(list)
            for item in evidence:
                grouped[item.canonical_firm].append(item)
            sections = []
            for index, (firm, items) in enumerate(grouped.items()):
                label = "ENTITY A" if index == 0 else "ENTITY B" if index == 1 else f"ENTITY {index + 1}"
                sections.append(f"{label} EVIDENCE: {firm}\n" + "\n\n".join(
                    item.prompt_section(f"S{evidence.index(item) + 1}") for item in items
                ))
            context = "\n\n".join(sections)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"USER QUESTION:\n"
            f"{question}\n\n"
            f"FCA EVIDENCE:\n"
            f"{context}\n\n"
            f"ANSWER:"
        )

        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": on_token is not None,
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

        generation_started = time.perf_counter()
        try:
            with OLLAMA_LOCK:
                if on_token is None:
                    response = requests.post(
                        f"{settings.ollama_url}/api/generate",
                        json=payload,
                        timeout=settings.llm_timeout_seconds,
                    )
                else:
                    response = requests.post(
                        f"{settings.ollama_url}/api/generate",
                        json=payload,
                        timeout=settings.llm_timeout_seconds,
                        stream=True,
                    )

            response.raise_for_status()

            if on_token is None:
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Malformed Ollama response")
                answer = str(data.get("response") or "").strip()
            else:
                pieces: list[str] = []
                try:
                    for raw_line in response.iter_lines(
                        chunk_size=1,
                        decode_unicode=True,
                    ):
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            raise ValueError("Malformed Ollama stream response")
                        piece = str(data.get("response") or "")
                        if piece:
                            pieces.append(piece)
                            on_token(piece)
                finally:
                    response.close()
                answer = "".join(pieces).strip()

            if answer and self.validator.validate(answer, len(evidence)):
                return answer, (time.perf_counter() - generation_started) * 1000, evidence

        except (
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
            ValueError,
        ):
            pass

        return fallback_answer(evidence), (time.perf_counter() - generation_started) * 1000, evidence

    def answer(
        self,
        question: str,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:

        started = time.perf_counter()

        question = (
            question or ""
        ).strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "intent": "GENERAL_SEARCH",
                "backend_ms": 0.0,
                "generation_ms": 0.0,
                "total_ms": 0.0,
                "indexed_chunks": getattr(self.store, "size", 0),
            }
        plan = self.retriever.analyze_question(question)
        enforcement_terms = (
            "fca", "fine", "fined", "penalty", "penalties", "breach", "breached",
            "principle", "principles", "enforcement", "final notice", "market abuse",
            "financial crime", "misconduct", "sanction", "regulatory", "rule", "rules",
        )
        unrelated_general_query = (
            plan.intent == "GENERAL_SEARCH"
            and not plan.entities
            and not plan.years
            and not any(term in question.casefold() for term in enforcement_terms)
        )
        if plan.ambiguous_entities and not plan.comparison:
            answer = (
                AMBIGUOUS_ENTITY_MESSAGE
                + ", ".join(plan.ambiguous_entities)
                + "."
            )
            hits: list[dict] = []
            generation_ms = 0.0
            backend_ms = (time.perf_counter() - started) * 1000
            evidence: list[Evidence] = []
        elif plan.unresolved_entities or unrelated_general_query:
            answer = FALLBACK_MESSAGE
            hits: list[dict] = []
            generation_ms = 0.0
            backend_ms = (time.perf_counter() - started) * 1000
            evidence: list[Evidence] = []
        else:
            backend_started = time.perf_counter()
            records = self._filtered_records(plan)
            structured = self._structured_answer(plan, records)
            if structured is not None:
                answer, evidence = structured
                hits = []
                generation_ms = 0.0
            else:
                hits = self.retriever.search(question)
                evidence = self._evidence(hits)
                if plan.intent == "COMMON_ISSUES":
                    unique_cases = {item.url for item in evidence if item.url}
                    if len(unique_cases) < 2:
                        answer = FALLBACK_MESSAGE
                        evidence = []
                        generation_ms = 0.0
                    elif settings.enable_llm_synthesis:
                        answer, generation_ms, evidence = self._generate(
                            question,
                            plan,
                            evidence,
                            on_token=on_token,
                        )
                    else:
                        answer = fallback_answer(evidence)
                        generation_ms = 0.0
                elif not evidence:
                    answer = FALLBACK_MESSAGE
                    generation_ms = 0.0
                elif settings.enable_llm_synthesis:
                    answer, generation_ms, evidence = self._generate(
                        question,
                        plan,
                        evidence,
                        on_token=on_token,
                    )
                else:
                    answer = fallback_answer(evidence)
                    generation_ms = 0.0
            backend_ms = (time.perf_counter() - backend_started) * 1000 - generation_ms

        citation_numbers = list(dict.fromkeys(
            int(value) for value in re.findall(r"\[S(\d+)\]", answer)
        ))
        cited = [index for index in citation_numbers if 1 <= index <= len(evidence)]
        citation_map = {old: new for new, old in enumerate(cited, start=1)}
        answer = re.sub(
            r"\[S(\d+)\]",
            lambda match: f"[S{citation_map[int(match.group(1))]}]"
            if int(match.group(1)) in citation_map
            else "",
            answer,
        )
        sources = [
            evidence[index - 1].source(f"S{citation_map[index]}")
            for index in cited
        ]
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "answer": answer,
            "sources": sources,
            "intent": plan.intent,
            "backend_ms": max(0.0, backend_ms),
            "generation_ms": generation_ms,
            "total_ms": elapsed,
            "indexed_chunks": getattr(self.store, "size", len(self.metadata)),
        }