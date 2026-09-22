from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from app.rag.entity_resolution import normalize_entity_name


@dataclass(frozen=True)
class Evidence:
    source_id: str
    firm: str
    canonical_firm: str
    year: int | None
    title: str
    url: str
    page: int | None
    text: str
    date: str | None = None
    amount: float | None = None
    amount_text: str | None = None
    score: float | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Evidence":
        firm = str(record.get("firm") or record.get("firm_normalized") or "Unknown firm")
        try:
            year = int(record.get("year")) if record.get("year") is not None else None
        except (TypeError, ValueError):
            year = None
        try:
            page = int(record.get("page")) if record.get("page") is not None else None
        except (TypeError, ValueError):
            page = None
        try:
            amount = float(record["amount"]) if record.get("amount") is not None else None
        except (TypeError, ValueError):
            amount = None

        text = str(record.get("text") or "")
        url = str(record.get("url") or record.get("source_url") or "")
        identity = "|".join(
            (
                str(year or ""),
                normalize_entity_name(firm),
                url,
                str(page or ""),
                text,
            )
        )
        source_id = str(record.get("source_id") or hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16])
        return cls(
            source_id=source_id,
            firm=firm,
            canonical_firm=firm,
            year=year,
            title=str(record.get("title") or f"FCA Final Notice: {firm}"),
            url=url,
            page=page,
            text=text,
            date=str(record.get("date") or "") or None,
            amount=amount,
            amount_text=str(record.get("amount_text") or "") or None,
            score=float(record["score"]) if record.get("score") is not None else None,
        )

    def source(self, citation: str | None = None) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "citation": citation,
            "title": self.title,
            "firm": self.firm,
            "canonical_firm": self.canonical_firm,
            "year": self.year,
            "date": self.date,
            "page": self.page,
            "url": self.url,
            "amount": self.amount,
            "amount_text": self.amount_text,
            "score": self.score,
        }

    def prompt_section(self, label: str) -> str:
        return "\n".join(
            (
                f"[{label}]",
                f"Source ID: {self.source_id}",
                f"Firm: {self.canonical_firm}",
                f"Year: {self.year}",
                f"Date: {self.date}",
                f"Page: {self.page}",
                f"Title: {self.title}",
                f"URL: {self.url}",
                f"Amount: {self.amount_text or self.amount}",
                f"Evidence: {' '.join(self.text.split())}",
            )
        )