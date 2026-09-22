from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
    )


class Source(BaseModel):
    source_id: str | None = None
    citation: str | None = None
    canonical_firm: str | None = None
    title: str = "FCA Final Notice"
    firm: str | None = None
    year: int | None = None
    date: str | None = None
    page: int | None = None
    url: str | None = None
    amount: float | None = None
    amount_text: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    intent: str = "GENERAL_SEARCH"
    backend_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    indexed_chunks: int = 0

    # Compatibility fields retained for the current frontend.
    request_duration_ms: float | None = None
    response_duration_ms: float | None = None