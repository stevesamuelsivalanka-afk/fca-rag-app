from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str
    firm: str | None = None
    year: int | None = None
    page: int | None = None
    url: str
    score: float | None = None


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    request_duration_ms: float = Field(
        default=0.0,
        ge=0,
    )
    response_duration_ms: float = Field(
        default=0.0,
        ge=0,
    )