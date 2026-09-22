import types

from backend.app.rag.models import AskResponse
from backend.app.rag.service import RAGService


class FakeRetriever:
    def search(self, question):
        return [{
            "score": 0.99,
            "text": "Sample regulatory text.",
            "firm": "Example Firm",
            "year": 2025,
            "page": 42,
            "url": "https://example.com/fca",
            "title": "Example Final Notice"
        }]


def test_rag_service_returns_response_duration(monkeypatch):
    def fake_post(url, json, timeout):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "A grounded answer."}

        return FakeResponse()

    monkeypatch.setattr("backend.app.rag.service.requests.post", fake_post)

    service = RAGService(FakeRetriever())
    answer, sources, response_duration_ms = service.answer("Why was Example Firm fined?")

    assert answer == "A grounded answer."
    assert sources
    assert response_duration_ms >= 0


def test_ask_response_model_accepts_duration_fields():
    payload = AskResponse(
        answer="Answer",
        sources=[],
        request_duration_ms=12.5,
        response_duration_ms=7.2,
    )

    assert payload.request_duration_ms == 12.5
    assert payload.response_duration_ms == 7.2
