import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag.models import AskRequest, AskResponse
from app.rag.store import VectorStore
from app.rag.retriever import Retriever
from app.rag.service import RAGService


app = FastAPI(
    title="FCA Fines RAG Assistant",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.trycloudflare\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# VECTOR STORE
# =========================================================

data_dir = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)

store = VectorStore(str(data_dir))


# =========================================================
# RAG
# =========================================================

retriever = Retriever(store)

service = RAGService(retriever)


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": len(store.metadata),
    }


# =========================================================
# ASK
# =========================================================

@app.post(
    "/api/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):
    started = time.perf_counter()

    try:
        result = service.answer(request.question)

        request_duration_ms = (
            time.perf_counter() - started
        ) * 1000

        # RAGService intentionally returns a dictionary.
        # Keep the API contract here instead of unpacking it
        # as a tuple.
        answer = str(
            result.get("answer", "")
        )

        sources = result.get(
            "sources",
            [],
        )

        response_duration_ms = float(
            result.get(
                "total_ms",
                request_duration_ms,
            )
        )

        return AskResponse(
            answer=answer,
            sources=sources,
            request_duration_ms=request_duration_ms,
            response_duration_ms=response_duration_ms,
        )

    except Exception as exc:
        print(
            f"\nERROR: "
            f"{type(exc).__name__}: "
            f"{exc}\n"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc