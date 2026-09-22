import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.rag.models import AskRequest, AskResponse
from app.rag.retriever import Retriever
from app.rag.service import RAGService
from app.rag.store import VectorStore


# ============================================================
# DATA
# ============================================================

data_dir = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)


# ============================================================
# APPLICATION COMPONENTS
# ============================================================

store = VectorStore(
    str(data_dir)
)

retriever = Retriever(
    store
)

service = RAGService(
    retriever
)


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print(
        "========== FCA RAG STARTUP =========="
    )

    print(
        "Indexed chunks:",
        store.size,
    )

    print(
        "LLM:",
        settings.llm_model,
    )

    print(
        "======================================"
    )

    yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_origin_regex=r"https://.*\.trycloudflare\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    validation = store.validation
    return {
        "status": "ok" if validation["valid"] else "degraded",
        "service": settings.app_name,
        "version": settings.app_version,
        "indexed_chunks": store.size,
        "metadata_count": validation["metadata_count"],
        "faiss_count": validation["faiss_count"],
        "index_valid": validation["valid"],
        "available_years": validation["available_years"],
        "index_issues": validation["issues"],
        "llm_model": settings.llm_model,
        "ollama_url": settings.ollama_url,
    }


# ============================================================
# ASK
# ============================================================

@app.post(
    "/api/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
):

    started = time.perf_counter()

    try:
        if not store.validation["valid"]:
            raise HTTPException(
                status_code=503,
                detail="The FCA index is missing or invalid.",
            )

        result = service.answer(
            request.question
        )

        request_duration_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            intent=result.get("intent", "GENERAL_SEARCH"),
            backend_ms=round(float(result.get("backend_ms", request_duration_ms)), 2),
            generation_ms=round(float(result.get("generation_ms", 0.0)), 2),
            total_ms=round(float(result.get("total_ms", request_duration_ms)), 2),
            indexed_chunks=int(result.get("indexed_chunks", store.size)),
            request_duration_ms=round(float(result.get("backend_ms", request_duration_ms)), 2),
            response_duration_ms=round(float(result.get("generation_ms", 0.0)), 2),
        )

    except HTTPException:
        raise
    except Exception as exc:

        print(
            f"ERROR "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal RAG service error.",
        ) from exc


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest):
    if not store.validation["valid"]:
        raise HTTPException(
            status_code=503,
            detail="The FCA index is missing or invalid.",
        )

    events: queue.Queue[dict | None] = queue.Queue()

    def run_answer() -> None:
        started = time.perf_counter()
        try:
            result = service.answer(
                request.question,
                on_token=lambda text: events.put({"type": "token", "text": text}),
            )
            duration = (time.perf_counter() - started) * 1000
            response = AskResponse(
                answer=result["answer"],
                sources=result["sources"],
                intent=result.get("intent", "GENERAL_SEARCH"),
                backend_ms=round(float(result.get("backend_ms", duration)), 2),
                generation_ms=round(float(result.get("generation_ms", 0.0)), 2),
                total_ms=round(float(result.get("total_ms", duration)), 2),
                indexed_chunks=int(result.get("indexed_chunks", store.size)),
                request_duration_ms=round(float(result.get("backend_ms", duration)), 2),
                response_duration_ms=round(float(result.get("generation_ms", 0.0)), 2),
            )
            events.put({"type": "complete", "data": response.model_dump()})
        except Exception as exc:
            print(f"STREAM ERROR {type(exc).__name__}: {exc}")
            events.put({
                "type": "error",
                "detail": "The answer stream ended unexpectedly. Please retry your question.",
            })
        finally:
            events.put(None)

    def event_stream():
        worker = threading.Thread(target=run_answer, daemon=True)
        worker.start()
        yield json.dumps({"type": "start"}) + "\n"
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )