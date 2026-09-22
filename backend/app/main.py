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


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------

data_dir = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)


store = VectorStore(
    str(data_dir)
)


# ---------------------------------------------------------
# RAG SERVICE
# ---------------------------------------------------------

retriever = Retriever(
    store
)

service = RAGService(
    retriever
)


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "indexed_chunks": len(
            store.metadata
        ),
    }


# ---------------------------------------------------------
# ASK
# ---------------------------------------------------------

@app.post(
    "/api/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
):

    started = time.perf_counter()

    try:

        answer, sources, response_duration_ms = (
            service.answer(
                request.question
            )
        )

        request_duration_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return AskResponse(
            answer=answer,
            sources=sources,
            request_duration_ms=(
                request_duration_ms
            ),
            response_duration_ms=(
                response_duration_ms
            ),
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



# import time
# import requests
# from pathlib import Path

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# from app.core.config import settings
# from app.rag.models import AskRequest, AskResponse
# from app.rag.store import VectorStore
# from app.rag.retriever import Retriever
# from app.rag.service import RAGService


# app = FastAPI(
#     title="FCA Fines RAG Assistant",
#     version="1.0.0"
# )


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # backend/app/main.py
# # parents[1] = backend

# store = VectorStore(
#     str(
#         Path(__file__).resolve().parents[1]
#         / "data"
#         / "processed"
#     )
# )

# service = RAGService(
#     Retriever(store)
# )


# @app.on_event("startup")
# def warm_up_ollama():
#     """
#     Load llama3.2:3b into Ollama when FastAPI starts.

#     This avoids making the user's first real question
#     pay the full model-loading cost.
#     """

#     print("\n========== OLLAMA WARM-UP ==========")
#     print(f"Model: {settings.llm_model}")
#     print("Loading Ollama model...")

#     started = time.perf_counter()

#     try:
#         response = requests.post(
#             f"{settings.ollama_url}/api/generate",
#             json={
#                 "model": settings.llm_model,
#                 "prompt": "Reply with OK.",
#                 "stream": False,
#                 "options": {
#                     "num_ctx": 512,
#                     "num_predict": 5,
#                     "temperature": 0,
#                 },
#                 "keep_alive": "30m",
#             },
#             timeout=120,
#         )

#         response.raise_for_status()

#         elapsed = time.perf_counter() - started

#         print(
#             f"Ollama warm-up complete: {elapsed:.2f} seconds"
#         )
#         print("====================================\n")

#     except requests.RequestException as exc:
#         elapsed = time.perf_counter() - started

#         print(
#             f"Ollama warm-up failed after "
#             f"{elapsed:.2f} seconds: {exc}"
#         )

#         # Do NOT stop FastAPI if Ollama warm-up fails.
#         # The normal /api/ask request can still try Ollama.


# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "indexed_chunks": len(store.metadata),
#         "llm_model": settings.llm_model,
#     }


# @app.post("/api/ask", response_model=AskResponse)
# def ask(request: AskRequest):
#     request_started = time.perf_counter()

#     try:
#         answer, sources, response_duration_ms = service.answer(
#             request.question
#         )

#         request_duration_ms = (
#             time.perf_counter() - request_started
#         ) * 1000

#         return AskResponse(
#             answer=answer,
#             sources=sources,
#             request_duration_ms=request_duration_ms,
#             response_duration_ms=response_duration_ms,
#         )

#     except Exception as exc:

#         print(
#             f"\nERROR: {type(exc).__name__}: {exc}\n"
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=f"{type(exc).__name__}: {exc}"
#         ) from exc

# import time
# from pathlib import Path

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# from app.rag.models import AskRequest, AskResponse
# from app.rag.store import VectorStore
# from app.rag.retriever import Retriever
# from app.rag.service import RAGService


# app = FastAPI(
#     title="FCA Fines RAG Assistant",
#     version="1.0.0"
# )


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # backend/app/main.py
# # parents[1] = backend
# store = VectorStore(
#     str(
#         Path(__file__).resolve().parents[1]
#         / "data"
#         / "processed"
#     )
# )

# service = RAGService(
#     Retriever(store)
# )


# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "indexed_chunks": len(store.metadata)
#     }


# @app.post("/api/ask", response_model=AskResponse)
# def ask(request: AskRequest):
#     request_started = time.perf_counter()

#     try:
#         answer, sources, response_duration_ms = service.answer(request.question)
#         request_duration_ms = (time.perf_counter() - request_started) * 1000

#         return AskResponse(
#             answer=answer,
#             sources=sources,
#             request_duration_ms=request_duration_ms,
#             response_duration_ms=response_duration_ms,
#         )

#     except Exception as exc:
#         print(f"\nERROR: {type(exc).__name__}: {exc}\n")

#         raise HTTPException(
#             status_code=500,
#             detail=f"{type(exc).__name__}: {exc}"
#         ) from exc
# from pathlib import Path
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from app.rag.models import AskRequest, AskResponse
# from app.rag.store import VectorStore
# from app.rag.retriever import Retriever
# from app.rag.service import RAGService

# app = FastAPI(title="FCA Fines RAG Assistant", version="1.0.0")
# app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# store = VectorStore(str(Path(__file__).resolve().parents[1] / "data" / "processed"))
# service = RAGService(Retriever(store))

# @app.get("/health")
# def health():
#     return {"status": "ok", "indexed_chunks": len(store.metadata)}

# @app.post("/api/ask", response_model=AskResponse)
# def ask(request: AskRequest):
#     try:
#         answer, sources = service.answer(request.question)
#         return {"answer": answer, "sources": sources}
#     except Exception as exc:
#         print(f"ERROR: {type(exc).__name__}: {exc}")
#     raise HTTPException(
#         status_code=500,
#         detail=f"{type(exc).__name__}: {exc}"
#     ) from exc