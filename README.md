# FCA Fines RAG Assistant

This repository contains a local RAG application that answers questions over FCA enforcement data for 2024, 2025, and 2026. The current implementation uses FastAPI, a local SentenceTransformers embedding model, FAISS, and Ollama.

## Project overview

The active app flow in the repository is:

Frontend -> FastAPI API -> Retriever -> VectorStore/FAISS -> Ollama -> answer + sources

The ingestion path is separate:

FCA annual page -> linked PDF download -> PyMuPDF text extraction -> chunking -> embeddings -> FAISS index + metadata

## Features

- Downloads the FCA annual fines pages and linked Final Notice PDFs.
- Parses table rows from the FCA annual pages using BeautifulSoup.
- Extracts PDF text with PyMuPDF.
- Stores per-chunk metadata including firm, year, date, amount, URL, and page number.
- Builds a local FAISS vector index using SentenceTransformer embeddings.
- Applies metadata filters such as year and normalized firm before semantic retrieval.
- Uses Ollama for answer generation from the retrieved FCA excerpts only.
- Exposes a simple React frontend chat UI.

## Architecture

### Runtime components

- Backend API: `backend/app/main.py`
- Retrieval logic: `backend/app/rag/retriever.py`
- RAG answer service: `backend/app/rag/service.py`
- Vector storage and embeddings: `backend/app/rag/store.py`
- Chunking: `backend/app/rag/chunker.py`
- Request/response models: `backend/app/rag/models.py`
- Configuration: `backend/app/core/config.py`
- Frontend: `frontend/src/main.jsx`
- Data ingest: `backend/scripts/ingest.py`
- Tests: `tests/`

### Repository structure

```text
fca_rag_app/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── rag/
│   │   │   ├── chunker.py
│   │   │   ├── models.py
│   │   │   ├── retriever.py
│   │   │   ├── service.py
│   │   │   └── store.py
│   │   └── main.py
│   ├── data/
│   │   ├── processed/
│   │   └── raw/
│   ├── requirements.txt
│   └── scripts/
│       └── ingest.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── index.html
├── docs/
│   └── TECHNICAL_WALKTHROUGH.md
├── tests/
│   ├── test_api_timing.py
│   └── test_chunker.py
├── docker-compose.yml
├── evaluation_questions.json
├── pytest.ini
├── README.md
├── render.yaml
└── .gitignore
```

## Prerequisites

- Python 3.12 is used by the backend Docker image.
- Ollama must be running locally at `http://localhost:11434` unless the environment is changed in the code.
- Node.js for the frontend dev server.
- Local access to the FCA pages used by the ingestion script.

## Configuration

The active settings are defined in `backend/app/core/config.py` and are read from `.env` through `pydantic-settings`.

```python
llm_model = "llama3.2:3b"
ollama_url = "http://localhost:11434"
ollama_keep_alive = "30m"

top_k = 3
retrieval_k = 15
min_relevance = 0.20

max_context_chars = 1800
llm_num_predict = 40
llm_num_ctx = 1024
llm_timeout_seconds = 75
```

Important notes:

- `settings.top_k` is the target number of retrieved evidence hits after selection.
- `settings.retrieval_k` is the internal search width before final hit selection.
- `settings.min_relevance` exists in config but the current code path still uses the `Retriever` and `VectorStore` score logic directly; it is not enforced as a strict gate in `search()`.
- The backend does not currently use `settings.top_k` or `settings.min_relevance` as environment-driven switch logic in the API flow; the actual values are active defaults.

## Local setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell
# .venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

If a `.env` file is used, it should be placed in the backend directory so `pydantic-settings` can read it.

### Ollama

Install and run Ollama, then make sure the default model is available:

```bash
ollama pull llama3.2:3b
```

The current code calls:

```python
requests.post(f"{settings.ollama_url}/api/generate", json=payload, timeout=settings.llm_timeout_seconds)
```

So the active model endpoint is `http://localhost:11434/api/generate`.

### Start the API

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend default API base is:

```js
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

## Data ingestion

The ingestion script is `backend/scripts/ingest.py`.

It does the following:

1. Reads the yearly FCA pages at `https://www.fca.org.uk/news/news-stories/{year}-fines` for 2024, 2025, and 2026.
2. Uses BeautifulSoup to find the page table.
3. Extracts each row as a record with:
   - `firm`
   - `firm_normalized`
   - `date`
   - `amount_text`
   - `amount`
   - `reason`
   - `url`
   - `year`
4. Downloads each linked PDF.
5. Opens each PDF with PyMuPDF (`fitz`).
6. Extracts page text with `page.get_text("text")`.
7. Converts the extracted page text into chunks via `chunk_text()` in `backend/app/rag/chunker.py`.
8. Builds the `VectorStore` and writes the FAISS index to `backend/data/processed/index.faiss` and metadata to `backend/data/processed/metadata.json`.

The actual ingestion flow is intentionally separate from the web API. The repository includes prebuilt processed artifacts under `backend/data/processed/`.

## RAG pipeline

### Embeddings

`backend/app/rag/store.py` initializes:

```python
self.model = SentenceTransformer("all-MiniLM-L6-v2")
```

When a document or query is embedded, it calls:

```python
self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
```

The index is a FAISS `IndexFlatIP` after L2 normalization of vectors:

```python
faiss.normalize_L2(vectors)
self.index = faiss.IndexFlatIP(vectors.shape[1])
```

### Retrieval

`backend/app/rag/retriever.py` loads the metadata and builds a dynamic entity catalogue from entries such as `firm_normalized` and `firm`.

The retriever then:

- extracts years with `YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")`
- extracts entity matches using exact and fuzzy matching
- detects comparisons like “compare”, “versus”, and “between”
- expands the search question with FCA enforcement vocabulary
- applies metadata filters like `year`, `years`, and `firm_normalized`
- searches the FAISS index using `VectorStore.search()`
- re-ranks selected hits using lexical + semantic + enforcement evidence scoring

### LLM generation

`backend/app/rag/service.py` builds a prompt using:

- `SYSTEM_PROMPT`
- the user question
- the retrieved FCA evidence snippets
- instructions to cite sources like `[S1]`, `[S2]`, etc.

Then it sends a JSON payload to Ollama with:

```python
payload = {
    "model": settings.llm_model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": settings.llm_num_predict,
        "num_ctx": settings.llm_num_ctx,
    },
    "keep_alive": settings.ollama_keep_alive,
}
```

This uses `settings.llm_model = "llama3.2:3b"` and `settings.ollama_url = "http://localhost:11434"`.

## API

The backend app exposes the following endpoints in `backend/app/main.py`:

### GET /health

Returns JSON:

```json
{
  "status": "ok",
  "indexed_chunks": 123
}
```

### POST /api/ask

Request body:

```json
{
  "question": "Why was Barclays fined in 2025?"
}
```

Response model (`backend/app/rag/models.py`):

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "FCA Final Notice: Barclays",
      "firm": "Barclays",
      "year": 2025,
      "page": 12,
      "url": "https://...",
      "score": 0.82
    }
  ],
  "request_duration_ms": 42.5,
  "response_duration_ms": 1234.8
}
```

The API returns HTTP 500 with a detailed exception message if a request fails unexpectedly.

## Frontend

The frontend app is a single React file at `frontend/src/main.jsx`.

It:

- uses the question textbox and example prompts
- sends POST requests to `VITE_API_URL/api/ask`
- renders returned answer text and sources
- outlines response timings: backend, generation, and browser total time
- shows each source as a link with the firm, year, and page number

## Tests

The repository currently includes these tests:

- `tests/test_chunker.py`
- `tests/test_api_timing.py`

Executed with:

```bash
pytest -q
```

The active tests verify:

- chunking preserves metadata through `chunk_text()`
- `RAGService.answer()` returns a valid answer and a non-negative response duration
- `AskResponse` accepts duration fields

## Docker and deployment

The repository contains a Dockerfile in `backend/Dockerfile` and a docker-compose file at the project root.

The backend Dockerfile is:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data ./data
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
```

The root `docker-compose.yml` file currently contains commented-out service definitions and is not currently active as a running multi-service configuration.

The repository also contains `render.yaml`, which defines a Render web service using:

- Python runtime
- `rootDir: backend`
- `buildCommand: pip install -r requirements.txt`
- `startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- health check path `/health`

## Technical walkthrough

See [docs/TECHNICAL_WALKTHROUGH.md](docs/TECHNICAL_WALKTHROUGH.md) for the detailed end-to-end walkthrough and code-level trace of the current implementation.

## Known limitations and findings

- The code contains many commented-out historical blocks, including older Ollama and OpenAI-related examples in `backend/app/main.py` and `backend/app/rag/service.py`.
- The application keeps the model warm in the app startup path only in older commented code; the active `main.py` does not call a startup warm-up function.
- The retriever uses metadata filters but the code does not perform a strict `min_relevance` threshold check before returning hits.
- The asynchronous or background ingestion pipeline is not implemented in the active app; ingestion is a reusable script, not a running service.
- The current app depends on local Ollama availability; if Ollama is unavailable or times out, the service deliberately falls back to retrieved evidence instead of failing the request outright.

## Security notes

The repository currently contains these concrete security-related behaviors:

- CORS is enabled in `backend/app/main.py` for `http://localhost:5173` and `http://127.0.0.1:5173`.
- There is no built-in authentication or authorization layer in the active backend API.
- There is no explicit secret management layer beyond environment variables read by `pydantic-settings`.
- The app calls outbound HTTP requests to the FCA site and to the local Ollama endpoint.
- User input is validated by `AskRequest` with `Field(min_length=3, max_length=2000)`.

## Troubleshooting

If the app is not working as expected:

1. Confirm the backend can start and the index exists at `backend/data/processed/index.faiss`.
2. Confirm Ollama is running and the model `llama3.2:3b` is available.
3. Confirm the frontend is not pointing at a different backend port.
4. Confirm the ingestion script has been run at least once to produce the processed FAISS metadata files.
5. Check the server logs for `ERROR:` messages from the FastAPI request handler.

## Important note

This project is a repository-accurate implementation of a local FCA RAG assistant. The codebase is the source of truth; this README documents the behavior that exists today rather than a hypothetical design.
