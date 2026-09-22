# FCA RAG Assistant - Technical Walkthrough

## 1. System overview

The current repository implements a local FCA enforcement research assistant with a simple web stack:

Frontend -> FastAPI -> Retriever -> VectorStore/FAISS -> Ollama -> answer + sources

The application uses the processed vector index under `backend/data/processed/` as the runtime knowledge base. The ingestion flow creates that index in a separate script before the API serves queries.

## 2. Repository structure

```text
fca_rag_app/
├── backend/
│   ├── app/
│   │   ├── core/config.py
│   │   ├── main.py
│   │   └── rag/
│   │       ├── chunker.py
│   │       ├── models.py
│   │       ├── retriever.py
│   │       ├── service.py
│   │       └── store.py
│   ├── data/
│   │   ├── processed/
│   │   └── raw/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── scripts/ingest.py
├── frontend/
│   ├── src/main.jsx
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

Important responsibilities by file:

- `backend/app/main.py`: FastAPI app creation, CORS, health route, ask endpoint, request timing.
- `backend/app/core/config.py`: Pydantic settings with model and retrieval defaults.
- `backend/app/rag/retriever.py`: entity extraction, year detection, metadata filtering, and candidate selection.
- `backend/app/rag/store.py`: local SentenceTransformer embedding generation and FAISS index creation/search.
- `backend/app/rag/service.py`: deterministic aggregate checks, context assembly, Ollama prompt, and answer generation.
- `backend/app/rag/chunker.py`: page text chunking with overlap.
- `backend/app/rag/models.py`: `AskRequest` and `AskResponse` models.
- `backend/scripts/ingest.py`: FCA data download and PDF extraction pipeline.
- `frontend/src/main.jsx`: React chat UI and API interaction.

## 3. Application startup

### 3.1 Initialization order in `backend/app/main.py`

`backend/app/main.py` creates the FastAPI app and initializes the runtime objects at import time:

1. `app = FastAPI(title="FCA Fines RAG Assistant", version="1.0.0")`
2. `CORSMiddleware` is added with `allow_origins` set to `http://localhost:5173` and `http://127.0.0.1:5173`.
3. `data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"`
4. `store = VectorStore(str(data_dir))`
5. `retriever = Retriever(store)`
6. `service = RAGService(retriever)`
7. Route definitions: `@app.get("/health")` and `@app.post("/api/ask")`

### 3.2 What `VectorStore.__init__` loads

In `backend/app/rag/store.py`:

- `self.index_path = self.dir / "index.faiss"`
- `self.meta_path = self.dir / "metadata.json"`
- `self.model = SentenceTransformer("all-MiniLM-L6-v2")`
- `self.index = faiss.read_index(...)` if the index exists
- `self.metadata = json.loads(...)` if metadata exists

This means the app assumes the processed index files already exist; it does not build them during startup. The backend does not run an ingestion pipeline on app launch.

### 3.3 Health endpoint

`health()` in `backend/app/main.py` returns:

```python
{
    "status": "ok",
    "indexed_chunks": len(store.metadata),
}
```

The endpoint is `GET /health`.

### 3.4 Ask endpoint

`ask()` in `backend/app/main.py` calls `service.answer(request.question)` and wraps exceptions in an HTTP 500 with a message like `TypeError: ...`.

## 4. Configuration

### 4.1 Active settings

`backend/app/core/config.py` defines `Settings` and sets these defaults:

```python
class Settings(BaseSettings):
    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"

    top_k: int = 3
    retrieval_k: int = 15
    min_relevance: float = 0.20

    max_context_chars: int = 1800
    llm_num_predict: int = 40
    llm_num_ctx: int = 1024
    llm_timeout_seconds: int = 75
```

### 4.2 Where those settings are consumed

- `settings.llm_model`: used in `backend/app/rag/service.py` for the Ollama payload.
- `settings.ollama_url`: used in `backend/app/rag/service.py` to form `http://localhost:11434/api/generate`.
- `settings.ollama_keep_alive`: used as `keep_alive` in the Ollama payload.
- `settings.top_k`: used in `Retriever.search()` as the target number of final hits.
- `settings.retrieval_k`: used in `Retriever.search()` as the internal search width before final selection.
- `settings.max_context_chars`: used in `build_context()` in `backend/app/rag/service.py` to cap the evidence snippet length.
- `settings.llm_num_predict`: used in the Ollama generation payload.
- `settings.llm_num_ctx`: used in the Ollama generation payload.
- `settings.llm_timeout_seconds`: used on the `requests.post(..., timeout=...)` call.

### 4.3 Environment override behavior

The settings class uses `SettingsConfigDict(env_file=".env", extra="ignore")`, so environment values are read from a `.env` file in the current working directory when the backend is started. The repository itself does not include a `.env` template file in the root tree.

## 5. Data ingestion

### 5.1 Script entry point

`backend/scripts/ingest.py` creates a single pipeline:

```python
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
```

The script uses:

- `requests` for HTTP get calls
- `BeautifulSoup` to parse annual FCA pages
- `fitz` (PyMuPDF) to read PDFs

### 5.2 FCA page discovery

`YEARS = [2024, 2025, 2026]` and:

```python
BASE = "https://www.fca.org.uk/news/news-stories/{year}-fines"
```

`scrape_year(year)` calls:

```python
html = requests.get(BASE.format(year=year), timeout=60, headers={"User-Agent": "fca-rag-takehome/1.0"}).text
```

Then it finds the first table and iterates over rows.

### 5.3 Table row representation

Each row becomes a dictionary with:

- `firm`
- `firm_normalized`
- `date`
- `amount_text`
- `amount`
- `reason`
- `url`
- `year`

The key logic is:

```python
def normalize_firm(name):
    return re.sub(r"\s+", " ", name.lower()).strip()
```

and:

```python
def money(value):
    m = re.search(r"£([0-9][0-9,]*(?:\.[0-9]+)?)", value or "")
    return float(m.group(1).replace(",", "")) if m else None
```

### 5.4 PDF download and text extraction

`download_pdf(url, path)` uses `requests.get(..., timeout=90, headers=...)` and writes bytes to disk.

Then `main()` loops every scraped row and does:

```python
pdf_path = pdf_dir / f"{i:03d}.pdf"

download_pdf(row["url"], pdf_path)
doc = fitz.open(pdf_path)

for page_no, page in enumerate(doc, 1):
    text = page.get_text("text")
    meta = {**row, "title": f"FCA Final Notice: {row['firm']}", "page": page_no, "pdf_path": str(pdf_path.relative_to(ROOT))}
    all_chunks.extend(chunk_text(text, meta))
```

### 5.5 Chunking

`backend/app/rag/chunker.py` defines `Chunk` and `chunk_text(...)`:

```python
def chunk_text(text, metadata, max_chars=3500, overlap=450)
```

It:

- cleans repeated whitespace and blank lines
- splits paragraphs on `\n\s*\n`
- accumulates paragraphs into chunk strings
- writes each chunk as a `Chunk` object with a metadata copy
- preserves page metadata such as year, firm, and URL on every chunk

### 5.6 Index build

After extraction, `main()` creates a `VectorStore(str(PROC))` and calls:

```python
store.build(all_chunks)
```

`store.build()` does:

- `vectors = self.embed([c.text for c in chunks])`
- `faiss.normalize_L2(vectors)`
- `self.index = faiss.IndexFlatIP(vectors.shape[1])`
- `self.index.add(vectors)`
- `self.metadata = [{"text": c.text, **c.metadata} for c in chunks]`
- `faiss.write_index(self.index, str(self.index_path))`
- `self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")`

The generated files are:

- `backend/data/processed/index.faiss`
- `backend/data/processed/metadata.json`

## 6. Metadata schema

The metadata written in `store.build()` is the stored JSON for each chunk. It contains the fields from each `Chunk.metadata` plus the chunk text.

Actual metadata fields present in the processed data include:

- `firm`
- `firm_normalized`
- `date`
- `amount_text`
- `amount`
- `reason`
- `url`
- `year`
- `title`
- `page`
- `pdf_path`
- `text`

This is visible in `backend/data/processed/metadata.json` and is used by the retriever and service.

## 7. Embeddings

### 7.1 Model

`backend/app/rag/store.py` uses:

```python
self.model = SentenceTransformer("all-MiniLM-L6-v2")
```

This is a local model and is not an API-backed embedding service.

### 7.2 Query embedding and document embedding

`VectorStore.embed()` returns `np.asarray(vectors, dtype="float32")` from:

```python
self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
```

During search it does:

```python
q = self.embed([query])
faiss.normalize_L2(q)
```

The index is normalized with `faiss.normalize_L2(vectors)` before `IndexFlatIP` is created. The code uses inner-product similarity after normalization, which is effectively cosine-like matching.

## 8. FAISS / vector search

### 8.1 Index type

The active code creates:

```python
self.index = faiss.IndexFlatIP(vectors.shape[1])
```

This is a flat inner-product index.

### 8.2 Search function

`VectorStore.search(query, filters=None, k=8)` does the following:

1. Rejects if `self.index is None` or `self.metadata` is empty.
2. Embeds the input query and normalizes it.
3. Builds `candidate_ids` from metadata filters.
4. Applies filter logic for:
   - `firm_normalized`: substring match on `actual_firm` and `requested_firm`
   - `year`: exact integer equality
   - `years`: membership in a set of integers
   - other keys: exact string equality
5. Reconstructs candidate vectors with `self.index.reconstruct(idx)`
6. Computes `scores = np.dot(candidate_vectors, q[0])`
7. Sorts descending by score
8. Returns metadata rows with an added `score` field

### 8.3 Meaning of `TOP_K`

The actual user-facing target is `settings.top_k`, which is set to `3` in config. `Retriever.search()` uses:

```python
target_k = max(1, settings.top_k)
retrieval_k = max(settings.retrieval_k, target_k * 5)
```

The retriever then selects a final set of `target_k` evidence hits using `select_diverse_hits()`.

## 9. Query processing

### 9.1 Question handling example

For a question like “Why was Barclays fined in 2025?” the flow is:

1. `frontend/src/main.jsx`: `fetch(`${API}/api/ask`, { ... JSON {question} })`
2. `backend/app/main.py`: `ask(request: AskRequest)`
3. `service.answer(question)` in `backend/app/rag/service.py`
4. `Retriever.search(question)` in `backend/app/rag/retriever.py`
5. `VectorStore.search(search_question, filters=..., k=retrieval_k)`
6. `select_diverse_hits(...)` chooses a final evidence set
7. `build_context(hits)` assembles the evidence excerpt
8. `requests.post(.../api/generate)` to Ollama
9. The response is returned as `AskResponse`

### 9.2 Query parsing in retriever

`Retriever.search()` does:

- `entities = extract_entities(question, self.entity_catalogue)`
- `years = extract_years(question)`
- `comparison = detect_comparison(question, entities)`
- `search_question = expand_question(question)`

The `extract_years()` helper uses `YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")`.

The entity catalogue is built from metadata fields `firm_normalized` and `firm` in `build_entity_catalogue()`.

### 9.3 Metadata filter logic

If the question includes a single year and a single entity, the retriever sets:

```python
filters = {"year": year, "firm_normalized": entity}
```

If there are multiple years or multiple entities, it uses `years` or per-entity retrieval logic.

## 10. Company / firm normalization

This code does not use a hardcoded alias map for Barclays. Instead, it builds a dynamic entity catalogue from the metadata and then matches by substring and token overlap.

Relevant functions:

- `normalize(value)` lowercases and strips whitespace
- `tokenize(value)` removes stop words
- `build_entity_catalogue(metadata)`
- `exact_entity_matches(question, catalogue)`
- `fuzzy_entity_matches(question, catalogue)`
- `extract_entities(question, catalogue)`

The normalizer simply normalizes `firm_normalized` values to lowercase strings with repeated whitespace collapsed. The retriever does not create a canonical map such as `Barclays -> Barclays Bank plc`; it relies on metadata values already present in the index and similarity-based text matching.

## 11. Filtering

The actual filters in `VectorStore.search()` are:

- `firm_normalized`: substring match with `requested_firm in actual_firm`
- `year`: exact match `actual_year == int(value)`
- `years`: set membership `actual_year in requested_years`
- non-special keys: exact value equality `str(m.get(key)) == str(value)`

This means filtering is not a strict normalized alias system; it is a metadata match in the existing JSON records.

## 12. LLM implementation

### 12.1 Model and endpoint

The current runtime uses:

- `settings.llm_model = "llama3.2:3b"`
- `settings.ollama_url = "http://localhost:11434"`

The request is sent to:

```python
f"{settings.ollama_url}/api/generate"
```

### 12.2 Request format

`RAGService.answer()` sends:

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

### 12.3 Timeout and fallback behavior

The actual code catches:

- `requests.exceptions.Timeout`
- `requests.exceptions.RequestException`

On failure, it calls `fallback_answer(question, hits)` rather than raising an HTTP error from the service layer.

The fallback message is:

```python
FALLBACK_MESSAGE = (
    "I couldn't find enough relevant evidence in the indexed "
    "FCA documents to answer that reliably."
)
```

## 13. Prompt construction

`SYSTEM_PROMPT` in `backend/app/rag/service.py` is:

```text
You are an FCA enforcement research assistant.

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
```

The full prompt is assembled in `RAGService.answer()` with:

```python
prompt = (
    f"{SYSTEM_PROMPT}\n\n"
    f"USER QUESTION:\n"
    f"{question}\n\n"
    f"FCA EVIDENCE:\n"
    f"{context}\n\n"
    f"ANSWER:"
)
```

## 14. Response generation

The service builds a text answer from the model response and then produces a list of sources from the retrieved hits.

Each source entry includes:

- `title`
- `firm`
- `year`
- `page`
- `url`
- `score`

The service returns:

```python
return answer, sources, elapsed
```

The API then wraps that into the Pydantic `AskResponse` model.

## 15. API endpoints

### 15.1 GET /health

File: `backend/app/main.py`
Function: `health()`

Returns JSON with `status` and `indexed_chunks`.

### 15.2 POST /api/ask

File: `backend/app/main.py`
Function: `ask(request: AskRequest)`

Input model: `AskRequest` in `backend/app/rag/models.py`

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
```

Response model: `AskResponse`

```python
class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    request_duration_ms: float = Field(default=0.0, ge=0)
    response_duration_ms: float = Field(default=0.0, ge=0)
```

If a request fails unexpectedly, it raises `HTTPException(status_code=500, detail=...)`.

## 16. Frontend

`frontend/src/main.jsx` is a React single-file app. It does:

- sets `const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';`
- stores the current question and chat messages in `useState()`
- calls `fetch(`${API}/api/ask`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) })`
- updates the UI with the answer, sources, and durations
- renders each source as an anchor with text like `Firm · Year · p.Page`

The examples included in the UI are:

- “Why was Barclays fined in 2025?”
- “What are the most common issues banks were fined for?”
- “Compare Barclays fines in 2024 and 2025.”
- “Which firm received the largest fine in 2025?”

## 17. Error handling

The actual error handling is quite modest and is concentrated in a few spots:

- `backend/app/main.py`: catches all exceptions in `ask()` and converts them to HTTP 500.
- `backend/app/rag/service.py`: catches `requests.exceptions.Timeout` and `requests.exceptions.RequestException` and falls back to evidence text.
- `backend/scripts/ingest.py`: catches per-PDF errors and prints warnings while continuing with the rest of the annual table.
- `retriever.py` and `store.py` return empty lists when no candidate metadata matches or when the FAISS index is missing.

The project does not implement a sophisticated start-up validation routine or a detailed per-stage retry policy.

## 18. Tests

### 18.1 `tests/test_chunker.py`

Test name: `test_chunking()`

What it verifies:

- `chunk_text()` returns chunks for multi-paragraph content
- metadata survives chunking

### 18.2 `tests/test_api_timing.py`

Test names:

- `test_rag_service_returns_response_duration()`
- `test_ask_response_model_accepts_duration_fields()`

These verify:

- `RAGService.answer()` returns a non-empty answer and a non-negative duration
- `AskResponse` accepts `request_duration_ms` and `response_duration_ms`

Execution command:

```bash
pytest -q
```

## 19. Docker / deployment

### 19.1 Backend Dockerfile

File: `backend/Dockerfile`

It uses:

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

### 19.2 docker-compose

File: `docker-compose.yml`

The file currently contains commented-out service definitions and is not a currently active multi-container runtime definition.

### 19.3 render.yaml

File: `render.yaml`

This project defines a Render web service with the following active values:

- `runtime: python`
- `rootDir: backend`
- `buildCommand: pip install -r requirements.txt`
- `startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- health check path `/health`

It also declares environment variables such as `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_EMBEDDING_MODEL`, but those are not used in the active Python code path. The active backend runtime uses local JSON/FAISS and Ollama instead of the Render env vars shown in the deployment file.

## 20. End-to-end example

### Example 1: “Why was Barclays fined in 2025?”

1. User types the question in `frontend/src/main.jsx`.
2. The React client sends `POST /api/ask` with `{"question": "Why was Barclays fined in 2025?"}`.
3. `backend/app/main.py` receives the request in `ask()`.
4. `service.answer(question)` is called.
5. `Retriever.search(question)` identifies the likely firm and year from the question.
6. `VectorStore.search()` applies metadata filters and runs FAISS similarity search.
7. `select_diverse_hits()` narrows the result set to the strongest evidence chunks.
8. `build_context()` formats the evidence as `[S1] ... [S2] ...` style excerpts.
9. The prompt is sent to Ollama with the configured model and context values.
10. The answer is returned with source metadata in `AskResponse`.
11. The frontend renders the answer and clickable source links.

### Example 2: “What are common issues banks or insurance companies are fined for?”

This is broad and not a single-firm or single-year question. The retriever expands the query with FCA enforcement vocabulary and searches across the stored metadata. The response depends on the semantic match and evidence scoring rather than a strict table aggregation. The key logic is in `expand_question()` and `evidence_score()` in `backend/app/rag/retriever.py`.

## 21. Performance-sensitive areas

The current performance-sensitive stages are:

- sentence-transformer embedding generation in `VectorStore.embed()`
- FAISS search in `VectorStore.search()`
- metadata filter processing before vector search
- prompt construction and context truncation in `build_context()`
- Ollama generation in `service.answer()`

The configuration values controlling runtime performance are:

- `top_k = 3`
- `retrieval_k = 15`
- `max_context_chars = 1800`
- `llm_num_predict = 40`
- `llm_num_ctx = 1024`
- `llm_timeout_seconds = 75`

No other background caching or pre-warm routine is active in the current app startup path.

## 22. Security flow

The actual security surface is small:

- CORS is enabled only for localhost ports 5173 and 127.0.0.1:5173.
- No auth or authorization is implemented.
- `AskRequest` applies `Field(min_length=3, max_length=2000)` validation.
- The app makes outbound HTTP calls to the FCA site and to the local Ollama server.
- There is no built-in secret store beyond `.env`-based settings via Pydantic.

## 23. Known technical issues / findings

### 1. Startup warm-up is not active

- File: `backend/app/main.py`
- Issue: A warm-up Ollama call is present only as commented-out historical code; the active runtime does not call it.
- Impact: first request may pay the model-loading latency.
- Severity: medium

### 2. Metrics and config drift in deployment file

- File: `render.yaml`
- Issue: The Render config lists Gemini environment variables and a different model configuration than the active Python runtime.
- Impact: deployment documentation can be misleading relative to the live app behavior.
- Severity: medium

### 3. `min_relevance` is configured but not enforced as a hard gate

- File: `backend/app/core/config.py` and `backend/app/rag/store.py`
- Issue: `settings.min_relevance` exists, but `VectorStore.search()` does not reject low-scoring results with an explicit threshold.
- Impact: retrieval quality is determined by ranking and candidate selection instead of a strict minimum relevance score.
- Severity: low to medium

### 4. No explicit startup ingestion check

- File: `backend/app/main.py`
- Issue: The app assumes processed files already exist.
- Impact: if `backend/data/processed/index.faiss` or `metadata.json` are missing, the app may load an empty or nonexistent index without a specific startup guard.
- Severity: medium

## 24. Final verification notes

The repository was inspected for the actual runtime implementation, not for a hypothetical design. The walkthrough above is traceable to:

- `backend/app/main.py`
- `backend/app/core/config.py`
- `backend/app/rag/retriever.py`
- `backend/app/rag/store.py`
- `backend/app/rag/service.py`
- `backend/app/rag/chunker.py`
- `backend/scripts/ingest.py`
- `frontend/src/main.jsx`
- `backend/data/processed/metadata.json`

Where the code is silent or not active, the documentation explicitly marks the behavior as not verified or not currently active, instead of inventing a new implementation.
