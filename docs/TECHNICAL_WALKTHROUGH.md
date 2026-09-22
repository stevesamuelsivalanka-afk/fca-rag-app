# FCA RAG Technical Walkthrough

## Runtime Architecture

```text
React frontend
  -> FastAPI AskRequest validation (/api/ask JSON or /api/ask/stream NDJSON)
  -> deterministic query planner
       intent, candidate entities, years, entity type, operation, topic, fields
  -> canonical entity/year metadata filter
  -> structured analytics OR hybrid retrieval
       lexical candidate scores + FAISS semantic scores + deterministic reranking
  -> Evidence records with stable source IDs
  -> optional Ollama llama3.2:3b synthesis (disabled by default)
  -> citation-label validation
  -> API sources projected from the cited Evidence objects
```

The streaming endpoint wraps the same service call: FastAPI runs one `service.answer()` call on a worker thread. If that question reaches enabled Ollama synthesis, the service makes one native Ollama streaming request and forwards token chunks as NDJSON events. The final `complete` event carries the response fields and sources. `/api/ask` remains the ordinary JSON endpoint. Deterministic questions can return `start` followed directly by `complete` with no token events.

Ingestion is an offline process:

```text
FCA annual fines HTML -> BeautifulSoup table parsing -> FCA linked document download
  -> PyMuPDF text extraction per page -> paragraph chunking
  -> SentenceTransformer vectors -> normalized FAISS IndexFlatIP + metadata.json
```

## Ownership and Data Flow

- `backend/scripts/ingest.py` scrapes 2024-2026 annual pages, retains each row's firm/date/amount/reason/year/URL, downloads each linked file and extracts PDF page text. A per-document error is logged and ingestion continues, so warnings need review before treating a newly rebuilt corpus as complete.
- `backend/app/rag/chunker.py` cleans and chunks extracted page text while copying its case metadata onto each chunk.
- `backend/app/rag/store.py` loads FAISS and metadata once, validates required fields/counts, lazily initializes `all-MiniLM-L6-v2`, exact-filters legal entity/year candidates, combines lexical overlap and semantic similarity, and adds a stable chunk source ID. Rebuilds replace the index/metadata files instead of appending. The embedding model is not loaded for `/health` or metadata-only analytics.
- `backend/app/rag/entity_resolution.py` normalizes Unicode accents and punctuation without stripping legal-name tokens. Full indexed names are exact matches. Prefix matches are candidate discovery only; fuzzy results are suggestion-only and are not retrieval filters. A broad brand query may resolve to multiple distinct entities.
- `backend/app/rag/retriever.py` plans intent, entities, years, entity type, aggregation, topic, and requested fields. Explicit years scope entity candidates. It runs independent searches for entity/year groups, handles broad common-issue searches across multiple source documents, reranks evidence, and returns selected hits.
- `backend/app/rag/service.py` collapses chunk metadata into unique `(year, URL, canonical firm)` cases for deterministic queries. Largest/smallest/top-N/top-N-total/count/total/average, amounts, dates, case-summary “why”, supported Principles, topic lists and comparisons use indexed records without Ollama. Optional synthesis receives selected typed evidence that is also used for API sources. It accepts a token callback only for the streaming route.
- `backend/app/rag/evidence.py` carries source identity, firm/canonical firm, year, title, URL, page, chunk text, and amount. API source objects are produced from the evidence entries named by valid `[S#]` labels; the service does not perform a second retrieval for citations.
- `backend/app/rag/answer_validator.py` rejects empty generations, obvious refusals, missing citations, and citations outside the supplied evidence range. A rejected generation falls back to a short, labelled evidence extract.
- `backend/app/main.py` owns startup, `/health`, JSON `/api/ask`, and NDJSON `/api/ask/stream`. Index problems are exposed in health and ask requests receive a useful 503 instead of a traceback. Stream events are `start`, optional `token`, then `complete`; unexpected worker errors produce a generic `error` event.
- `backend/app/rag/models.py` defines the stable response contract. Legacy timing fields are retained as compatibility aliases.
- `frontend/src/main.jsx` calls `/api/ask/stream` on the same origin by default, displays token text progressively, then replaces it with the final validated answer and sources. Vite proxies `/api` to `http://localhost:8000`. `VITE_API_URL` can override this base, but no tunnel URL is hard-coded in tracked frontend source.

## Index Validation and Health

At construction, `VectorStore` verifies the FAISS file and metadata JSON can be read and checks vector/metadata counts, required metadata fields (`firm`, `firm_normalized`, `year`, `URL`, `page`, `text`), valid years/pages, malformed amounts, empty chunks, duplicate chunks, and orphan counts. `/health` returns:

```json
{
  "status": "ok",
  "indexed_chunks": 1831,
  "metadata_count": 1831,
  "faiss_count": 1831,
  "index_valid": true,
  "available_years": [2024, 2025, 2026],
  "index_issues": []
}
```

Any validation issue marks the index degraded; `/api/ask` returns HTTP 503 rather than silently answering from mismatched files.

## Response and Citation Contract

The service returns a dictionary with `answer`, `sources`, `intent`, `backend_ms`, `generation_ms`, `total_ms`, and `indexed_chunks`. Each source has a `source_id`, answer citation label, canonical/display firm, year, title, URL, page, and optional score. For deterministic largest-fine queries the selected result and citation are generated from the same unique case record. The API does not independently retrieve sources after generation.

The largest-fine regression uses the bundled metadata: answer “Nationwide Building Society, £44,078,500” must cite the Nationwide source URL for 2025/page 1, never Arian Financial LLP. The answer validator checks citation labels and source indices. It is not a general-purpose natural-language entailment verifier; high-stakes users must review the FCA source itself.

Ollama synthesis is disabled by default (`ENABLE_LLM_SYNTHESIS=false`). “Why” answers use the indexed case reason and amount; ranking, top-N, sum-of-top-N, average, count, date, Principles, comparisons and recognized topic lists use deterministic logic. If model synthesis is explicitly enabled, timeout/unavailable/invalid output falls back to cited FCA evidence.

Streamed token text is provisional until the `complete` event. Citation validation runs after generation; the final event may therefore replace interim text with the established evidence fallback. Streaming changes perceived responsiveness only and does not reduce retrieval/embedding time, total LLM generation time, CPU, or RAM use.

## Data Snapshot and Scope

The currently checked-in processed corpus has 1,831 chunks, 66 distinct source records and available years 2024, 2025, 2026. It includes several separate Barclays legal entities and does not contain Close Brothers Limited or CBAM. Therefore a question that requires those absent cases returns the exact insufficient-evidence fallback. This is a limit of the packaged index, not a claim about FCA records outside the index.

The count intent uses the indexed records whose firm name contains “Bank” for a bank-only query. It excludes Nationwide Building Society. A general fine/case count describes source records, not chunk count. Annual-page rows sometimes link to FCA press-release pages; returned URLs/pages preserve those stored records and may not represent a PDF Final Notice.

## Local Commands

From repository root, with Python 3.12+ and Node.js installed:

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Ollama is optional for deterministic routes. To enable synthesis, install/run Ollama locally, pull `llama3.2:3b`, then set `ENABLE_LLM_SYNTHESIS=true` in the private backend `.env`. The checked-in default is false.

Start the API in a backend terminal:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Run tests from the repository root:

```powershell
backend\.venv\Scripts\python.exe -m pytest -q
```

Rebuild from FCA sources from the repository root:

```powershell
backend\.venv\Scripts\python.exe backend\scripts\ingest.py
```

## Demonstration Deployment

The demo runs on the submitter's Windows PC and uses a Cloudflare Quick Tunnel for temporary public ingress:

```text
Browser
  -> temporary TryCloudflare URL
  -> cloudflared tunnel targeting localhost:5173
  -> React/Vite frontend
  -> Vite /api proxy to FastAPI on localhost:8000
  -> local FAISS/FCA index
  -> local Ollama llama3.2:3b only when synthesis is enabled
```

Start the backend and frontend with the commands above, then run in a separate terminal:

```powershell
cloudflared tunnel --url http://localhost:5173
```

Provide the current TryCloudflare URL separately with the submission; it is temporary and may change after restart. The PC, backend/frontend processes, local index/model, and tunnel process must remain available. This is not permanent or 24/7 cloud hosting. The repository allows TryCloudflare hosts through Vite and FastAPI CORS but does not provision or launch `cloudflared`.

`render.yaml`, Dockerfiles, and `docker-compose.yml` are alternative artifacts, not the active demo. `docker-compose.yml` is commented out, and Render's free service does not provide local Ollama. There is no configured hosted LLM, paid API, Redis, or external vector database. The demo has no authentication; anyone with its public tunnel URL can reach it.

## Verification Snapshot

At this audit, the bundled index validated at 1,831 vectors and metadata rows, 66 distinct case/source records, and years 2024-2026. The current automated suite covers retrieval-related regressions, deterministic calculations, source/citation mapping, native stream event delivery, one Ollama request per generation, and stream failure fallback. Tests do not establish complete FCA-wide coverage or live Ollama availability.