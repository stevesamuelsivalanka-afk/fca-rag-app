# FCA Fines RAG Assistant

A research assistant over the FCA fines and enforcement records packaged under `backend/data/`. It combines a FastAPI API, React frontend, local FAISS index, deterministic metadata-based calculations, semantic retrieval, and optional local Ollama synthesis. It represents only its packaged FCA material, not the complete FCA website or all FCA enforcement records.

## Architecture

```text
Browser
  -> React frontend
  -> FastAPI request validation
  -> query analysis (intent, years, legal entity candidates)
  -> metadata filtering and retrieval/reranking
  -> deterministic case analytics OR FCA evidence retrieval
  -> evidence objects with stable source IDs
  -> optional Ollama llama3.2:3b synthesis
  -> validated final answer and sources mapped from the same evidence
```

Deterministic questions such as amounts, dates, ranks, counts, totals, averages, comparisons, and supported case-summary reasons do not need Ollama. If synthesis is enabled for a retrieval question, `/api/ask/stream` can deliver Ollama output as NDJSON token events followed by a final event containing the validated answer, sources, and timings. The ordinary `/api/ask` endpoint remains JSON.

Ingestion is separate from serving:

```text
FCA annual fines pages -> BeautifulSoup row extraction -> linked document download
  -> PyMuPDF page text -> chunks + case metadata -> local embeddings -> FAISS + metadata.json
```

The runtime validates the index/metadata counts and required fields. `/health` reports the counts, valid state, issues, and indexed years. The embedder loads lazily on the first semantic retrieval or index rebuild. Fine/case analytics use unique `(year, URL, legal entity)` records, not chunk counts.

## Current Corpus

The checked-in index currently contains 1,831 chunks across 66 distinct case/source records for 2024, 2025, and 2026. It includes distinct Barclays plc, Barclays Bank plc, and Barclays Bank UK plc records. It does not contain Close Brothers Limited or a CBAM record; queries requiring those absent records should return the insufficient-evidence response. These counts and coverage do not describe the complete FCA website or all FCA enforcement activity.

Some indexed annual-table entries link to FCA press-release pages rather than a final-notice PDF. Their indexed URL/page are returned as stored, so the source title is descriptive and does not establish that every linked record is a PDF Final Notice.

## Local Setup

Prerequisites: Python 3.12+ (the checked-in environment currently uses Python 3.13), Node.js, and optionally Ollama. Network access to FCA pages is needed for a fresh ingestion run. The processed index is included for local demo use.

### Install backend dependencies

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Optional local Ollama

```powershell
ollama pull llama3.2:3b
ollama serve
```

Ollama is local and uses `llama3.2:3b`. Synthesis is disabled by default (`ENABLE_LLM_SYNTHESIS=false`); deterministic analytics and case-summary answers work without it. To enable synthesis for other evidence questions, set `ENABLE_LLM_SYNTHESIS=true` in the local `backend/.env`. If the model is unavailable or returns invalid/uncited output, the service uses its evidence fallback. The project does not use hosted LLM or embedding APIs.

### Start backend

From `backend/`:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Start frontend

In another PowerShell window, from `frontend/`:

```powershell
npm install
npm run dev
```

The frontend defaults to a same-origin API path. Vite proxies `/api` to `http://localhost:8000`; `VITE_API_URL` may override the API base for another setup. No tunnel URL is hard-coded in the tracked frontend source.

### Verify

From the repository root:

```powershell
backend\.venv\Scripts\python.exe -m pytest -q
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/ask -ContentType 'application/json' -Body '{"question":"Which firm received the largest fine in 2025?"}'
```

## Ingestion

From the repository root, with the backend environment active or by using its Python path:

```powershell
backend\.venv\Scripts\python.exe backend\scripts\ingest.py
```

The script scrapes the 2024, 2025, and 2026 annual FCA fines pages, downloads linked documents, extracts PDF pages and rebuilds `backend/data/processed/index.faiss` and `metadata.json`. Ingestion requires the FCA pages and linked files to remain available. It records per-document download failures as warnings; check the final output and `/health` after ingestion. The current script downloads all rows and re-embeds the corpus on a text/index change; metadata-only changes avoid re-embedding when chunk text/order is unchanged. Rerunning ingestion is intended to replace, not append to, the index.

## Query and Evidence Behavior

- Explicit years filter the canonical entity catalogue and metadata candidates before semantic search.
- Full legal names match exactly after punctuation/Unicode normalization. A short brand name such as “Barclays” can resolve to multiple distinct FCA names; candidates remain separate and are not merged. Fuzzy matching is not used as a final filter.
- Ambiguous short names produce a clarification request rather than blending legal entities. Explicit full legal names are filtered before retrieval.
- Comparison searches are run by entity/year group. Structured year comparisons return each record separately.
- Hybrid retrieval combines metadata filters, lexical overlap, FAISS cosine-like similarity, deterministic scoring, and per-source diversity.
- Largest/smallest, top-N, top-N total, overall total, average, count, exact amount, dates, comparisons, and case-summary “why” answers are calculated from unique structured case records without Ollama.
- Named topic lists and recurring bank-issue summaries use FCA case-summary classifications and distinct case records. “Recurring” means supported by multiple indexed cases; it is not a claim of a statistically complete FCA-wide ranking.
- Every evidence object includes a stable source ID, legal entity, year, title, URL, page, text, and amount where available. API sources are selected by the citation labels in the answer, from those same objects.
- Model output must cite supplied labels. Invalid/uncited generations are replaced with an evidence extract. Citations guarantee source identity and citation-number validity, not a formal proof that every generated paraphrase is entailed; review the cited FCA material for high-stakes use.
- Missing or unresolved entities return: `I couldn't find enough relevant evidence in the indexed FCA documents to answer that reliably.`

### Representative FCA checks

These examples describe the current packaged index, not all FCA records:

| Question | Expected behavior in this index |
|---|---|
| `How much was Barclays Bank plc fined in 2025?` | Barclays Bank plc (2025), £39,314,700 |
| `Which firm received the largest fine in 2025?` | Nationwide Building Society, £44,078,500 |
| `List the five largest FCA fines in 2025.` | Nationwide Building Society; Barclays Bank plc; Monzo Bank Limited; The London Metal Exchange (RIE); Barclays Bank UK plc, in descending amount order |
| `Compare Barclays fines in 2024 and 2025.` | Barclays Bank plc (2024), Barclays plc (2024), Barclays Bank UK plc (2025), and Barclays Bank plc (2025), kept as four separate cases |
| `Why was Metro Bank fined in 2024?` | £16,675,200; indexed summary cites PRIN 3 and associated SYSC systems/control issues concerning financial crime |
| `Which Principles did Barclays Bank plc breach in 2025?` | Principle 2, with a source to the indexed Barclays Bank plc notice |
| `What are the common issues banks were fined for in 2024 and 2025?` | Recurring categories from distinct indexed cases; not a complete or statistically ranked FCA-wide result |
| `What was Close Brothers Limited fined for in 2025?` | Insufficient-evidence fallback because the case is not in the packaged index |

## API Contract

`POST /api/ask` accepts `{"question":"..."}` (3 to 2,000 characters) and returns:

```json
{
  "answer": "Nationwide Building Society received the largest fine in 2025, £44,078,500. [S1]",
  "sources": [{
    "source_id": "...",
    "citation": "S1",
    "firm": "Nationwide Building Society",
    "canonical_firm": "Nationwide Building Society",
    "year": 2025,
    "title": "FCA Final Notice: Nationwide Building Society",
    "url": "https://www.fca.org.uk/news/press-releases/fca-fines-nationwide-44m-failings-financial-crime-controls",
    "page": 1,
    "score": null
  }],
  "intent": "LARGEST_FINE",
  "backend_ms": 1.0,
  "generation_ms": 0.0,
  "total_ms": 2.0,
  "indexed_chunks": 1831
}
```

Timing values vary by machine. `request_duration_ms` and `response_duration_ms` remain as compatibility aliases. `/health` returns `index_valid`, `indexed_chunks`, `metadata_count`, `faiss_count`, `available_years`, and validation issues.

`POST /api/ask/stream` accepts the same JSON request and returns `application/x-ndjson`: a `start` event, zero or more `token` events, and a final `complete` event with the response data above. When synthesis is disabled or the route is deterministic, it may return `start` followed directly by `complete`. Text shown while generating is provisional; citation validation occurs after generation, so the final event may replace it with the evidence fallback. Streaming affects perceived responsiveness only, not retrieval, embedding, total generation time, CPU, or RAM use.

## Configuration

Copy `backend/.env.example` to `backend/.env`. Main settings are grouped by purpose:

- Ollama: `LLM_MODEL`, `OLLAMA_URL`, `OLLAMA_KEEP_ALIVE`, `LLM_NUM_PREDICT`, `LLM_NUM_CTX`, `LLM_TIMEOUT_SECONDS`.
- Model synthesis: `ENABLE_LLM_SYNTHESIS` (defaults to `false`; set `true` only when Ollama is reachable).
- Retrieval: `TOP_K`, `RETRIEVAL_K`, `MIN_RELEVANCE`.
- Context: `MAX_CONTEXT_CHARS`, `MAX_CHARS_PER_HIT`.
- API: `ALLOWED_ORIGINS`.

The settings are defaults and operational controls; they do not replace metadata correctness or entity/year filtering.

## Deployment

### Cloudflare Quick Tunnel demo

The demonstration runs on the submitter's Windows PC. Cloudflare Quick Tunnel provides temporary public ingress; it does not host the application or Ollama. Start the backend and frontend locally, then run Cloudflare Tunnel in another terminal targeting the Vite frontend:

```powershell
cloudflared tunnel --url http://localhost:5173
```

Vite forwards `/api` to local FastAPI at `http://localhost:8000`; FastAPI reads the local FCA index and uses local Ollama only when synthesis is enabled. The public demo URL is provided separately with the submission/current TryCloudflare URL. No fixed hostname is documented because Quick Tunnel URLs are temporary and may change after restart. Availability depends on the Windows PC, backend/frontend processes, local index/model availability, and tunnel process remaining online; this is not 24/7 cloud hosting.

The Vite host allow-list and FastAPI CORS permit TryCloudflare hosts, but this repository does not launch or provision `cloudflared`. `render.yaml`, Dockerfiles, and the commented `docker-compose.yml` are alternative artifacts, not the active Quick Tunnel demo. Render's free service does not include the local Ollama model. No paid API or hosted model is configured by this repository.

The app currently has no authentication or authorization. Anyone who can reach the temporary public tunnel can access the demo API. Do not expose sensitive or non-public data without adding access control. `.env` files are ignored by Git; `.env.example` contains non-secret local defaults. Never commit private `.env` files or credentials.

## Tests

```powershell
backend\.venv\Scripts\python.exe -m pytest -q
```

Tests cover chunking, response shape, stable citations, exact legal-entity filtering, year-aware entity planning, unresolved/ambiguous entity handling, deterministic “why” answers, entity-specific Principles, topic classification, inclusive year ranges, spelled-out top-N, top-five rankings, top-three totals, single-call native Ollama streaming, NDJSON response shape, stream failure fallback, and the checked-in largest-2025 regression that verifies Nationwide's answer and source cannot be Arian Financial LLP.