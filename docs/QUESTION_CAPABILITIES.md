# FCA RAG Question Capabilities

This guide describes what the currently packaged application can answer from its indexed FCA material. It is a practical boundary, not a guarantee that every wording or every FCA case will be recognized. The active index contains 1,831 chunks, 66 distinct case/source records, and years 2024, 2025, and 2026. The application does not represent the complete FCA website or all FCA enforcement records.

## Strongest Question Types

These paths use structured indexed case records and do not depend on Ollama:

- Fine amount and date for a uniquely identified firm/year. Example: `How much was Barclays Bank plc fined in 2025?`
- Largest or smallest fine within a year. Example: `Which firm received the largest fine in 2025?`
- Ranked top-N fines and sum of the top N. Example: `List the five largest FCA fines in 2025.`
- Total or average fines across the requested indexed scope.
- Counts of indexed records, including bank records where the firm name contains “Bank”. This is a count of indexed case records, not chunks.
- Year comparisons. Example: `Compare Barclays fines in 2024 and 2025.` Each legal entity/year case remains separately sourced.
- Fine “why” questions for a uniquely identified case. The answer uses the indexed annual case reason and structured amount. Example: `Why was Metro Bank fined in 2024?`
- Direct Principles questions where the breach statement appears in the first five indexed pages of the notice. Example: `Which Principles did Barclays Bank plc breach in 2025?`
- Lists of cases classified from indexed case-summary reasons, including financial crime, market abuse/insider dealing, transaction reporting, customer treatment, systems/controls/governance, and prudential terms.
- Common-issue questions return issue categories found in multiple distinct indexed bank case summaries. These are examples of recurring wording in this index, not a statistically complete FCA-wide ranking.
- Date and source questions for records whose annual-page metadata includes a date and source URL.

## Entity and Year Boundaries

Full legal names are resolved separately. Barclays plc, Barclays Bank plc, and Barclays Bank UK plc are not interchangeable. A short name such as `Barclays` can match more than one legal entity; the app asks for a more specific name instead of silently combining cases.

If the requested firm, year, or topic is absent from the packaged index, the app should return the insufficient-evidence response. The current index does not contain Close Brothers Limited or CBAM records. This says nothing about FCA material outside this local dataset.

## Evidence and Citations

Answers return FCA source metadata: firm, year, page, URL, and citation ID. Deterministic calculations cite the exact records used in the calculation. Retrieval answers cite the selected indexed chunks. Check the linked FCA source for high-stakes use; citation-ID validation confirms the source mapping, but is not a formal natural-language entailment proof.

## Ollama Synthesis

Ollama synthesis is disabled by default (`ENABLE_LLM_SYNTHESIS=false`) and the configured model is `llama3.2:3b`. Deterministic answers remain available without it. When synthesis is explicitly enabled and a query reaches generation, Ollama receives one request for that question. `/api/ask/stream` can deliver output incrementally as NDJSON; the final `complete` event carries the validated answer and source/timing metadata. Streamed text is provisional until that event because final citation validation runs after generation and the final answer may be replaced with the evidence fallback. Streaming improves perceived responsiveness only; it does not reduce retrieval/embedding latency, total model generation time, CPU, or RAM. Ollama being unreachable, malformed output, or unusable citations use the established evidence fallback.

## Demo Deployment

The demonstration is hosted on the submitter's Windows PC, with a Cloudflare Quick Tunnel to the local Vite frontend:

```text
Browser -> temporary TryCloudflare URL -> Windows PC
	-> Vite frontend (localhost:5173) -> /api proxy
	-> FastAPI (localhost:8000) -> local FAISS/FCA index
	-> local Ollama llama3.2:3b when synthesis is enabled
```

The operator starts the tunnel separately with `cloudflared tunnel --url http://localhost:5173`. The public demo URL is provided separately with the submission/current TryCloudflare URL; no fixed hostname is embedded in the tracked frontend. The hostname can change when the temporary tunnel restarts. Availability depends on the PC, app processes, local index/model, and tunnel process staying online. This is a demonstration setup, not permanent or 24/7 hosting. Render and Docker artifacts in the repository are not the active Quick Tunnel deployment; Render's free service does not host the local Ollama model.

## Wording That May Need Refinement

- Include a year when possible, especially for fines, dates, and comparisons.
- Include the exact legal entity name when a parent/brand name has multiple FCA records.
- For topic lists, use a recognizable subject phrase such as `market abuse`, `financial crime`, or `transaction reporting`.
- Broad questions about every FCA fine outside 2024–2026, full legal analysis, or records not present in the index cannot be answered reliably by this package.
