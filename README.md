# Internal Knowledge Assistant

An agentic RAG service that answers questions over an internal document corpus
with **grounded, cited answers**, and refuses to answer when the documents don't
support a response. Built with FastAPI, Postgres + pgvector, and AWS Bedrock
(Amazon Titan embeddings + Claude for generation).

This is Phase 1 of a larger build. It delivers the RAG core: ingestion,
retrieval, and grounded generation with citations. Later phases add self-critique
for faithfulness, agentic tool use, and evaluation harnesses (see
[Roadmap](#roadmap)).

---

## What it does

`POST /query` with a question. The service:

1. Embeds the question with Titan Text Embeddings v2.
2. Retrieves the nearest chunks from pgvector by cosine similarity.
3. Applies a **relevance gate** — if nothing clears a similarity threshold, it
   returns `no_answer: true` instead of guessing.
4. Otherwise asks Claude (via the Bedrock Converse API) to answer using **only**
   the retrieved chunks, citing each fact with `[n]` markers.
5. Returns the answer plus a `citations` array resolved from the markers the
   model actually used — each citation carries the document, heading path,
   source file, similarity score, and the exact source snippet.

```jsonc
// POST /query  { "question": "Is multi-factor authentication required?" }
{
  "question": "Is multi-factor authentication required?",
  "answer": "Yes. MFA is mandatory for all Meridian accounts, including email, the code repository, cloud consoles, and the VPN [1]. Engineers with production access must use a FIDO2 hardware key as the second factor; SMS codes are not permitted [1].",
  "no_answer": false,
  "citations": [
    {
      "marker": 1,
      "document_id": "security-policy",
      "document_title": "Meridian Information Security Policy",
      "heading_path": "Meridian Information Security Policy > Authentication and Multi-Factor Authentication",
      "source_path": "corpus/security-policy.md",
      "similarity": 0.62,
      "snippet": "..."
    }
  ],
  "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
  "retrieved_count": 6,
  "latency_ms": 812
}
```

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  POST /query  ───▶ │ FastAPI                                       │
                    │   routes_query → RagService                   │
                    │                                               │
                    │   RagService.answer():                        │
                    │     1. Retriever.retrieve()                   │
                    │          embed_query() ──────────▶ Bedrock (Titan)
                    │          repo.search()  ──────────▶ Postgres/pgvector (HNSW, cosine)
                    │     2. relevance gate (min similarity)        │
                    │     3. build grounded prompt                  │
                    │     4. generate() ────────────────▶ Bedrock (Claude, Converse)
                    │     5. parse [n] markers → citations          │
                    └──────────────────────────────────────────────┘

  Ingestion (offline):  load_corpus → chunk_markdown → embed_texts → upsert
```

### Module layout

| Path | Responsibility |
|------|----------------|
| `app/main.py` | App factory + lifespan (builds pool, repository, service once) |
| `app/config.py` | Typed settings via pydantic-settings |
| `app/api/` | Routers (`/query`, `/health`) and DI providers |
| `app/rag/retriever.py` | Embed query, vector search, relevance gate |
| `app/rag/prompts.py` | Grounding system prompt + source formatting |
| `app/rag/service.py` | Orchestration; citation resolution; no-answer control flow |
| `app/rag/embeddings.py` / `llm.py` | Bedrock Titan / Claude clients (async-wrapped) |
| `app/ingestion/chunker.py` | Structure-aware Markdown chunking |
| `app/ingestion/pipeline.py` | load → chunk → embed → store, idempotent |
| `app/db/repository.py` | All SQL, including the pgvector search |
| `corpus/` | Generated internal-docs corpus (7 documents) |

---

## Design decisions (defensible in interview)

### Chunking strategy — structure-aware, token-bounded, with overlap

The chunker (`app/ingestion/chunker.py`) makes four deliberate choices:

1. **Split on Markdown headings first.** Internal docs are written in sections;
   a heading is a topic boundary. Cutting there keeps each chunk about a single
   subject — one coherent idea per vector, which is what maximises retrieval
   precision. The full heading path (`H1 > H2 > H3`) is kept as metadata *and*
   prepended to the embedded text, so the section's context is part of what gets
   embedded and citations can point to exactly where in a document an answer came
   from.
2. **Pack to a token budget (~450), not characters.** Embedding and generation
   both operate on tokens, and chunk size is a precision/recall dial: chunks that
   are too large blur multiple facts into one vector (hurts precision); chunks
   that are too small strand the context needed to answer (hurts recall). ~450
   tokens ≈ a few paragraphs, a good middle ground for policy prose.
3. **Respect natural boundaries.** Whole paragraphs are packed together, and we
   only fall back to sentence splitting when a single paragraph exceeds the
   budget. We never cut mid-sentence.
4. **Overlap (~60 tokens) between adjacent chunks.** The tail of one chunk is
   carried into the head of the next, so a fact that straddles a boundary is
   still fully present in at least one chunk. Costs a little storage for
   materially better recall.

A guard folds tiny trailing fragments back into the previous chunk **only within
the same section** — never across sections, which would file text under the wrong
heading and corrupt citations. (This exact bug was caught by a unit test during
the build; see `tests/test_chunker.py`.)

Tokenisation uses `tiktoken` (`cl100k_base`) as a fast approximation. Titan and
Claude tokenise slightly differently, but for *sizing* chunks we only need
consistent, roughly model-scale counts.

**When would I change this?** For code or tables, structure-aware splitting on
different delimiters. For very large, uniform documents, a semantic/recursive
splitter. For a corpus with heavy cross-references, I'd add parent-document
retrieval (embed small, return the surrounding parent).

### Embedding choice — Amazon Titan Text Embeddings v2

- **Native to Bedrock**, so it shares credentials and networking with Claude —
  one vendor, one IAM story, no extra key to rotate. That coherence is worth a
  lot operationally for an *internal* tool.
- **Configurable dimensionality** (256/512/1024). We use 1024 for headroom on a
  heterogeneous corpus; a narrow corpus could drop to 512/256 to cut storage and
  speed up ANN search with little quality loss.
- **Returns L2-normalised vectors**, so cosine distance in pgvector maps cleanly
  to `similarity = 1 − distance` and the relevance threshold is interpretable.
- **Query and document text are embedded in the same space** (symmetric model),
  and we embed the same text we store (heading path included) so query/document
  geometry lines up.

**Trade-off / swap path:** Titan is a solid general embedder but not always
top-of-leaderboard for retrieval. The embedding call is isolated behind
`embed_query` / `embed_texts`, so swapping to Cohere Embed (also on Bedrock) or
an open model is a one-file change — plus a re-index, because changing the model
means re-embedding the whole corpus and the `vector(...)` column dimension must
match `EMBEDDING_DIM`.

### Vector store — Postgres + pgvector, HNSW, cosine

- **pgvector** keeps vectors next to relational metadata (documents, chunks,
  future ACLs/audit) in one transactional store — no separate vector DB to
  operate, and re-ingestion is a normal SQL transaction.
- **HNSW index** (`vector_cosine_ops`) gives strong recall/latency at this scale
  with no training step (unlike IVFFlat). At millions of vectors I'd revisit
  IVFFlat tuning or a dedicated ANN service.
- The similarity search is plain SQL in `repository.py` (`ORDER BY embedding <=>
  $1 LIMIT k`) — the pgvector operator is explicit and auditable.

### Grounding & the "I don't know" path

Faithfulness starts at retrieval, not generation. The **relevance gate** in
`Retriever` drops chunks below `RETRIEVAL_MIN_SIMILARITY`; if none survive,
`RagService` **never calls the model** and returns the no-answer sentinel. When
the model does run, the system prompt forbids outside knowledge, mandates `[n]`
citations, and specifies an exact refusal sentence. Citations in the response are
derived from the markers the model *actually used* (`_cited_markers`), so the
`citations` array reflects grounding, not just what was retrieved. Generation
runs at `temperature=0` for faithfulness and reproducibility.

This is the foundation for Phase 2's self-critique loop, which will grade the
answer against its cited sources and retry or downgrade to no-answer on low
faithfulness.

---

## Running it

### Prerequisites

- Python 3.11, Docker Desktop.
- AWS credentials with Bedrock access, and **model access enabled** in the
  Bedrock console for your Titan and Claude models, in your chosen region.
  (Bedrock model availability is region-specific; `us-east-1` has the widest
  selection.)

### 1. Configure

```bash
cp .env.example .env
# edit .env: set AWS_REGION / AWS_PROFILE and BEDROCK_LLM_MODEL_ID to a model
# you have enabled. List what you can call:
#   aws bedrock list-inference-profiles --region us-east-1
```

### 2. Install

```bash
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

### 3. Start Postgres + pgvector

```bash
docker compose up -d
```

The schema and `vector` extension are created automatically on first boot
(`db/init/001_schema.sql`). Postgres is published on host port **5434** (5432 and
5433 were already taken on the build machine — change it in `docker-compose.yml`
and `.env` if needed).

### 4. Verify Bedrock access

```bash
py -3.11 -m scripts.check_bedrock
```

### 5. Ingest the corpus

```bash
py -3.11 -m scripts.ingest          # embeds changed docs only
py -3.11 -m scripts.ingest --force  # re-embed everything
```

### 6. Run the API

```bash
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive OpenAPI UI, or:

```bash
curl -s localhost:8000/query -H "content-type: application/json" \
  -d '{"question":"How many days of annual leave do employees get?"}' | jq
```

---

## Testing

```bash
./.venv/Scripts/python.exe -m pytest
```

The suite covers the parts that determine correctness and can run with **no
Bedrock and no database**: chunking invariants (heading hierarchy, token budget,
overlap, no cross-section merges), prompt construction, citation-marker
extraction, and the no-answer / grounding control flow (using fakes for the
retriever and the LLM). The database layer is verified separately against the
live pgvector container with synthetic embeddings.

---

## What's verified

- ✅ Unit suite green (chunking, prompts, citation parsing, no-answer flow).
- ✅ Corpus (7 docs) chunks cleanly into ~51 heading-tagged chunks.
- ✅ Postgres/pgvector stack builds; schema + `vector` extension initialise.
- ✅ DB write/read path end-to-end: idempotent upsert + cosine search returns an
  exact hit at similarity 1.0 (synthetic embeddings, no Bedrock cost).
- ⏳ Live Bedrock embedding/generation requires your AWS credentials — run
  `scripts/check_bedrock.py`, then ingest, then query.

---

## Roadmap

- **Phase 2 — Faithfulness self-critique.** An LLM-judge pass that grades the
  answer against its cited sources; retry or downgrade to no-answer on low
  faithfulness. Retrieval improvements: hybrid (BM25 + dense) search and
  reranking.
- **Phase 3 — Agentic loop.** Query decomposition, multi-step retrieval, and
  tool use, with a bounded planner.
- **Phase 4 — Evaluation harness.** A seeded Q&A set over this corpus measuring
  retrieval hit-rate, answer faithfulness, and citation correctness in CI.

## Cost & teardown

Bedrock is pay-per-call with no standing infrastructure — there is nothing to
leave running that accrues charges (unlike a managed vector service). To stop the
local database:

```bash
docker compose down       # keep data
docker compose down -v    # drop the volume (forces a fresh ingest next time)
```
