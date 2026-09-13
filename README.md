# Internal Knowledge Assistant

An agentic retrieval system that answers questions over a company's internal
documents. It gives answers grounded in the documents, cites the exact sources it
used, and refuses to answer when the documents do not support a response. It is
built with FastAPI, PostgreSQL with the pgvector extension, a local embedding
model, and the Anthropic API for generation.

The point of the project is not just to retrieve and generate. It is to treat
model output as something to check rather than trust. There is a real evaluation
layer that scores whether answers follow from their sources and flags the ones
that do not, with measured numbers you can reproduce.

## Contents

- [What it does](#what-it-does)
- [Measured results](#measured-results)
- [Architecture](#architecture)
- [The RAG design decisions](#the-rag-design-decisions)
- [The agent design](#the-agent-design)
- [The evaluation layer](#the-evaluation-layer)
- [Running it locally](#running-it-locally)
- [Deploying to AWS with Terraform](#deploying-to-aws-with-terraform)
- [Production engineering included](#production-engineering-included)
- [What is verified](#what-is-verified)
- [Honest scope: what a full production assistant would add](#honest-scope-what-a-full-production-assistant-would-add)

## What it does

You send a question to `POST /query`. The agent decides which of its tools can
answer, gathers evidence, and writes an answer that cites the evidence. If it
cannot find support, it says so.

This is a real response from the deployed service:

```jsonc
// POST /query  { "question": "How many days of annual leave do employees get, and is MFA required?" }
{
  "question": "How many days of annual leave do employees get, and is MFA required?",
  "answer": "Employees accrue 28 days of paid annual leave per calendar year ... [1]. Yes, MFA is mandatory for all Meridian accounts ... Engineers with production access must use hardware security keys (FIDO2) ... [7].",
  "no_answer": false,
  "citations": [
    { "marker": 1, "source_type": "document", "document_title": "Meridian Employee Handbook",
      "heading_path": "Meridian Employee Handbook > Annual Leave", "similarity": 0.65 },
    { "marker": 7, "source_type": "document", "document_title": "Meridian Information Security Policy",
      "heading_path": "Meridian Information Security Policy > Authentication and Multi-Factor Authentication", "similarity": 0.78 }
  ],
  "model_id": "claude-sonnet-5",
  "steps": [
    "search_documents(query='annual leave days entitlement') -> 6 passage(s)",
    "search_documents(query='multi-factor authentication requirement') -> 6 passage(s)"
  ]
}
```

An out-of-corpus question returns `no_answer: true` with the refusal sentence and
no citations.

There are two query endpoints:

- `POST /query` runs the agent. It can decide whether to retrieve, call more than
  one tool, and take several steps before answering.
- `POST /query/simple` runs a single retrieval followed by one generation, with
  no agent loop. It is kept as a baseline for comparison.

`GET /health` reports whether the database is reachable and how many documents
and chunks are loaded.

## Measured results

These numbers come from running `scripts/evaluate.py` against the real models on
a labeled set of 19 questions over the 7-document corpus, with Claude Sonnet 5.

| Metric | Result | What it means |
|--------|--------|---------------|
| hit@k | 1.00 | the right document was retrieved for every document question |
| recall@k | 1.00 | all relevant documents were found |
| MRR | 1.00 | the right document ranked first every time |
| precision@k | 0.71 | of the chunks retrieved, about 71 percent came from the relevant document |
| behaviour accuracy | 1.00 | answered all 14 answerable questions, refused all 5 unanswerable ones |
| hallucination rate | 0.00 | zero unanswerable questions were answered |
| faithfulness (mean) | 1.00 | all 14 answered cases were judged fully supported by their sources |

Be clear about scope when reading these: this is a small, clean corpus of 7
documents and 19 questions. The numbers show the system is built correctly and
behaves as intended. They are not a claim that the same scores hold on a large,
messy production corpus. The value here is the method: the quality is measured on
a fixed reference, not assumed.

One finding worth calling out, because it is the kind of thing this layer exists
to catch. The first run flagged two unanswerable questions as false answers, yet
the faithfulness judge marked them supported. On inspection the model had given
grounded refusals that first explain what the documents do cover, then end with
the refusal sentence, for example "the documents describe medical insurance but
not a dental provider. I don't know based on the available documents." Those are
correct refusals, not hallucinations. The refusal detector was too strict, so it
was changed to recognise the sentence at the end of a grounded refusal, and the
run was repeated. That is a real example of checking the output rather than
trusting the first metric.

## Architecture

A question flows through the agent, which uses retrieval and a structured lookup,
then generation. Ingestion is a separate offline path that fills the vector
store.

```mermaid
flowchart LR
    Q[Question] --> API[FastAPI /query]
    API --> AGENT[LangGraph agent]
    AGENT -->|embed query| EMB[Local bge-base embeddings]
    AGENT -->|vector search| PG[(PostgreSQL + pgvector)]
    AGENT -->|structured lookup| DIR[(Directory JSON)]
    AGENT -->|generate answer| LLM[Anthropic API, Claude]
    AGENT --> RESP[Answer + citations + no_answer]

    subgraph Ingestion (offline)
      DOCS[corpus/*.md] --> CHUNK[chunker] --> E2[bge-base embeddings] --> PG
    end
```

### Repository layout

| Path | What lives there |
|------|------------------|
| `app/main.py` | App factory, lifespan, middleware, error handlers |
| `app/config.py` | Typed settings loaded from environment and `.env` |
| `app/api/` | Routes, middleware, error responses |
| `app/rag/retriever.py` | Embed the query, run the vector search, apply the relevance gate |
| `app/rag/prompts.py` | The grounding prompt, source formatting, refusal detection |
| `app/rag/service.py` | The single-shot baseline path |
| `app/rag/llm.py` | Anthropic API calls, including tool use |
| `app/rag/embeddings.py` | Local sentence-transformers embeddings |
| `app/agent/graph.py` | The LangGraph state machine (agent, tools, finalize nodes) |
| `app/agent/tools.py` | The two tools and their executors |
| `app/agent/directory.py` | The structured directory lookup |
| `app/ingestion/chunker.py` | Structure-aware Markdown chunking |
| `app/db/repository.py` | All SQL, including the vector search |
| `app/eval/` | The evaluation layer (dataset, metrics, faithfulness judge, runner) |
| `corpus/` | The internal document corpus (7 documents) |
| `data/directory.json` | The structured org directory |
| `eval/dataset.json` | The labeled evaluation questions |
| `deploy/terraform/` | Terraform for the AWS EC2 deploy |

## The RAG design decisions

### Chunking: split on structure, size by tokens, add overlap

The chunker is in `app/ingestion/chunker.py`. It makes four choices.

1. It splits on Markdown headings first. Internal documents are written in
   sections, and a heading marks a change of topic. Cutting on headings keeps
   each chunk about one subject, which is what a vector search needs, because one
   vector should represent one idea. The full heading path, for example
   "Security Policy > Authentication > MFA", is kept as metadata and placed at the
   top of the chunk before embedding, so the section context is part of what gets
   embedded and citations can say where in a document an answer came from.

2. It packs text up to a token budget of about 450 tokens rather than a character
   count. Embedding and generation both work in tokens. Chunk size is a dial
   between precision and recall. Chunks that are too large put several facts into
   one vector, which lowers precision. Chunks that are too small lose the context
   needed to answer, which lowers recall. About 450 tokens is a few paragraphs,
   which suits policy text.

3. It respects natural boundaries. It packs whole paragraphs and only splits on
   sentences when a single paragraph is larger than the budget. It never cuts in
   the middle of a sentence.

4. It adds an overlap of about 60 tokens between neighbouring chunks, so a fact
   that sits on a chunk boundary is still complete in at least one chunk.

A tiny leftover fragment is folded into the previous chunk only when it belongs
to the same section, never across sections, because that would file text under
the wrong heading and produce a wrong citation. A unit test covers exactly this.

### Embeddings: a local model (bge-base-en-v1.5)

Embeddings turn text into vectors so similar passages sit close together. This
project runs a local model, `BAAI/bge-base-en-v1.5`, at 768 dimensions.

- No extra vendor or API key, and the document text never leaves the server,
  which is the right default for internal documents.
- Free to run and reproducible: the same model file gives the same vectors.
- Strong quality for its size, and small enough to run on a modest CPU box.

The cost of the choice is that it pulls in PyTorch and needs about 1 GB of
memory, so the host must be sized for it. The call sits behind `embed_query` and
`embed_texts`, so switching to a hosted embedding model later is a change to one
file plus a re-index.

One retrieval detail: bge models retrieve better when the query carries a short
instruction prefix and the stored passages do not. The prefix is added to
queries only, which matches how the model was trained.

### Vector store: PostgreSQL with pgvector, HNSW index, cosine distance

- pgvector keeps the vectors next to the ordinary relational data, so documents,
  chunks, and any future access rules live in one database with normal
  transactions. There is no separate vector service to run.
- The index is HNSW with the cosine operator class. HNSW gives good recall and
  latency at this scale and needs no training step.
- The search is plain SQL in `app/db/repository.py`. It orders by the pgvector
  cosine operator and returns the similarity as one minus the distance.

### Grounding: refusing before generating

Faithfulness starts at retrieval, not at the model. The retriever drops any chunk
below a similarity threshold of 0.40. That number was chosen from the data: real
matches score around 0.7 with this model and unrelated text scores around 0.3, so
0.40 separates them cleanly. If nothing clears the threshold, the code does not
call the model and returns a refusal. When the model does run, the prompt tells
it to use only the given sources, cite each fact, and refuse with a fixed
sentence when the sources do not answer the question.

## The agent design

A single model call cannot decide to retrieve, retrieve a second time, or combine
two different sources. Making the assistant agentic means giving the model a loop
where it chooses a tool, sees the result, and chooses again. This project uses
LangGraph to model that loop as a state machine you can read and test. The calls
to Anthropic are the project's own code, so the model choice stays a
configuration value and LangGraph is only the control flow.

```mermaid
flowchart LR
    START([start]) --> agent
    agent -- wants a tool --> tools
    tools --> agent
    agent -- has an answer or hit the budget --> finalize
    finalize --> END([end])
```

### Why each node exists

- The `agent` node is the decision maker. It calls Claude with the two tool
  descriptions. Claude either asks to call a tool or writes a final answer. This
  is the node that decides whether and what to retrieve, and the loop is what
  lets it handle a question with more than one part.
- The `tools` node runs whichever tool the agent asked for, adds the result back
  into the conversation, and records the sources. Keeping tool execution in its
  own node keeps the database and directory access out of the model-facing node
  and makes the tools easy to test.
- The `finalize` node is the grounding check. It decides the `no_answer` flag and
  works out which sources the answer actually cited. This is where the rule "say
  you do not know rather than guess" is enforced in code. If no tool produced a
  usable source, or the model returned a refusal, the answer is forced to a
  refusal.

The edge out of the `agent` node is the routing decision. It loops to `tools`
while the model wants a tool and the step budget remains, and otherwise goes to
`finalize`. The step budget stops a confused agent from retrieving forever.

### The two tools

Two differently shaped sources make tool choice a real decision.

- `search_documents` runs the retriever over the prose corpus.
- `lookup_directory` reads a small structured directory in `data/directory.json`
  holding facts you would not write as prose, such as which team owns a topic,
  the contact channel, the escalation path, who is on call, and office addresses.

The directory match is on whole words, not raw substrings, because an earlier
substring version matched the alias "it" inside "security".

### Citations

Every fact the agent can cite, whether a document chunk or a directory record,
gets a number, handed out in the order sources are gathered across the whole run.
The model cites those numbers, and the finalize node returns only the ones the
answer actually used, so the citation list reflects what the answer relied on,
not just what was retrieved.

## The evaluation layer

This is the part that checks the system rather than trusting it. It lives in
`app/eval/` and runs against `eval/dataset.json`, which has 19 questions in three
kinds: policy questions answered from the documents, directory questions answered
from the structured directory, and unanswerable questions where the correct
behaviour is to refuse. The unanswerable questions are how hallucination is
measured.

For each question the evaluator takes three independent measurements. Running
retrieval separately from the agent is deliberate, because it lets a failure be
blamed on the right layer.

- Retrieval metrics compare the documents the retriever returns against the
  labeled relevant documents: hit@k, recall@k, precision@k, and MRR. See the
  definitions in the results table above.
- Behaviour metrics score the decision to answer or refuse, including the false
  answer rate, which is the hallucination rate on unanswerable questions.
- Faithfulness is judged by a second model call, the LLM-as-judge approach. The
  judge sees only the question, the answer, and the cited sources, and is told to
  use no outside knowledge. It returns a verdict of supported, partial, or
  unsupported, and lists any claim it could not find support for.

This is honest about its limits. The judge is itself a model and can be wrong, so
its prompt is strict and it reports the exact claims it doubts. A stronger setup
would use a different model family as the judge and calibrate it against human
labels. The judge parser never raises on a malformed reply; it records an
"unknown" verdict instead, so one bad judge call cannot crash a run.

The metric math, the judge parser, and the dataset labels are covered by unit
tests that need no models or database, so quality checks run in normal CI. The
full scored run against the real models is a separate script:

```bash
py -3.11 -m scripts.evaluate            # the whole dataset
py -3.11 -m scripts.evaluate --limit 5  # the first 5 cases while iterating
```

## Running it locally

### Prerequisites

- Python 3.11 and Docker.
- An Anthropic API key. If your key is organization-scoped rather than
  workspace-scoped, you also need the workspace id.

### Steps

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID if needed).
```

```bash
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

```bash
docker compose up -d              # PostgreSQL with pgvector on host port 5434
py -3.11 -m scripts.check_models  # confirm the key and the embedding model work
py -3.11 -m scripts.ingest        # embed the corpus into the database
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`, or ask a question directly:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"How many days of annual leave do employees get?"}'
```

The first embedding call downloads the model, about 440 MB, once.

### Tests

```bash
./.venv/Scripts/python.exe -m pytest
```

The suite covers chunking, prompts, refusal detection, citation resolution, the
agent routing and grounding guard, the directory lookup, the evaluation metrics,
the judge parser, the dataset labels, and the error handling, all without models
or a database.

## Deploying to AWS with Terraform

The deploy is defined as code in `deploy/terraform/`. Terraform describes the
cloud resources so the deploy is repeatable and reviewable, and so teardown is
one command. It provisions one EC2 instance in the default VPC, a security group,
and an SSH key. The instance's startup script installs Docker, clones this repo,
writes the `.env`, builds the stack with `docker-compose.prod.yml`, and ingests
the corpus once.

Cost: the instance runs until you destroy it. Left running, expect a small number
of tens of US dollars per month for the size used here. Anthropic usage is billed
per token on top and is cents for a demo. Stand it up, show it, and destroy it
the same day if you want to avoid the standing cost.

```bash
# 1. Generate an SSH key and create your tfvars from the example.
ssh-keygen -t ed25519 -f deploy_key -N ""
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# put your ANTHROPIC_API_KEY, workspace id, and the contents of deploy_key.pub
# into terraform.tfvars

# 2. Provision and deploy.
terraform init
terraform apply

# 3. Terraform prints the app_url. The stack takes a few minutes to build and
#    ingest; poll the health endpoint until corpus_chunks is above zero.
curl http://<public_ip>/health
```

Teardown, which removes everything and stops the cost:

```bash
terraform destroy
```

Secrets and state are gitignored: `terraform.tfvars` holds the key, and the state
file can contain sensitive values, so neither is committed.

An honest note on secret handling. The key is written into the instance startup
script, which is readable by anyone who can describe the instance. That is
acceptable for a demo. A production setup would store the key in AWS SSM Parameter
Store as a SecureString and give the instance a role to read it. That path needs
IAM permissions this deploy intentionally avoids, to keep the required access
small.

## Production engineering included

- Structured logging. Logs are plain text locally and one JSON object per line
  when `JSON_LOGS` is on, so a log system can index the fields. It is on in the
  container.
- Request tracing. Every request gets a short id, taken from an inbound
  `X-Request-ID` header if present or generated otherwise. The id is on the
  logging context, so every line logged during a request carries it, and it is
  returned on the response header.
- Error handling. A dependency failure becomes a 503 the caller can retry, and an
  unexpected error becomes a 500 that leaks no internal detail. Both carry the
  request id, and the raw error is logged on the server, not sent to the client.
- Clean config. All settings are typed and read from the environment or a `.env`
  file in one place, with no scattered environment reads and no hardcoded keys.
- A health endpoint for the platform to check readiness.
- Continuous integration that runs the linter and the full test suite on push,
  with no cloud access needed.

## What is verified

- The unit suite passes: 56 tests.
- The corpus of 7 documents chunks into 51 heading-tagged chunks.
- Ingestion, retrieval, and generation all run for real. Retrieval returns the
  right sections with a clear score gap between real matches and noise.
- The agent routes between the two tools, handles multi-part questions with more
  than one tool call, and refuses out-of-corpus questions.
- The evaluation ran on the full 19-case set and produced the numbers above.
- The service is deployed live on AWS EC2, provisioned by Terraform, and answers
  real questions over HTTP.

## Honest scope: what a full production assistant would add

This is a real working system, deployed and measured, but it is scoped to
demonstrate the RAG core, the agent, and the evaluation. A full internal
assistant used across a company would need more. The main gaps, in rough order of
importance:

- Access control. Real internal documents have per-document permissions. A
  production system must filter retrieval by who is asking, so a person never sees
  a chunk from a document they are not allowed to read. That needs identity,
  authorization, and access-aware retrieval, none of which is here.
- Prompt injection defence. Retrieved documents are untrusted input. A document
  could contain text that tries to steer the model. A production system needs
  handling that treats document text as data, not instructions, plus testing for
  it.
- A real ingestion pipeline. Here the corpus is a folder of Markdown. A real one
  needs connectors to the actual sources, incremental sync as documents change,
  handling for PDFs and other formats, and deletion so removed documents leave
  the index.
- Retrieval quality work. Hybrid search that combines keyword and vector scoring,
  a reranking step, and query rewriting would raise quality beyond the dense-only
  search used here.
- Conversation and memory. The service answers one question at a time. A real
  assistant holds a multi-turn conversation.
- Serving concerns. Streaming responses, response caching, rate limiting and
  per-user quotas, and per-request cost tracking are all absent.
- Observability. Beyond logs, a production service needs traces, metrics
  dashboards, and alerts.
- A larger evaluation loop. The set here is 19 questions. A production system
  needs a much larger set, human labels to calibrate the judge, evaluation gates
  in CI that block a regression, and captured user feedback.
- Secret handling and resilience. Move the API key to SSM Parameter Store, put
  HTTPS in front of the service, and give the database backups and a recovery
  plan. A single instance is a single point of failure.
