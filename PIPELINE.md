# Pipeline Reference

How data flows through the RAG Agent Harness, in two phases:

1. **Ingestion** — offline: documents → chunks → embeddings → indexes
2. **Answering** — online: question → agentic retrieval loop → cited answer

All embedding, vector search, BM25, and reranking run **locally**. The only
external API call is the LLM itself (Gemini by default, via LiteLLM).

---

## Phase 1 — Ingestion (embedding & storing)

**Entry point**

```bash
python -m ingestion.ingest --path <file|dir> --collection <name>
```

This calls `ingest()` in [`ingestion/pipeline.py`](ingestion/pipeline.py).

### Steps

1. **Read & parse** — [`ingestion/chunker.py`](ingestion/chunker.py)
   - `.pdf` files are extracted with `liteparse` (local, no API key).
   - All other files are read as UTF-8 text.
   - A `doc_id` is derived from the filename, sanitized to `[a-zA-Z0-9_-]` and
     capped at 80 characters.

2. **Chunk** — `RecursiveCharacterTextSplitter` (LangChain)
   - Default **512-character chunks** with **64-character overlap**
     (override with `--chunk-size` / `--chunk-overlap`).
   - Each chunk gets a sequential ID: `<doc_id>__c0000`, `__c0001`, …
     This ordering is what lets `get_context` fetch neighboring chunks later.

3. **Embed in batches** — [`retrieval/embeddings.py`](retrieval/embeddings.py)
   - Chunks are processed in **batches of 64**.
   - Each batch is embedded with **`BAAI/bge-small-en-v1.5`** (SentenceTransformer),
     `normalize_embeddings=True` so vectors are unit-length (cosine = dot product).
   - Output dimension: **384**.
   - Each chunk becomes a `Chunk(chunk_id, doc_id, collection, text, vector)`.

4. **Store in LanceDB** — [`retrieval/vector_store.py`](retrieval/vector_store.py)
   - `insert_chunks()` groups by collection and creates the table if needed
     (vector dimension inferred from the first vector).
   - Schema: `chunk_id, doc_id, collection, text, vector`.
   - Rows are appended via `tbl.add(rows)`.

5. **Rebuild & persist the BM25 index** — [`retrieval/bm25.py`](retrieval/bm25.py)
   - After insert, **all** rows for the collection are read back and the BM25
     index is rebuilt from scratch (for consistent IDF), then saved to disk as
     `<bm25_base_path>_<collection>.bm25`.

### Result

Each collection ends up with **two indexes**, both required for hybrid search:

| Index | File | Purpose |
|---|---|---|
| Vector (semantic) | `corpus.lance/<collection>.lance/` | Cosine similarity search |
| BM25 (keyword) | `corpus_<collection>.bm25` | Keyword/lexical search |

### Notes & caveats

- **BM25 is fully rebuilt on every ingest** — re-ingesting into a large
  collection re-reads and re-indexes the entire collection (O(total chunks),
  not O(new chunks)). Fine at current scale; revisit if collections grow large.
- **No dedup / upsert** — re-ingesting the same file appends duplicate chunks
  rather than replacing them, since `tbl.add` is a plain append.

### Ingestion flow

```
file/dir
   │  read & parse (liteparse for PDF, UTF-8 otherwise)
   ▼
RecursiveCharacterTextSplitter  →  chunks (512 chars, 64 overlap, sequential IDs)
   │  embed in batches of 64  (BAAI/bge-small-en-v1.5, normalized, 384-dim)
   ▼
LanceDB table  (chunk_id, doc_id, collection, text, vector)
   │
   └─ rebuild + persist BM25 index over the whole collection
```

---

## Phase 2 — Answering questions

A question enters `agent.run()` in [`harness/agent.py`](harness/agent.py) and the
agent runs a **streaming ReAct loop** — up to `MAX_ITERATIONS = 10` round-trips.

### The loop (each iteration)

1. **Ask the LLM what to do next**
   - History + system prompt + the four tool schemas are streamed to the LLM
     (Gemini by default, via LiteLLM).
   - The model returns either **text** (a candidate answer) or **tool calls**.

2. **Execute tool calls** — run **sequentially** (never in parallel)

   | Tool | What it does |
   |---|---|
   | `list_collections` | Lists available collections |
   | `search_documents` | Hybrid retrieval core (see below) |
   | `get_context` | Fetches a chunk + its neighbors for deeper reading |
   | `think` | Records reasoning; returns the thought back to the model |

3. **Feed results back** into history, then repeat. The model sees each result
   and decides the next action — search again with a reworded query, expand a
   chunk, or write the answer.

### Inside `search_documents` — the retrieval core

[`retrieval/search.py`](retrieval/search.py). One search runs four sub-steps:

1. **Embed the query** with `BAAI/bge-small-en-v1.5` (normalized, local).
2. **Two retrievals in parallel:**
   - **Semantic** — LanceDB cosine search over vectors.
   - **Keyword** — BM25 search over the `.bm25` index.
3. **Fuse with RRF** (Reciprocal Rank Fusion): combine the two ranked lists by
   rank position, `score = Σ 1 / (60 + rank)`.
4. **Rerank** the fused candidates with a cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`, local) and keep the top results.

Flow: `search_k=10` from each retriever → fused → reranked → `rerank_top_k=5`
returned, formatted as `[chunk_id] (doc: …) text` for the LLM.

### Guardrails around every tool call

[`harness/hooks.py`](harness/hooks.py):

- **pre_hook** — enforces the context budget. If retrieved tokens exceed
  `MAX_RETRIEVED_TOKENS = 8000`, further retrieval is blocked and the model is
  told to synthesize from what it has.
- **post_hook** — records every `chunk_id` actually retrieved into
  `state.retrieved_chunk_ids` (used by citation validation).

### Finishing — answer + citation validation

When the LLM returns text instead of a tool call ([`harness/agent.py`](harness/agent.py)):

1. **Validate citations** (`validate_citation`): every `[chunk_id]` in the
   answer must match a chunk that was actually retrieved.
   - **Hallucinated ID** → answer rejected; a correction is appended and the
     model must revise (up to `MAX_CITATION_RETRIES = 3`).
   - **"Not found" answers** with no citations are allowed.
2. If valid → the final answer is streamed to the UI and returned.

### Answering flow

```
question
   │
   ▼
┌─────────────────────────────────────────────┐
│  LLM decides next action                     │
│     │                                        │
│     ├─ tool call → execute → feed result ────┤  repeat ≤ 10×
│     │      search_documents =                │
│     │        embed → (cosine ‖ BM25)         │
│     │              → RRF → rerank            │
│     │                                        │
│     └─ text answer                           │
└───────────────────│─────────────────────────┘
                    ▼
        validate citations  ──reject──▶ revise (≤ 3×)
                    │ ok
                    ▼
            stream final answer
```

### How this differs from naive RAG

- Retrieval is **not** a fixed first step — the **LLM decides** when and whether
  to search.
- The agent can search **multiple times** with rewritten queries, and expand
  context only where a chunk is too thin.
- The answer is **rejected if its citations aren't grounded** in retrieved
  chunks, making every claim traceable.

---

## Key settings ([`config.py`](config.py))

| Setting | Default | Description |
|---|---|---|
| `provider` | `gemini-fast` | LLM provider key |
| `search_k` | `10` | Candidates per retriever before fusion |
| `rerank_top_k` | `5` | Results kept after reranking |
| `context_window_size` | `2` | Neighbor chunks fetched by `get_context` |
| `max_retrieved_tokens` | `8000` | Hard context budget |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Local embedding model (384-dim) |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder |
| `lance_db_path` | `./corpus.lance` | Vector store path |
| `bm25_index_path` | `./corpus` | BM25 index base path |

Loop limits live in [`harness/agent.py`](harness/agent.py)
(`MAX_ITERATIONS = 10`, `MAX_CITATION_RETRIES = 3`) and
[`harness/hooks.py`](harness/hooks.py) (`MAX_RETRIEVED_TOKENS = 8000`).
