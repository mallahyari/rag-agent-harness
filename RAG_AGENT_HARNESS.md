# RAG / Question-Answering Agent Harness — Design Document

The user asks a natural language question about a document corpus. The agent searches for relevant content, drills deeper when needed, reasons across multiple sources, and synthesizes a grounded answer with citations. No pre-loaded context. No fixed pipeline. The harness enforces retrieval quality and citation discipline; the model does the reasoning.

---

## How This Differs from the SQL Harness

The SQL harness has one hard constraint: **correctness** (the DB is ground truth). The RAG harness has a different hard constraint: **groundedness** (the model must not assert things not supported by retrieved text). Hallucination — not SQL errors — is the primary failure mode.

| SQL Analytics Harness | RAG / QA Harness |
|---|---|
| Ground truth = database | Ground truth = retrieved documents |
| Failure mode = wrong query | Failure mode = hallucinated answer |
| Safety = block DML/DDL | Safety = enforce citation, penalize unsupported claims |
| Schema is structured, known | Document structure is unstructured, varied |
| One right retrieval path | Multiple valid retrieval paths |
| Results are exact | Results are probabilistic (similarity) |
| `think` → plan SQL | `think` → plan retrieval strategy |

Everything else — the 5-pillar skeleton, the `think` tool, the weak-model compensation strategy, the ReAct loop — carries over directly.

---

## The Agent Loop (ReAct Pattern)

```
User Question
      │
      ▼
┌─────────────────────┐
│  Orient             │  list_collections() — what corpora exist?
├─────────────────────┤
│  Plan               │  think(strategy) — what to search for, in what order
├─────────────────────┤
│  Retrieve (broad)   │  search_documents(query, k=10) — hybrid search
├─────────────────────┤
│  Expand             │  get_context(chunk_id) — fetch surrounding chunks if excerpt is thin
├─────────────────────┤
│  Re-retrieve        │  search_documents(refined_query) — multi-hop if needed
├─────────────────────┤
│  Synthesize         │  end_turn — answer with inline citations [chunk_id]
└─────────────────────┘
```

The model decides when it has enough. The harness enforces that every factual claim in the final answer references a retrieved chunk ID.

---

## Architecture Overview

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────┐
│  PILLAR 1: STATE CONTROLLER                     │
│  history[] — no context pre-loaded              │
│  System prompt = rules + citation discipline    │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  PILLAR 2: TOOL REGISTRY                        │
│  list_collections │ search_documents            │
│  get_context │ think                            │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  PILLAR 3: RETRIEVAL PIPELINE                   │
│  Hybrid search (semantic + BM25)                │
│  Cross-encoder reranking                        │
│  Context budget enforcement                     │
│  Result formatter with doc IDs                  │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  PILLAR 4: MIDDLEWARE & LIFECYCLE HOOKS         │
│  pre:  context budget check, query expansion    │
│  post: citation presence check, 0-result hint   │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  PILLAR 5: BOUNDARY CONTROLS                    │
│  max_iterations, max_chunks_per_session         │
│  citation enforcement on end_turn               │
└──────────────┬──────────────────────────────────┘
               │
               ▼
      Grounded answer with citations
```

---

## Two Pipelines: Ingestion and Query

Unlike the SQL harness (which only has a query path), RAG has two distinct pipelines that must be designed separately.

```
INGESTION PIPELINE (offline, run once or on document update)
  Documents → Chunker → Embedder → Vector Store + BM25 Index

QUERY PIPELINE (online, per user question)
  Question → Agent Loop → Tools → Retrieval → Reranking → Synthesis
```

The agent harness only touches the **query pipeline**. The ingestion pipeline runs separately and produces the indexes the agent reads from.

---

## Pillar 1: State Controller

No documents are pre-loaded into the system prompt. The agent discovers what's available through tools. The system prompt contains only rules and citation discipline.

```python
SYSTEM_PROMPT = """\
You are a research assistant. You answer questions by searching a document corpus
and reasoning over what you find.

## Workflow — follow this order every time
1. Use list_collections to understand what document sets are available.
2. Use think to plan your retrieval strategy before searching.
3. Use search_documents with a precise query to retrieve relevant passages.
4. Use get_document to read the full text of the most relevant results.
5. If the question is complex, search again with a refined or follow-up query.
6. When you have enough evidence, write a direct answer.

## Citation Rules — non-negotiable
- Every factual claim must include a citation: [doc_id]
- If you cannot find supporting evidence, say so explicitly.
- Never assert facts you did not find in the retrieved documents.
- If retrieved documents conflict, surface the conflict rather than picking one.

## What to avoid
- Do not summarize documents you haven't retrieved.
- Do not pad answers with context that wasn't asked for.
- Do not guess when the evidence is absent — say "not found in the corpus."
"""
```

**Why no pre-loaded context?** For large corpora (thousands of documents), pre-loading is impossible. Even for small ones, pre-loading injects noise — the model reads irrelevant content and hallucinates connections. Dynamic retrieval means the model only sees what's relevant to this specific question.

---

## Pillar 2: Tool Registry

Four tools. Same philosophy as the SQL harness — minimal, each maps to a stage in the ReAct loop.

### `list_collections() → str`
Returns available document collections with a description and document count. Orients the agent before it searches.

```python
{
    "name": "list_collections",
    "description": (
        "List available document collections with a brief description and document count. "
        "Call this first to understand what corpora are available before searching."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}
```

### `search_documents(query, collection=None, k=10) → str`
Hybrid search (semantic + keyword) returning the top-k passages with document IDs, relevance scores, and source metadata. The `k=10` retrieval is broad — reranking happens inside the tool before results are returned, truncating to top-5.

```python
{
    "name": "search_documents",
    "description": (
        "Search the document corpus for passages relevant to a query. "
        "Uses hybrid semantic + keyword search with reranking. "
        "Returns top passages with doc_id, excerpt, and relevance score. "
        "Use precise, specific queries — vague queries return noisy results. "
        "You can search multiple times with different queries to cover a topic fully."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query":      {"type": "string",  "description": "Specific search query"},
            "collection": {"type": "string",  "description": "Collection name (omit to search all)"},
            "k":          {"type": "integer", "description": "Number of results (default 5, max 10)"},
        },
        "required": ["query"],
    },
}
```

### `get_context(chunk_id, window=2) → str`
Returns the matched chunk plus `window` chunks before and after it from the same document. This gives the model the surrounding context without fetching the full document.

**Why not `get_document`?**  
Fetching the full document is almost always wrong:
- A 50-page PDF is ~40,000 tokens — instantly blows the context budget
- Most of it is irrelevant to the question
- It floods the model with noise, which degrades synthesis quality (especially for weaker models)
- You already have the most relevant chunk from search — you need its neighbors, not the whole document

The right primitive is **context window expansion**: fetch the N chunks surrounding the matched chunk. This is the "parent-document retrieval" pattern — embed small for precision, return a bit more for context. The returned text is capped at ~1,000 tokens regardless of `window` size.

```python
{
    "name": "get_context",
    "description": (
        "Fetch the surrounding context for a chunk from search results. "
        "Returns the matched chunk plus a few chunks before and after it "
        "from the same document — enough to understand the passage in context. "
        "Use this when a search excerpt is too brief to reason from. "
        "Do NOT use this to read entire documents — use search instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_id": {"type": "string", "description": "Chunk ID from search results"},
            "window":   {"type": "integer", "description": "Number of chunks before/after to include (default 2, max 4)"},
        },
        "required": ["chunk_id"],
    },
}

# Implementation
def get_context(chunk_id: str, window: int = 2, max_tokens: int = 1000) -> tuple[str, bool]:
    chunk = vector_store.get_by_id(chunk_id)
    if not chunk:
        return f"Error: chunk '{chunk_id}' not found", True

    # Fetch surrounding chunks from the same document, ordered by position
    neighbors = vector_store.get_neighbors(
        doc_id=chunk.doc_id,
        chunk_index=chunk.chunk_index,
        window=min(window, 4),
    )

    text = "\n\n".join(c.text for c in neighbors)

    # Hard token cap — never return more than budget allows
    if count_tokens(text) > max_tokens:
        text = truncate_to_tokens(text, max_tokens)
        text += "\n\n[Truncated. Use search with a more specific query to find other sections.]"

    return f"[Context for {chunk_id}]\n{text}", False
```

### `think(plan) → str`
Identical to the SQL harness. Forces the model to articulate its retrieval strategy before searching. Especially important here because retrieval quality depends on query formulation — a model that plans its queries first writes better ones.

```python
# Implementation: echo back
def think(plan: str) -> str:
    return f"Plan recorded:\n{plan}"
```

---

## Pillar 3: Retrieval Pipeline

This is the core of the RAG harness. The quality of retrieval determines the quality of the answer — more than the model choice does.

### Stage 1: Chunking (Ingestion)

Chunk quality is the highest-leverage ingestion decision. Rules of thumb:

| Document type | Strategy | Chunk size |
|---|---|---|
| Prose / articles | Recursive paragraph split | 300–500 tokens |
| Technical docs / manuals | Section-aware split (headers as boundaries) | 500–800 tokens |
| Q&A / FAQ | One Q+A pair per chunk | Variable |
| Tables / structured content | Keep table intact as one chunk | Variable |
| Short documents (< 1 page) | No chunking — whole document is one chunk | — |

**Overlap:** Add 10–15% overlap between chunks (e.g. 50 tokens for a 400-token chunk) to prevent answers from being split across boundaries.

**Metadata per chunk:** Store `doc_id`, `source_filename`, `page_number`, `section_title`, `chunk_index`. The agent cites `doc_id`; the UI uses the rest for display.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],  # semantic boundaries first
)
```

### Stage 2: Embedding (Ingestion)

Use a small, fast embedding model for ingestion. The quality difference between models is smaller than the quality difference between chunking strategies.

| Recommendation | Model | Notes |
|---|---|---|
| Default | `text-embedding-3-small` (OpenAI) | Fast, cheap, good quality |
| Local / offline | `BAAI/bge-small-en-v1.5` | Runs on CPU, no API call |
| High accuracy | `text-embedding-3-large` | 3× cost of small, ~10% better |

Embed both chunks and queries with the **same model**.

### Stage 3: Hybrid Search (Query)

Semantic search alone misses exact-match queries ("what is the definition of X?", product codes, names, dates). BM25 alone misses conceptual queries. Hybrid search combines both via **Reciprocal Rank Fusion (RRF)**:

```python
def hybrid_search(query: str, k: int = 10) -> list[SearchResult]:
    # Run both searches in parallel
    semantic_results = vector_store.similarity_search(query, k=k)
    keyword_results  = bm25_index.search(query, k=k)

    # Reciprocal Rank Fusion
    scores = {}
    for rank, result in enumerate(semantic_results):
        scores[result.id] = scores.get(result.id, 0) + 1 / (60 + rank)
    for rank, result in enumerate(keyword_results):
        scores[result.id] = scores.get(result.id, 0) + 1 / (60 + rank)

    # Sort by fused score, return top-k
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [lookup(id) for id in ranked_ids]
```

### Stage 4: Reranking (Query)

Retrieve `k=10` candidates from hybrid search, then rerank with a cross-encoder to top-5. This two-stage approach keeps search fast (bi-encoder at scale) and accurate (cross-encoder on a small candidate set).

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [c for _, c in ranked[:top_k]]
```

**Bi-encoder alone: 65–80% relevance accuracy. With cross-encoder reranking: 85–90%.** This is the single highest-leverage quality improvement after chunking.

### Stage 5: Result Formatter

```python
def format_search_results(results: list[SearchResult]) -> str:
    lines = []
    for r in results:
        lines.append(
            f"[{r.doc_id}] (score: {r.score:.2f}) {r.source}\n"
            f"{r.excerpt}\n"
        )
    return "\n---\n".join(lines) if lines else "(no results found)"
```

Every result is prefixed with `[chunk_id]` — this is the token the model uses for citations in its final answer. The chunk_id encodes both document and position (e.g. `report-2024_chunk_14`), so citations are traceable back to the exact passage.

### Vector Store Recommendation: LanceDB

| Option | Type | Best for |
|---|---|---|
| **LanceDB** | Embedded, disk-based | Default choice — no server, large-than-RAM datasets, fast |
| ChromaDB | Embedded | Simpler API, good for small corpora |
| Qdrant | Server | Production deployments, advanced filtering |
| FAISS | In-process | Maximum speed, no metadata filtering |

**Use LanceDB.** It's embedded (no server to manage), disk-based (handles larger-than-RAM corpora), and supports hybrid search natively. Same developer experience as SQLite — a single file.

```python
import lancedb

db = lancedb.connect("./corpus.lance")
table = db.open_table("documents")

def vector_search(query_embedding, k=10):
    return table.search(query_embedding).limit(k).to_list()
```

---

## Pillar 4: Middleware & Lifecycle Hooks

### Pre-Hook: Context Budget Check

The harness tracks total tokens of retrieved text accumulated across all tool calls in a session. If the budget is approaching the context window limit, the pre-hook blocks further `get_document` calls and injects a hint to synthesize from what's already been retrieved.

```python
MAX_RETRIEVED_TOKENS = 8_000  # conservative for weaker models

def pre_hook(tool_name: str, args: dict, session_state: dict) -> dict:
    if tool_name in ("search_documents", "get_document"):
        if session_state["retrieved_tokens"] > MAX_RETRIEVED_TOKENS:
            raise ContextBudgetExceeded(
                "Retrieved context budget reached. Synthesize from what you have."
            )
    return args
```

### Post-Hook: 0-Result Hint

Mirrors the SQL harness 0-row hint. If `search_documents` returns nothing, inject a hint rather than letting the model give up.

```python
def post_hook(tool_name: str, result: str, is_error: bool) -> str:
    if tool_name == "search_documents" and "(no results found)" in result:
        return (
            result + "\n\n[Hint: No results. Try a shorter or more general query, "
            "or use list_collections to verify the right collection is being searched.]"
        )
    return result
```

### Post-Hook: Citation Presence Check (on `end_turn`)

When the model produces its final answer, a post-hook validates that it contains at least one citation bracket. If not, the harness injects the answer back as a user message with a correction prompt.

```python
def validate_final_answer(answer: str, retrieved_chunk_ids: set[str]) -> str | None:
    """Returns a correction prompt if citations are missing, else None."""
    import re
    cited = set(re.findall(r'\[([^\]]+)\]', answer))
    if not cited:
        return (
            "Your answer contains no citations. "
            "Please revise it to include [chunk_id] references for each factual claim, "
            "using the chunk IDs from the search results."
        )
    unknown = cited - retrieved_chunk_ids
    if unknown:
        return (
            f"Your answer cites {unknown} which were not in the retrieved results. "
            "Only cite chunks you actually retrieved."
        )
    return None
```

This is the primary hallucination mitigation mechanism. It's not perfect, but it forces the model to stay tethered to retrieved content.

---

## Pillar 5: Boundary Controls

| Control | Recommended Value | Rationale |
|---|---|---|
| `max_iterations` | 10 | Complex multi-hop questions need more steps than SQL |
| `max_retrieved_tokens` | 8,000 | Keeps context clean for weaker models; raise for stronger ones |
| `max_k_per_search` | 10 | Broad retrieval before reranking narrows to 5 |
| `rerank_top_k` | 5 | More than 5 retrieved passages degrades weak-model synthesis |
| `max_tokens` | 4,096 | Sufficient for synthesis |

**Hard stop message:** "Reached iteration limit. Here is a partial answer based on what was retrieved. The question may need to be broken into smaller parts."

---

## Weak Model Compensation

Same principle as the SQL harness: the harness enforces what the model can't self-impose.

| Risk with weak model | Harness mitigation |
|---|---|
| Vague queries → poor retrieval | `think` tool forces query planning before searching |
| Hallucinated citations | Post-hook validates cited IDs exist in retrieved set |
| Too many chunks → confused synthesis | Context budget cap; rerank to top-5 not top-10 |
| Gives up on 0 results | 0-result hint in post-hook |
| Asserts without evidence | System prompt citation rules + citation validator |
| Reads only first result | Tool description explicitly says "search multiple times" |

**Few-shot example in system prompt** is even more valuable here than in the SQL harness. Show one worked example of the full tool sequence: `think` → `search_documents` → `get_document` → answer with `[doc_id]`. Small models follow patterns reliably.

---

## Multi-Hop Retrieval

Complex questions naturally require multiple searches:

> "How did the company's refund policy change between 2023 and 2024, and what drove the change?"

The agent searches for refund policy → finds 2024 doc → reads it → searches for 2023 policy → finds change memo → reads it → synthesizes both. This happens naturally in the ReAct loop without any special handling, as long as:

1. `search_documents` can be called multiple times (it can — no restriction)
2. `get_document` returns full text (not just excerpts)
3. History accumulates both retrieved sets so the model can reason across them

The only risk is context budget — two full document reads can consume 4,000+ tokens. The context budget hook prevents runaway accumulation.

---

## File Structure

```
rag-agent-harness/
├── harness.py          # AgentHarness — loop, hooks, history (Pillars 1, 4, 5)
├── tools.py            # list_collections, search_documents, get_context, think
├── retrieval.py        # hybrid_search, rerank, format_results
├── ingestion.py        # chunk, embed, index — runs offline, not at query time
├── vector_store.py     # LanceDB wrapper — insert, search, get_by_id
├── bm25_index.py       # BM25 index wrapper (rank_bm25 or tantivy)
├── config.py           # k, rerank_top_k, budget, model name, embedding model
└── main.py             # CLI entrypoint — load corpus, run question, print answer
```

---

## Build Order

```
Phase 1 — Working baseline
  ├── Ingest 10 documents into LanceDB (fixed-size chunking, any embedding model)
  ├── Semantic search only (no BM25 yet)
  ├── search_documents + get_context + think tools
  └── Test: 5 representative questions, check citation quality

Phase 2 — Retrieval quality
  ├── Add BM25 index + RRF hybrid search
  ├── Add cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
  ├── 0-result hint in post-hook
  └── Citation validator post-hook

Phase 3 — Robustness
  ├── Context budget tracking + enforcement
  ├── list_collections tool
  ├── Semantic/recursive chunking (replace fixed-size)
  └── Structured logging

Phase 4 — Optional
  ├── Query expansion (generate multiple query variants, merge results)
  ├── Parent-document retrieval (embed chunks, return parent section)
  ├── Factuality scorer post-generation
  └── Evaluation harness (RAGAS or similar)
```

---

## Dependency Summary

```
lancedb              # vector store
sentence-transformers # embeddings + cross-encoder reranker
rank-bm25            # BM25 keyword index
langchain-text-splitters  # recursive chunking
anthropic            # Claude API
```

Minimal footprint. No LangChain orchestration, no LlamaIndex — the harness is the orchestrator.

### datasets examples
For your two harnesses:

SQL Analytics (DuckDB)

Start with TPC-H — it's built into DuckDB, zero data download:


INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=1);  -- generates ~1GB, 8 tables
8 tables (orders, customers, suppliers, lineitems, parts, etc.), temporal data, complex multi-table relationships. Purpose-built for exactly the kind of analytical queries your agent will handle.

When you want something more readable, add the Online Retail dataset from Kaggle — real e-commerce transactions, easy to reason about business-domain questions.

RAG / QA (Unstructured Documents)

Start with HotpotQA — purpose-built for multi-hop reasoning, which is your hardest case:


from datasets import load_dataset
ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
113k QA pairs over Wikipedia articles, ground truth supporting facts included (so you can measure whether your retrieval actually found the right chunks). It's the most direct test of whether your search_documents + get_context loop works.

Recommended start order:

Phase	SQL	RAG
Development	TPC-H sf=0.1 (small, fast)	HotpotQA (500 question sample)
Integration	TPC-H sf=1	HotpotQA full
Scale test	TPC-H sf=10	Wikipedia subset
TPC-H and HotpotQA together give you everything you need to validate both harnesses before touching real data.
