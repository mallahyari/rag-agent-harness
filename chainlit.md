# RAG Agent Harness

Ask questions about your ingested document corpus. The agent searches, retrieves, and cites sources automatically.

**How to use:**
- Type any question and press Enter
- The agent will search your documents and answer with citations
- Each `[chunk_id]` in the answer links back to a specific passage in the corpus

**Available tools the agent uses behind the scenes:**
- 📋 `list_collections` — discover indexed document collections
- 🔍 `search_documents` — hybrid BM25 + semantic search with reranking
- 📄 `get_context` — fetch a chunk and its surrounding passages
- 💭 `think` — structured reasoning for complex multi-hop questions
