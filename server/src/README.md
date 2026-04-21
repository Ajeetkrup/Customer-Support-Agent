# Customer Support Agent — `server/src`

Full **core architecture**, repository layout, environment variables, setup, and run commands are documented in the **[repository root `README.md`](../../README.md)**.

## Quick map (this package)

| Module / path | Purpose |
|----------------|---------|
| `main.py` | FastAPI app, mounts `/api` router |
| `router/api.py` | REST: seed DB, KB ingest/retrieve, trigger batch agent |
| `agent/agent.py` | LangGraph `SUPPORT_AGENT`: gather_context → triage → resolve \| escalate \| clarify → finish |
| `agent/tools.py` | Tool implementations (JSON fixtures + RAG bridge) |
| `agent/prompts.py`, `agent/state.py` | Prompts and typed agent state |
| `run.py` | Async batch runner over `tickets.json` |
| `rag/` | Qdrant (sync/async clients, Redis KV), ingest (`knowledge-base.md` → Qdrant + persisted `./storage`), hybrid retrieval (dense + BM25 fusion, reranker, HyDE) |
| `audit.py` | Audit JSONL writer |
| `*.json` | Fixture data for tools and batch runs |
