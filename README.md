# L90 — Closed-World Agentic Swarm RAG System

Zero-hallucination, deterministic, multi-agent scientific reasoning system built with LangGraph, Gemini 2.5 Flash, and ChromaDB.

## Architecture

**Pipeline:** Manager → Retrieval → Analysis → Verification → (Correction loop) → Generator → Grounding Enforcer

**Key Features:**
- **Blackboard Architecture** — centralized shared reasoning state
- **5 Operation Modes** — STRICT, PARTIAL, GENERAL, INCOGNITO, WORKSPACE
- **Grounding Enforcement** — independent, mandatory layer that validates every claim
- **Audit Trail** — complete reasoning trace with structured JSON logging
- **Model Abstraction** — swap LLM providers without touching core architecture

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Set GOOGLE_API_KEY in .env

# 3. Run tests
python -m pytest tests/ -v

# 4. Start API
python -m uvicorn l90.api.app:app --reload
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Run a query through the pipeline |
| `POST` | `/upload` | Upload and ingest a document |
| `POST` | `/session/start` | Start incognito session |
| `POST` | `/session/end` | End incognito session (deletes all data) |

## License

MIT
