<p align="center">
  <h1 align="center">🧠 L90 — Closed-World Agentic Swarm RAG System</h1>
  <p align="center">
    <em>Zero-hallucination, deterministic, multi-agent scientific reasoning system</em>
  </p>
  <p align="center">
    Built with <strong>LangGraph</strong> · <strong>Google Gemini</strong> · <strong>ChromaDB</strong> · <strong>FastAPI</strong>
  </p>
</p>

---

## 🚀 Overview

**L90** is a production-grade Retrieval-Augmented Generation (RAG) system designed for **scientific accuracy and zero hallucination**. Unlike traditional RAG pipelines that simply retrieve and generate, L90 deploys a **swarm of specialized AI agents** that collaboratively reason, verify, and ground every claim against uploaded source documents.

Every answer is deterministic, auditable, and backed by evidence — or it doesn't get generated.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🐝 **Agentic Swarm** | 9 specialized agents (Manager, Planner, Retriever, Analyzer, Verifier, Corrector, Deep Reasoner, Math Executor, Generator) collaborate via a shared blackboard |
| 🔒 **Grounding Enforcement** | Independent validation layer that verifies every claim against source documents before delivery |
| 🎛️ **5 Operation Modes** | `STRICT` · `PARTIAL` · `GENERAL` · `INCOGNITO` · `WORKSPACE` — each with different grounding thresholds |
| 🧮 **Math & Code Verification** | Sandboxed code execution for verifying mathematical and scientific computations |
| 📋 **Full Audit Trail** | Complete reasoning trace with structured JSON logging for every query |
| 🔄 **Self-Correction Loop** | Up to 3 correction cycles when verification fails, ensuring answer quality |
| 📄 **Multi-Format Ingestion** | Upload and ingest PDF, DOCX, and TXT documents with automatic chunking and embedding |
| 🏗️ **Model Abstraction** | Swap LLM providers without touching core architecture |
| 🖥️ **Web UI** | Beautiful single-page interface for document upload and querying |

---

## 🏗️ Architecture

```
                    ┌─────────────┐
                    │   Manager   │  ← Central orchestrator
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Planner  │ │Retriever │ │ Analyzer │
        └──────────┘ └────┬─────┘ └──────────┘
                          │
                    ┌─────┴─────┐
                    │ ChromaDB  │  ← Vector Store
                    └───────────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │ Verifier │ │Corrector │ │Deep Reasoner │
        └──────────┘ └──────────┘ └──────────────┘
              │            │
              ▼            ▼
        ┌──────────┐ ┌──────────────────┐
        │Generator │ │Grounding Enforcer│  ← Final validation
        └──────────┘ └──────────────────┘
```

**Pipeline Flow:** Query → Manager → Planning → Retrieval → Analysis → Verification → (Correction Loop) → Generation → Grounding Enforcement → Response

### Blackboard Architecture

All agents share a centralized **Blackboard** — a shared reasoning state where each agent reads from and writes to. This enables transparent, collaborative decision-making with a complete audit trail.

---

## 📁 Project Structure

```
L90/
├── l90/
│   ├── agents/           # 9 specialized AI agents
│   │   ├── manager.py        # Central swarm orchestrator
│   │   ├── planner.py        # Query decomposition & planning
│   │   ├── retriever.py      # Vector search & document retrieval
│   │   ├── analyzer.py       # Evidence analysis & extraction
│   │   ├── verifier.py       # Claim verification against sources
│   │   ├── corrector.py      # Self-correction when verification fails
│   │   ├── deep_reasoner.py  # Multi-step scientific reasoning
│   │   ├── math_executor.py  # Sandboxed math/code verification
│   │   └── generator.py      # Final answer generation
│   ├── api/              # FastAPI application & endpoints
│   ├── blackboard/       # Shared reasoning state (memory / Redis)
│   ├── graph/            # LangGraph orchestration & state machine
│   ├── grounding/        # Independent grounding enforcement
│   ├── ingestion/        # Document chunking & embedding pipeline
│   ├── models/           # Pydantic data models
│   ├── modes/            # Operation mode enforcement (5 modes)
│   ├── security/         # Auth, isolation & access control
│   ├── tracing/          # Structured audit logging
│   ├── vectordb/         # ChromaDB vector store abstraction
│   └── config.py         # Central configuration (env-driven)
├── static/
│   └── index.html        # Web UI for querying & uploads
├── tests/                # Pytest test suite
├── pyproject.toml        # Project metadata & dependencies
└── .env.example          # Environment variable template
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.11+**
- **Google Gemini API Key** — [Get one here](https://aistudio.google.com/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/adithyasrivatsa/L90-Beta.git
cd L90-Beta

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

### Run the Server

```bash
python -m uvicorn l90.api.app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser to access the Web UI.

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📡 API Reference

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI (serves `index.html`) |
| `GET` | `/health` | Health check |
| `POST` | `/query` | Run a query through the full swarm pipeline |
| `POST` | `/upload` | Upload and ingest a document (PDF, DOCX, TXT) |

### Authentication

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/login` | Authenticate and receive a session token |
| `POST` | `/logout` | Invalidate session |
| `GET` | `/me` | Get current user info |
| `GET` | `/users` | List all users (admin only) |

### Admin & Sessions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/admin/upload` | Upload to approved library (admin only) |
| `POST` | `/session/start` | Start an incognito session |
| `POST` | `/session/end` | End incognito session (deletes all data) |

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the mechanism of action described in the paper?",
    "mode": "STRICT",
    "user_id": "default_user"
  }'
```

---

## 🎛️ Operation Modes

| Mode | Grounding | Temperature | Use Case |
|------|-----------|-------------|----------|
| **STRICT** | ≥ 85% confidence | 0.0 | Scientific research, compliance — only source-backed answers |
| **PARTIAL** | ≥ 70% confidence | 0.2 | General research — allows light inference |
| **GENERAL** | Standard | Default | Everyday Q&A with retrieval augmentation |
| **INCOGNITO** | Session-scoped | Default | Private sessions — all data deleted on end |
| **WORKSPACE** | Workspace-scoped | Default | Team collaboration with shared document collections |

---

## ⚙️ Configuration

All settings are driven by environment variables (`.env` file). Key options:

```bash
# LLM
GOOGLE_API_KEY=your_key_here
WORKER_MODEL_NAME=gemini-1.5-flash
MANAGER_MODEL_NAME=gemini-1.5-flash
EMBEDDING_MODEL_NAME=gemini-embedding-001

# Grounding thresholds
GROUNDING_CONFIDENCE_THRESHOLD=0.7
STRICT_GROUNDING_THRESHOLD=0.85
PARTIAL_GROUNDING_THRESHOLD=0.70

# Pipeline
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CORRECTION_LOOPS=3
PIPELINE_TIMEOUT_SECONDS=30.0
MAX_PARALLEL_AGENTS=3

# Blackboard persistence
BLACKBOARD_BACKEND=memory  # or "redis"
REDIS_URL=redis://localhost:6379/0
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini 1.5 Flash |
| **Embeddings** | Gemini Embedding 001 |
| **Orchestration** | LangGraph + LangChain |
| **Vector DB** | ChromaDB |
| **API** | FastAPI + Uvicorn |
| **Document Parsing** | PyPDF, python-docx |
| **Frontend** | Vanilla HTML/CSS/JS |

---

## 📄 License

MIT

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/adithyasrivatsa">Adithya Srivatsa</a>
</p>
