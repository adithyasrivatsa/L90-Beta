# L90 — Project Documentation

## How It All Started

L90 began with a simple question: *What if an AI could never hallucinate?*

Large Language Models are powerful, but they have a fundamental flaw — they make things up. In scientific, medical, and engineering contexts, a hallucinated answer isn't just wrong, it's dangerous. We wanted a system where every single claim in every answer is backed by evidence from uploaded source documents, and if it can't be backed up, the system says *"I don't have enough information"* instead of fabricating an answer.

That's the core philosophy of L90: **zero hallucination, or no answer at all.**

---

## Inspiration: The Kimi K2 Swarm

The architecture of L90 was heavily inspired by the **Kimi K2.5 Agent Swarm**, developed by Moonshot AI.

### What Kimi K2.5 Does

Kimi K2.5 introduced a paradigm shift in how AI systems handle complex tasks. Instead of a single model processing everything sequentially, it deploys a **swarm of specialized sub-agents** orchestrated by a central manager:

- A **trainable orchestrator** breaks complex tasks into parallelizable sub-tasks
- It **dynamically spawns up to 100 frozen worker agents**, each handling a specific sub-task
- All sub-agents **run in parallel**, using tools like web search, code execution, and file generation independently
- The system was trained using **Parallel-Agent Reinforcement Learning (PARL)**, which taught the orchestrator *when* and *how* to decompose work into parallel streams
- This achieves a **4.5x speedup** over single-agent setups

The key insight we took from Kimi K2 was this: **one model trying to do everything is inherently limited. A team of specialists, each doing one thing well, produces better results.**

### How L90 Adapts This Philosophy

L90 applies the multi-agent swarm philosophy to a **closed-world, deterministic RAG context**. Here's what we kept, what we changed, and why:

| Kimi K2.5 Swarm | L90 Adaptation | Why We Changed It |
|---|---|---|
| Dynamic agent creation (up to 100) | 9 fixed, specialized agents | Determinism — fixed roles make every pipeline run auditable and reproducible |
| Open-ended tool use (web search, browsing) | Closed-world retrieval from uploaded documents only | Zero hallucination — no external sources means no unverifiable information |
| PARL-trained orchestrator | Rule-based + LLM-assisted planning | We don't need RL to learn task decomposition — our task space (document QA) is well-defined |
| Massive parallelism | Selective parallelism where data dependencies allow | Most of our agents depend on the previous agent's output (you can't verify what hasn't been analyzed), so sequential execution is correct, not a limitation |
| General-purpose (coding, research, writing) | Scientific reasoning and document QA only | Narrow scope = deeper quality guarantees |

The result is a system that shares Kimi K2's **philosophy** (central manager, specialized agents, complexity-aware routing) while being purpose-built for scientific accuracy rather than general-purpose speed.

---

## System Architecture

### The Big Picture

When a user sends a query to L90, this is what happens:

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    MANAGER AGENT                      │
│  • Validates query and operation mode                 │
│  • Delegates to PlannerLayer for strategy generation  │
│  • Enforces access control via ModeEnforcer           │
│  • Classifies complexity: BASIC / INTERMEDIATE /      │
│    ADVANCED / RESEARCH_GRADE                          │
│  • Writes execution plan to Blackboard                │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              SWARM ORCHESTRATOR                       │
│  Routes to the correct pipeline based on complexity:  │
│                                                       │
│  BASIC ──────► Retriever → Generator → Grounding      │
│                                                       │
│  INTERMEDIATE ► Retriever → Analyzer ‖ Verifier       │
│                            → Generator → Grounding    │
│                                                       │
│  ADVANCED ───► Retriever → Analyzer → Verifier        │
│                → [Correction Loop] → Generator        │
│                → Grounding                            │
│                                                       │
│  RESEARCH ───► Retriever → Analyzer → MathExecutor    │
│  GRADE         → Verifier → [Correction Loop]         │
│                → DeepReasoning → Generator            │
│                → Grounding                            │
└──────────────────────────────────────────────────────┘
                       │
                       ▼
              Final Verified Answer
```

Every single pipeline ends with the **Grounding Enforcer** — an independent validation layer that cannot be bypassed. If any claim in the generated answer isn't grounded in the source documents, the answer is **rejected** and replaced with "Insufficient verified information."

### The Blackboard: Shared Reasoning Memory

All agents communicate through a centralized **Blackboard** — a shared data structure that serves as the system's reasoning memory. The Blackboard tracks 22 fields:

- **Identity**: Session ID, timestamps
- **Query Context**: The user's query, operation mode, user/workspace IDs
- **Access Control**: Which document collections this query is allowed to search
- **Retrieval Results**: Chunks pulled from ChromaDB with distances and metadata
- **Analysis**: Structured analysis of the retrieved evidence
- **Verification**: Whether claims pass verification, confidence scores
- **Correction**: Results from correction loops when verification fails
- **Output**: The final answer and confidence score
- **Audit Trail**: A complete append-only reasoning trace — every decision by every agent is logged with timestamps
- **Grounding Report**: Claim-by-claim grounding validation results
- **Code Verification**: Results from sandboxed math/code execution
- **Deep Reasoning**: Multi-step reasoning chains for complex queries
- **LaTeX Equations**: Extracted mathematical expressions

This design is inspired by the **Blackboard Architecture pattern** from AI research, where multiple knowledge sources (agents) collaborate by reading from and writing to a shared workspace. The key benefit is **complete transparency** — you can replay exactly how the system arrived at any answer by reading the Blackboard's reasoning trace.

---

## The 9 Agents

Each agent in L90 has a single, well-defined responsibility. They inherit from a common `BaseAgent` class and implement an `execute(blackboard)` method.

### 1. Manager Agent
**Role**: Central orchestrator. Never directly answers the user.

The Manager is the brain of L90. When a query arrives, it:
1. Enforces mode-based access control (which document collections can be searched)
2. Delegates to the PlannerLayer for strategy generation
3. Writes the execution plan to the Blackboard

The Manager **never generates answers itself** — it only decides *how* the query should be processed.

### 2. Planner Layer
**Role**: Analyzes queries and produces structured execution plans.

The Planner is the Manager's internal reasoning engine. It uses a two-tier classification approach:
- **Fast path**: Heuristic-based keyword matching for obvious query types (simple factual lookups, greetings, etc.). This avoids an LLM call entirely.
- **LLM fallback**: For non-trivial queries, it prompts Gemini to produce a JSON execution plan with fields like `complexity_level`, `retrieval_strategy`, `requires_math_verification`, and `question_domain`.

The Planner detects math/physics keywords (e.g., "Schrödinger", "differential equation", "Fourier transform") and scientific domains to properly classify query complexity.

### 3. Retriever Agent
**Role**: Searches ChromaDB for relevant document chunks.

The Retriever takes the query and searches across all allowed collections (determined by the Manager). It uses cosine similarity search and returns the top-k most relevant chunks with their metadata and distance scores. It can query multiple collections in one call, merging and sorting results by relevance.

### 4. Analyzer Agent
**Role**: Deep analysis of retrieved evidence.

The Analyzer examines the retrieved chunks and extracts structured insights — key claims, supporting evidence, contradictions, and gaps. It produces a structured analysis that downstream agents use.

### 5. Verifier Agent
**Role**: Validates that analysis results are supported by source documents.

The Verifier independently checks every claim from the analysis against the retrieved chunks. It assigns a confidence score and sets `verification_passed` on the Blackboard. If verification fails, the Corrector is triggered.

### 6. Corrector Agent
**Role**: Self-correction when verification fails.

When the Verifier rejects an analysis, the Corrector examines what went wrong and generates corrective queries. The system then re-retrieves, re-analyzes, and re-verifies — up to 3 correction loops (configurable via `MAX_CORRECTION_LOOPS`). This self-healing behavior ensures the system doesn't give up after one bad retrieval.

### 7. Math Executor Agent
**Role**: Sandboxed code execution for mathematical verification.

For queries involving calculations, equations, or physics, the Math Executor runs code in a sandboxed environment with a whitelist of allowed modules (`math`, `numpy`, `scipy`, `sympy`, etc.). It verifies mathematical claims computationally — if a paper says "the integral evaluates to 42", this agent runs the actual computation to confirm.

### 8. Deep Reasoning Agent
**Role**: Multi-step scientific reasoning for complex queries.

For RESEARCH_GRADE queries that require connecting concepts across multiple domains or synthesizing insights from many sources, the Deep Reasoner performs chain-of-thought reasoning. It produces structured reasoning chains that the Generator incorporates into the final answer.

### 9. Generator Agent
**Role**: Produces the final human-readable answer.

The Generator takes everything on the Blackboard — retrieved chunks, analysis results, verification results, code verification, deep reasoning — and synthesizes a comprehensive, well-structured answer. It includes LaTeX equations when needed and cites sources.

### The Grounding Enforcer (Not an Agent — An Enforcement Layer)

The Grounding Enforcer is deliberately **not** classified as an agent. It's an independent, mandatory validation layer that sits between the Generator and the user. It performs three checks:

1. **Citation Check**: Every claim must map to a retrieved source chunk
2. **Verification Check**: Only verified information can appear in the answer
3. **Confidence Threshold**: The overall confidence must meet the mode-specific threshold (85% for STRICT, 70% for PARTIAL)

If any check fails, the answer is **replaced** with "Insufficient verified information." This is non-negotiable — the Grounding Enforcer **cannot be bypassed or overridden**.

---

## Frameworks & Technologies

### Why LangGraph?

We evaluated several orchestration frameworks before choosing LangGraph:

| Framework | Why Not | 
|---|---|
| **LangChain (chains only)** | Too linear — we needed conditional branching and parallel execution |
| **AutoGen** | Designed for conversational multi-agent debates, not deterministic pipelines |
| **CrewAI** | Higher-level abstraction than we needed, less control over execution flow |
| **Raw asyncio** | Too low-level — we'd have to build state management, tracing, and routing from scratch |

**LangGraph** gave us the perfect middle ground:
- **State machine semantics** — each agent is a node, transitions are edges with conditions
- **Built-in state persistence** — the graph state tracks everything between nodes
- **Conditional routing** — the Manager's complexity classification directly maps to graph edges
- **Async-native** — every node can be an async function, enabling parallel execution where needed
- **LangChain ecosystem** — access to text splitters, document loaders, and model abstractions

In practice, our `SwarmOrchestrator` uses LangGraph's concepts but implements the routing manually via `asyncio.gather` for finer control over parallel execution and timing.

### Why Google Gemini?

We chose Gemini 1.5 Flash as our LLM for several reasons:

- **Speed**: Flash is one of the fastest production LLMs, critical for our 5-second response target
- **Cost**: Significantly cheaper than GPT-4 or Claude for the volume of agent calls we make (up to 8+ LLM calls per query in RESEARCH_GRADE)
- **JSON mode**: Native structured output generation, essential for our Planner and Grounding Enforcer which need JSON responses
- **Long context window**: 1M tokens allows us to stuff many retrieved chunks into a single prompt
- **Model abstraction**: Our `ModelProvider` class wraps the Gemini SDK, so swapping to another provider (OpenAI, Anthropic, etc.) requires changing one file

We use **Gemini Embedding 001** for document embeddings, which produces 768-dimensional vectors stored in ChromaDB.

### Why ChromaDB?

ChromaDB was chosen as our vector database because:

- **Embedded mode**: Runs in-process, no separate server needed. This simplifies deployment massively — the entire system is a single Python process
- **Persistent storage**: Data survives restarts via local disk storage
- **Collection isolation**: Native support for multiple collections, which maps perfectly to our 5-collection architecture (user private, workspace, internal library, incognito, approved library)
- **Metadata filtering**: Rich query-time filtering by owner, workspace, security level, domain, etc.
- **Cosine similarity**: Built-in support for cosine distance, which works well with Gemini embeddings

### Why FastAPI?

FastAPI is the API framework because:

- **Async-native**: Our entire pipeline is async (agents, ChromaDB operations, LLM calls). FastAPI handles async routes natively
- **Automatic OpenAPI docs**: Every endpoint is auto-documented at `/docs`
- **Pydantic models**: Request/response validation is declarative via Pydantic
- **File uploads**: Built-in `UploadFile` support for document ingestion
- **Middleware**: Easy to add audit logging, CORS, and authentication middleware
- **Static file serving**: Serves our Web UI directly from the same server

---

## The 5 Operation Modes

L90 supports 5 distinct operation modes, each with different grounding thresholds and collection access:

### STRICT Mode
- **Grounding threshold**: 85%
- **Temperature**: 0.0 (fully deterministic)
- **Collections**: User private + Approved library
- **Use case**: Scientific research, compliance, regulatory documents
- **Philosophy**: If you can't cite it, don't say it

### PARTIAL Mode
- **Grounding threshold**: 70%
- **Temperature**: 0.2 (allows light inference)
- **Collections**: User private + Workspace + Approved library
- **Use case**: General research where some inference is acceptable
- **Philosophy**: Extend slightly beyond the source material when confident

### GENERAL Mode
- **Grounding threshold**: Standard
- **Temperature**: Default
- **Collections**: All non-incognito collections
- **Use case**: Everyday Q&A with retrieval augmentation

### INCOGNITO Mode
- **Grounding threshold**: Standard
- **Temperature**: Default
- **Collections**: Incognito session collection only
- **Use case**: Privacy-sensitive queries
- **Philosophy**: Nothing persists. When the session ends, all documents and data are permanently deleted

### WORKSPACE Mode
- **Grounding threshold**: Standard
- **Temperature**: Default
- **Collections**: Workspace collection (shared with team)
- **Use case**: Team collaboration on shared document sets
- **Philosophy**: Everyone on the team sees the same knowledge base

---

## Document Ingestion Pipeline

When a user uploads a document, it goes through a 4-stage pipeline:

```
Upload (PDF/DOCX/TXT/MD)
        │
        ▼
   ┌─────────┐
   │  LOAD   │  Extract raw text (PyPDF for PDF, python-docx for DOCX)
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │  CHUNK  │  Split into overlapping chunks (1000 chars, 200 overlap)
   └────┬────┘  using LangChain's RecursiveCharacterTextSplitter
        │
        ▼
   ┌─────────┐
   │  EMBED  │  Generate 768-dim vectors via Gemini Embedding 001
   └────┬────┘  with rate limiting (15 RPM) and batch processing
        │
        ▼
   ┌─────────┐
   │  STORE  │  Persist in ChromaDB with rich metadata:
   └─────────┘  source, owner, workspace, security_level, domain,
                document_id, timestamp, chunk_index
```

Each chunk gets:
- A **deterministic ID** based on content hash (so re-uploading the same document doesn't create duplicates)
- **Rich metadata** for filtering at query time (owner isolation, workspace scoping, security classification)
- **Cosine similarity indexing** via ChromaDB's HNSW algorithm

---

## How Parallelism Works

L90 uses `asyncio.gather` to run independent agents simultaneously. Here's the exact parallelism profile for each complexity tier:

| Tier | Timeline | Parallel? |
|---|---|---|
| **BASIC** | `Retriever → Generator → Grounding` | No — 3 sequential steps |
| **INTERMEDIATE** | `Retriever → (Analyzer ‖ Verifier) → Generator → Grounding` | **Yes** — Analyzer and Verifier run simultaneously |
| **ADVANCED** | `Retriever → Analyzer → Verifier → [Correction×3] → Generator → Grounding` | No — data dependencies enforce sequential flow |
| **RESEARCH_GRADE** | `Retriever → Analyzer → MathExecutor → Verifier → [Correction×3] → DeepReasoning → Generator → Grounding` | No — full chain required |

For INTERMEDIATE queries, the orchestrator clones the Blackboard into two copies, runs the Analyzer and Verifier in parallel, then merges their results back. This saves approximately 1-2 seconds per query.

The remaining tiers are sequential because each agent genuinely depends on the previous agent's output — the Verifier needs the Analyzer's results, the Corrector needs the Verifier's rejection reason, and the Generator needs everything.

---

## Audit Trail & Tracing

Every action by every agent is logged in the Blackboard's `reasoning_trace` — an append-only list of structured entries:

```json
{
  "timestamp": 1708234567.123,
  "agent": "VerificationAgent",
  "phase": "verification",
  "action": "claim_verified",
  "decision": "Claim supported by chunk abc123",
  "confidence": 0.92
}
```

This means you can:
1. **Replay** any query's reasoning process step by step
2. **Audit** which agents ran, what they decided, and how long each took
3. **Debug** failures by tracing exactly where the pipeline went wrong
4. **Measure** agent-level performance via the `agent_timings` metadata

---

## Design Decisions

### Why Fixed Agents Instead of Dynamic Spawning?

Kimi K2 spawns agents dynamically. We don't. The reason: **scientific accountability**. When a researcher asks "how did you arrive at this answer?", we can point to a fixed, documented pipeline: "The Retriever found these chunks, the Analyzer extracted these claims, the Verifier confirmed them against the source, and the Generator composed the answer." Dynamic spawning would make this trace unpredictable.

### Why a Separate Grounding Enforcer?

The Generator could theoretically self-check its grounding. But LLMs are biased toward their own output — they tend to rate their answers as grounded even when they're not. The Grounding Enforcer is **architecturally separate** from the Generator specifically to avoid this bias. It's an independent validation layer, like a peer reviewer who hasn't seen the draft.

### Why Reject Instead of Repair?

When grounding fails, L90 replaces the answer with "Insufficient verified information" instead of trying to fix it. This is deliberate. An answer that got fixed at the last minute has unknown reliability — did the fix actually make it accurate, or just make it sound more convincing? Rejecting outright is safer.

### Why Rule-Based Planning + LLM Fallback?

The Planner uses a two-tier approach: fast keyword-based classification first, LLM only if needed. This means simple queries like "What is the title of this document?" get classified as BASIC instantly (no LLM call), shaving 1-2 seconds off the response time. Only ambiguous or complex queries trigger the LLM planner.

### Why 5 Collections Instead of 1?

Source isolation is critical for multi-tenant and privacy-sensitive deployments:
- **User Private**: Only you can see your documents
- **Workspace**: Shared with your team
- **Approved Library**: Admin-curated reference material
- **Incognito**: Temporary, deleted on session end
- **Internal Library**: System reference documents

A single collection with metadata filters would technically work, but separate collections provide stronger isolation guarantees and simpler access control logic.

---

## Running the System

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Set GOOGLE_API_KEY in .env

# Start the server
python -m uvicorn l90.api.app:app --reload --port 8000

# Open the Web UI
# Navigate to http://localhost:8000

# Run tests
python -m pytest tests/ -v
```

---

*Built as a research project exploring trustworthy AI — proving that LLMs can be rigorous when you architect the right guardrails around them.*
