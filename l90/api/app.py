"""FastAPI application — L90 API endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from l90 import config
from l90.blackboard.blackboard import Blackboard
from l90.blackboard.persistence import PersistenceFactory
from l90.graph.builder import build_graph
from l90.ingestion.pipeline import IngestionPipeline
from l90.modes.enforcement import ModeEnforcer, OperationMode
from l90.security.isolation import IsolationManager
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# ── Application setup ──────────────────────────────────────────

app = FastAPI(
    title="L90 — Closed-World Agentic Swarm RAG",
    description="Zero-hallucination, deterministic, multi-agent scientific reasoning system.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared resources ───────────────────────────────────────────

_store = ChromaStore()
_pipeline = IngestionPipeline(store=_store)
_persistence = PersistenceFactory.get()
_isolation = IsolationManager(store=_store)
_mode_enforcer = ModeEnforcer()

# ── Request / Response models ─────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    mode: str = Field(default="STRICT", description="STRICT|PARTIAL|GENERAL|INCOGNITO|WORKSPACE")
    user_id: str = Field(default="default_user")
    workspace_id: str = Field(default="default_workspace")
    session_id: str | None = Field(default=None, description="For incognito: existing session ID")


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    confidence_score: float
    mode: str
    reasoning_trace: list[dict[str, Any]]
    grounding_report: dict[str, Any]
    metadata: dict[str, Any]


class UploadResponse(BaseModel):
    filename: str
    collection: str
    chunks_stored: int
    document_id: str


class SessionResponse(BaseModel):
    session_id: str
    status: str


# ── Audit logging middleware ───────────────────────────────────

@app.middleware("http")
async def audit_log_middleware(request, call_next):
    """Log every request for audit compliance."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        "AUDIT | %s %s | status=%d | duration=%.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "collections": _store.list_collections(),
    }


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """Execute a query through the L90 pipeline.

    1. Validates mode
    2. Builds the LangGraph pipeline
    3. Runs the full agent swarm
    4. Returns the grounded answer with reasoning trace
    """
    # Validate mode
    try:
        OperationMode(request.mode.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Valid: {[m.value for m in OperationMode]}",
        )

    # Set up trace logger for this request
    trace_logger = ReasoningTraceLogger()

    # Build and run the graph
    compiled_graph = build_graph(store=_store, trace_logger=trace_logger)

    session_id = request.session_id or str(uuid.uuid4())

    initial_state = {
        "session_id": session_id,
        "query": request.query,
        "mode": request.mode.upper(),
        "user_id": request.user_id,
        "workspace_id": request.workspace_id,
        "allowed_sources": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "verification_results": [],
        "verification_passed": False,
        "correction_results": [],
        "correction_loop_count": 0,
        "confidence_score": 0.0,
        "final_answer": "",
        "reasoning_trace": [],
        "execution_plan": {},
        "grounding_report": {},
        "metadata": {"request_timestamp": time.time()},
    }

    try:
        result = await compiled_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # Persist blackboard state
    await _persistence.save(session_id, result)

    return QueryResponse(
        session_id=session_id,
        answer=result.get("final_answer", "Insufficient verified information."),
        confidence_score=result.get("confidence_score", 0.0),
        mode=result.get("mode", request.mode),
        reasoning_trace=result.get("reasoning_trace", []),
        grounding_report=result.get("grounding_report", {}),
        metadata=result.get("metadata", {}),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default="user_private_collection"),
    owner: str = Form(default="default_user"),
    workspace_id: str = Form(default="default_workspace"),
    security_level: str = Form(default="standard"),
    domain: str = Form(default=""),
):
    """Upload and ingest a document into a specified collection."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md, .docx",
        )

    # Save to temp file
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / file.filename
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        chunks_stored = _pipeline.ingest(
            file_path=temp_path,
            collection_name=collection,
            owner=owner,
            workspace_id=workspace_id,
            security_level=security_level,
            domain=domain,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)
        temp_dir.rmdir()

    return UploadResponse(
        filename=file.filename,
        collection=collection,
        chunks_stored=chunks_stored,
        document_id=f"doc_{uuid.uuid4().hex[:8]}",
    )


@app.post("/session/start", response_model=SessionResponse)
async def start_incognito_session():
    """Start a new incognito session with isolated storage."""
    session_id = str(uuid.uuid4())
    _isolation.create_incognito_session(session_id)
    return SessionResponse(session_id=session_id, status="created")


@app.post("/session/end", response_model=SessionResponse)
async def end_incognito_session(session_id: str):
    """End an incognito session — deletes ALL associated data.

    Zero storage guarantee.
    """
    _isolation.cleanup_incognito_session(session_id)
    await _persistence.delete(session_id)
    return SessionResponse(session_id=session_id, status="deleted")
