"""FastAPI application — L90 API endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from l90 import config
from l90.blackboard.blackboard import Blackboard
from l90.blackboard.persistence import PersistenceFactory
from l90.db import database as db
from l90.graph.builder import build_swarm_orchestrator
from l90.ingestion.pipeline import IngestionPipeline
from l90.modes.enforcement import ModeEnforcer, OperationMode
from l90.security.isolation import IsolationManager
from l90.tracing.logger import ReasoningTraceLogger
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# ── Init database on startup ──────────────────────────────────
db.init_db()

# ── Application setup ──────────────────────────────────────────

app = FastAPI(
    title="L90 — Closed-World Agentic Swarm RAG",
    description="Zero-hallucination, deterministic, multi-agent scientific reasoning system.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────

_static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Shared resources ───────────────────────────────────────────

_store = ChromaStore()
_pipeline = IngestionPipeline(store=_store)
_persistence = PersistenceFactory.get()
_isolation = IsolationManager(store=_store)
_mode_enforcer = ModeEnforcer()

# ── Helpers ────────────────────────────────────────────────────

def _get_user(token: str | None) -> dict:
    """Validate auth token and return user dict, or raise 401."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get_session_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


def _require_admin(user: dict) -> None:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


# ── Request / Response models ─────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    mode: str = Field(default="STRICT")
    user_id: str = Field(default="default_user")
    workspace_id: str = Field(default="default_workspace")
    session_id: str | None = Field(default=None)

class QueryResponse(BaseModel):
    session_id: str
    answer: str
    confidence_score: float
    mode: str
    reasoning_trace: list[dict[str, Any]]
    grounding_report: dict[str, Any]
    code_verification: dict[str, Any] = Field(default_factory=dict)
    deep_reasoning: list[dict[str, Any]] = Field(default_factory=list)
    latex_equations: list[str] = Field(default_factory=list)
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
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        "AUDIT | %s %s | status=%d | duration=%.3fs",
        request.method, request.url.path, response.status_code, duration,
    )
    return response


# ═══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "L90 API running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.2.0", "collections": _store.list_collections()}


@app.post("/login")
async def login(req: LoginRequest):
    user = db.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = db.create_session(user["id"])
    return {"token": token, "user": {"id": user["id"], "username": user["username"], "role": user["role"]}}


@app.post("/logout")
async def logout(x_token: str | None = Header(default=None)):
    if x_token:
        db.delete_session(x_token)
    return {"ok": True}


@app.get("/me")
async def get_me(x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


@app.get("/users")
async def list_users(x_token: str | None = Header(default=None)):
    _get_user(x_token)
    return db.list_all_users()


# ═══════════════════════════════════════════════════════════════
# QUERY (personal RAG chat)
# ═══════════════════════════════════════════════════════════════

@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)

    try:
        OperationMode(request.mode.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    trace_logger = ReasoningTraceLogger()
    # Build the dynamic SwarmOrchestrator (routes based on complexity)
    orchestrator = build_swarm_orchestrator(
        store=_store,
        trace_logger=trace_logger,
    )
    session_id = request.session_id or str(uuid.uuid4())

    initial_state = {
        "session_id": session_id,
        "query": request.query,
        "mode": request.mode.upper(),
        "user_id": user["username"],
        "workspace_id": request.workspace_id,
        "metadata": {"request_timestamp": time.time()},
    }

    try:
        result = await asyncio.wait_for(
            orchestrator.run(initial_state),
            timeout=config.PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Pipeline timed out after %.1fs for query: %s",
            config.PIPELINE_TIMEOUT_SECONDS, request.query[:80],
        )
        result = {
            "final_answer": "The query is still being processed. Please try again or simplify your question.",
            "confidence_score": 0.0,
            "mode": request.mode.upper(),
            "reasoning_trace": [{"agent": "SwarmOrchestrator", "action": "timeout", "detail": f"Exceeded {config.PIPELINE_TIMEOUT_SECONDS}s"}],
            "grounding_report": {"final_verdict": "TIMEOUT"},
            "metadata": {"timed_out": True, "timeout_seconds": config.PIPELINE_TIMEOUT_SECONDS},
        }
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    await _persistence.save(session_id, result)

    return QueryResponse(
        session_id=session_id,
        answer=result.get("final_answer", "Insufficient verified information."),
        confidence_score=result.get("confidence_score", 0.0),
        mode=result.get("mode", request.mode),
        reasoning_trace=result.get("reasoning_trace", []),
        grounding_report=result.get("grounding_report", {}),
        code_verification=result.get("code_verification", {}),
        deep_reasoning=result.get("deep_reasoning", []),
        latex_equations=result.get("latex_equations", []),
        metadata=result.get("metadata", {}),
    )


# ═══════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD (user's personal collection)
# ═══════════════════════════════════════════════════════════════

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default="user_private_collection"),
    owner: str = Form(default="default_user"),
    workspace_id: str = Form(default="default_workspace"),
    security_level: str = Form(default="standard"),
    domain: str = Form(default=""),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".docx"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / file.filename
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        chunks_stored = await _pipeline.ingest(
            file_path=temp_path, collection_name=collection,
            owner=owner, workspace_id=workspace_id,
            security_level=security_level, domain=domain,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)
        temp_dir.rmdir()

    return UploadResponse(
        filename=file.filename, collection=collection,
        chunks_stored=chunks_stored, document_id=f"doc_{uuid.uuid4().hex[:8]}",
    )


# ═══════════════════════════════════════════════════════════════
# ADMIN — Approved Library Upload
# ═══════════════════════════════════════════════════════════════

@app.post("/admin/upload")
async def admin_upload(
    file: UploadFile = File(...),
    domain: str = Form(default=""), 
    x_token: str | None = Header(default=None),
):
    user = _get_user(x_token)
    _require_admin(user)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".docx"}:
        raise HTTPException(status_code=400, detail=f"Unsupported: {suffix}")

    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / file.filename
    temp_path.write_bytes(await file.read())

    try:
        chunks = await _pipeline.ingest_approved_library_document(
            file_path=temp_path, domain=domain,
            document_name=file.filename,
            approval_authority=user["username"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)
        temp_dir.rmdir()

    return {"filename": file.filename, "chunks_stored": chunks, "collection": "approved_library"}


# ═══════════════════════════════════════════════════════════════
# NOTEBOOKS
# ═══════════════════════════════════════════════════════════════

@app.get("/notebooks")
async def get_notebooks(x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    return db.list_notebooks(user["id"])


@app.post("/notebooks")
async def create_notebook(title: str = "Untitled", x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    return db.create_notebook(user["id"], title)


class NotebookUpdate(BaseModel):
    title: str | None = None
    content: str | None = None

@app.put("/notebooks/{nb_id}")
async def update_notebook(nb_id: int, body: NotebookUpdate, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    result = db.update_notebook(nb_id, user["id"], body.title, body.content)
    if not result:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return result


class AppendRequest(BaseModel):
    text: str

@app.post("/notebooks/{nb_id}/append")
async def append_notebook(nb_id: int, body: AppendRequest, x_token: str | None = Header(default=None)):
    _get_user(x_token)
    result = db.append_to_notebook(nb_id, body.text)
    if not result:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return result


@app.delete("/notebooks/{nb_id}")
async def delete_notebook(nb_id: int, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    ok = db.delete_notebook(nb_id, user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"ok": True}


class ShareRequest(BaseModel):
    workspace_id: int
    permission: str = "read"

@app.post("/notebooks/{nb_id}/share")
async def share_nb(nb_id: int, body: ShareRequest, x_token: str | None = Header(default=None)):
    _get_user(x_token)
    db.share_notebook(nb_id, body.workspace_id, body.permission)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════
# WORKSPACES
# ═══════════════════════════════════════════════════════════════

@app.get("/workspaces")
async def get_workspaces(x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    return db.list_user_workspaces(user["id"])


class WorkspaceCreate(BaseModel):
    name: str

@app.post("/workspaces")
async def create_workspace(body: WorkspaceCreate, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    return db.create_workspace(user["id"], body.name)


@app.get("/workspaces/{ws_id}")
async def get_workspace(ws_id: int, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    ws = db.get_workspace(ws_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403, detail="Not a member")
    return ws


class MemberAdd(BaseModel):
    username: str
    permission: str = "read"

@app.post("/workspaces/{ws_id}/members")
async def add_member(ws_id: int, body: MemberAdd, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    ws = db.get_workspace(ws_id)
    if not ws or ws["owner_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Only owner can add members")
    ok = db.add_workspace_member(ws_id, body.username, body.permission)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


@app.get("/workspaces/{ws_id}/members")
async def get_members(ws_id: int, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403, detail="Not a member")
    return db.get_workspace_members(ws_id)


# ── Workspace docs ─────────────────────────────────────────

class DocAdd(BaseModel):
    doc_id: str
    doc_name: str

@app.post("/workspaces/{ws_id}/docs")
async def add_ws_doc(ws_id: int, body: DocAdd, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403)
    db.add_workspace_doc(ws_id, body.doc_id, body.doc_name, user["id"])
    return {"ok": True}


@app.get("/workspaces/{ws_id}/docs")
async def get_ws_docs(ws_id: int, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403)
    return db.get_workspace_docs(ws_id)


@app.put("/workspaces/{ws_id}/docs/{doc_id}/toggle")
async def toggle_doc(ws_id: int, doc_id: str, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403)
    db.toggle_workspace_doc(ws_id, doc_id)
    return {"ok": True}


# ── Workspace notebooks ───────────────────────────────────

@app.get("/workspaces/{ws_id}/notebooks")
async def get_ws_notebooks(ws_id: int, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403)
    return db.get_workspace_notebooks(ws_id)


# ── Workspace chat ─────────────────────────────────────────

class ChatMsg(BaseModel):
    content: str

@app.post("/workspaces/{ws_id}/chat")
async def ws_chat(ws_id: int, body: ChatMsg, x_token: str | None = Header(default=None)):
    """Shared RAG chat in a workspace — runs the query pipeline and saves messages."""
    user = _get_user(x_token)
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403)

    # Save user message
    db.add_chat_message(ws_id, user["id"], "user", body.content)

    # Run through pipeline
    trace_logger = ReasoningTraceLogger()
    orchestrator = build_swarm_orchestrator(store=_store, trace_logger=trace_logger)
    session_id = str(uuid.uuid4())

    initial_state = {
        "session_id": session_id,
        "query": body.content,
        "mode": "WORKSPACE",
        "user_id": user["username"],
        "workspace_id": str(ws_id),
        "metadata": {"request_timestamp": time.time()},
    }

    try:
        result = await asyncio.wait_for(
            orchestrator.run(initial_state),
            timeout=config.PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        result = {
            "final_answer": "The query is still being processed. Please try again.",
            "confidence_score": 0.0,
            "grounding_report": {"final_verdict": "TIMEOUT"},
            "reasoning_trace": [],
        }
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    answer = result.get("final_answer", "Insufficient verified information.")
    db.add_chat_message(ws_id, user["id"], "assistant", answer)

    return {
        "answer": answer,
        "confidence_score": result.get("confidence_score", 0.0),
        "grounding_report": result.get("grounding_report", {}),
        "reasoning_trace": result.get("reasoning_trace", []),
    }


@app.get("/workspaces/{ws_id}/chat")
async def get_ws_chat(ws_id: int, x_token: str | None = Header(default=None)):
    user = _get_user(x_token)
    if not db.is_workspace_member(ws_id, user["id"]):
        raise HTTPException(status_code=403)
    return db.get_chat_messages(ws_id)


# ═══════════════════════════════════════════════════════════════
# INCOGNITO SESSIONS
# ═══════════════════════════════════════════════════════════════

@app.post("/session/start", response_model=SessionResponse)
async def start_incognito_session():
    session_id = str(uuid.uuid4())
    _isolation.create_incognito_session(session_id)
    return SessionResponse(session_id=session_id, status="created")


@app.post("/session/end", response_model=SessionResponse)
async def end_incognito_session(session_id: str):
    _isolation.cleanup_incognito_session(session_id)
    await _persistence.delete(session_id)
    return SessionResponse(session_id=session_id, status="deleted")
