"""SQLite database — users, notebooks, workspaces, permissions."""

from __future__ import annotations

import sqlite3
import hashlib
import os
import time
from pathlib import Path
from typing import Any

_DB_PATH = Path(os.environ.get("L90_DB_PATH", "./l90_local.db"))

# ── Hardcoded credentials ────────────────────────────────────
SEED_USERS = [
    ("Batman", "Joker", "admin"),
    *[(f"user{i}", "1234", "user") for i in range(1, 11)],
]


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables and seed data if needed."""
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE NOT NULL,
        password  TEXT NOT NULL,
        role      TEXT NOT NULL DEFAULT 'user',
        email     TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
        token     TEXT PRIMARY KEY,
        user_id   INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS notebooks (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_id  INTEGER NOT NULL,
        title     TEXT NOT NULL DEFAULT 'Untitled',
        content   TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (owner_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS workspaces (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        name      TEXT NOT NULL,
        owner_id  INTEGER NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (owner_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS workspace_members (
        workspace_id INTEGER NOT NULL,
        user_id      INTEGER NOT NULL,
        permission   TEXT NOT NULL DEFAULT 'read',
        PRIMARY KEY (workspace_id, user_id),
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS workspace_docs (
        workspace_id INTEGER NOT NULL,
        doc_id       TEXT NOT NULL,
        doc_name     TEXT NOT NULL,
        enabled      INTEGER DEFAULT 1,
        added_by     INTEGER NOT NULL,
        PRIMARY KEY (workspace_id, doc_id),
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
        FOREIGN KEY (added_by) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS notebook_shares (
        notebook_id  INTEGER NOT NULL,
        workspace_id INTEGER NOT NULL,
        permission   TEXT NOT NULL DEFAULT 'read',
        PRIMARY KEY (notebook_id, workspace_id),
        FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id INTEGER NOT NULL,
        user_id      INTEGER NOT NULL,
        role         TEXT NOT NULL DEFAULT 'user',
        content      TEXT NOT NULL,
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """)

    # Seed users
    existing = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing == 0:
        for uname, pw, role in SEED_USERS:
            c.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (uname, _hash(pw), role),
            )
    conn.commit()
    conn.close()


# ── Auth helpers ─────────────────────────────────────────────

def authenticate(username: str, password: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, _hash(password)),
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def create_session(user_id: int) -> str:
    import uuid
    token = uuid.uuid4().hex
    conn = get_conn()
    conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))
    conn.commit()
    conn.close()
    return token


def get_session_user(token: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT u.* FROM users u JOIN sessions s ON u.id=s.user_id WHERE s.token=?",
        (token,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()


# ── Notebook CRUD ────────────────────────────────────────────

def list_notebooks(owner_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM notebooks WHERE owner_id=? ORDER BY updated_at DESC", (owner_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_notebook(owner_id: int, title: str = "Untitled") -> dict:
    conn = get_conn()
    c = conn.execute(
        "INSERT INTO notebooks (owner_id, title) VALUES (?, ?) RETURNING *",
        (owner_id, title),
    )
    row = c.fetchone()
    conn.commit()
    conn.close()
    return dict(row)


def update_notebook(nb_id: int, owner_id: int, title: str | None = None, content: str | None = None) -> dict | None:
    conn = get_conn()
    nb = conn.execute("SELECT * FROM notebooks WHERE id=? AND owner_id=?", (nb_id, owner_id)).fetchone()
    if not nb:
        conn.close()
        return None
    t = title if title is not None else nb["title"]
    c = content if content is not None else nb["content"]
    conn.execute(
        "UPDATE notebooks SET title=?, content=?, updated_at=datetime('now') WHERE id=?",
        (t, c, nb_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM notebooks WHERE id=?", (nb_id,)).fetchone()
    conn.close()
    return dict(row)


def append_to_notebook(nb_id: int, text: str) -> dict | None:
    conn = get_conn()
    nb = conn.execute("SELECT * FROM notebooks WHERE id=?", (nb_id,)).fetchone()
    if not nb:
        conn.close()
        return None
    new_content = (nb["content"] or "") + "\n\n---\n\n" + text
    conn.execute(
        "UPDATE notebooks SET content=?, updated_at=datetime('now') WHERE id=?",
        (new_content, nb_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM notebooks WHERE id=?", (nb_id,)).fetchone()
    conn.close()
    return dict(row)


def delete_notebook(nb_id: int, owner_id: int) -> bool:
    conn = get_conn()
    c = conn.execute("DELETE FROM notebooks WHERE id=? AND owner_id=?", (nb_id, owner_id))
    conn.commit()
    conn.close()
    return c.rowcount > 0


# ── Notebook sharing ─────────────────────────────────────────

def share_notebook(nb_id: int, workspace_id: int, permission: str = "read") -> None:
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO notebook_shares (notebook_id, workspace_id, permission) VALUES (?, ?, ?)",
        (nb_id, workspace_id, permission),
    )
    conn.commit()
    conn.close()


def get_workspace_notebooks(workspace_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT n.*, ns.permission, u.username as owner_name
        FROM notebooks n
        JOIN notebook_shares ns ON n.id = ns.notebook_id
        JOIN users u ON n.owner_id = u.id
        WHERE ns.workspace_id = ?
    """, (workspace_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Workspace CRUD ───────────────────────────────────────────

def list_user_workspaces(user_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT DISTINCT w.*, wm.permission
        FROM workspaces w
        LEFT JOIN workspace_members wm ON w.id = wm.workspace_id AND wm.user_id = ?
        WHERE w.owner_id = ? OR wm.user_id = ?
        ORDER BY w.created_at DESC
    """, (user_id, user_id, user_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_workspace(owner_id: int, name: str) -> dict:
    conn = get_conn()
    c = conn.execute(
        "INSERT INTO workspaces (name, owner_id) VALUES (?, ?) RETURNING *",
        (name, owner_id),
    )
    row = c.fetchone()
    ws_id = row["id"]
    # owner is automatically a write member
    conn.execute(
        "INSERT INTO workspace_members (workspace_id, user_id, permission) VALUES (?, ?, 'write')",
        (ws_id, owner_id),
    )
    conn.commit()
    conn.close()
    return dict(row)


def add_workspace_member(ws_id: int, username: str, permission: str = "read") -> bool:
    conn = get_conn()
    user = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if not user:
        conn.close()
        return False
    conn.execute(
        "INSERT OR REPLACE INTO workspace_members (workspace_id, user_id, permission) VALUES (?, ?, ?)",
        (ws_id, user["id"], permission),
    )
    conn.commit()
    conn.close()
    return True


def get_workspace_members(ws_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT u.id, u.username, u.role, wm.permission
        FROM workspace_members wm
        JOIN users u ON wm.user_id = u.id
        WHERE wm.workspace_id = ?
    """, (ws_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_workspace_member(ws_id: int, user_id: int) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM workspace_members WHERE workspace_id=? AND user_id=?",
        (ws_id, user_id),
    ).fetchone()
    conn.close()
    return row is not None


def get_workspace(ws_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM workspaces WHERE id=?", (ws_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Workspace docs ───────────────────────────────────────────

def add_workspace_doc(ws_id: int, doc_id: str, doc_name: str, added_by: int) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO workspace_docs (workspace_id, doc_id, doc_name, enabled, added_by) VALUES (?, ?, ?, 1, ?)",
        (ws_id, doc_id, doc_name, added_by),
    )
    conn.commit()
    conn.close()


def toggle_workspace_doc(ws_id: int, doc_id: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT enabled FROM workspace_docs WHERE workspace_id=? AND doc_id=?",
        (ws_id, doc_id),
    ).fetchone()
    if not row:
        conn.close()
        return False
    new_val = 0 if row["enabled"] else 1
    conn.execute(
        "UPDATE workspace_docs SET enabled=? WHERE workspace_id=? AND doc_id=?",
        (new_val, ws_id, doc_id),
    )
    conn.commit()
    conn.close()
    return True


def get_workspace_docs(ws_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM workspace_docs WHERE workspace_id=?", (ws_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Chat messages ────────────────────────────────────────────

def add_chat_message(ws_id: int, user_id: int, role: str, content: str) -> dict:
    conn = get_conn()
    c = conn.execute(
        "INSERT INTO chat_messages (workspace_id, user_id, role, content) VALUES (?, ?, ?, ?) RETURNING *",
        (ws_id, user_id, role, content),
    )
    row = c.fetchone()
    conn.commit()
    conn.close()
    return dict(row)


def get_chat_messages(ws_id: int, limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT cm.*, u.username FROM chat_messages cm
        JOIN users u ON cm.user_id = u.id
        WHERE cm.workspace_id = ?
        ORDER BY cm.created_at ASC
        LIMIT ?
    """, (ws_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all_users() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT id, username, role FROM users ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]
