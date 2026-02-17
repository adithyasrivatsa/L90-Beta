-- ╔══════════════════════════════════════════════════════════════╗
-- ║  L90 — Supabase Schema (copy-paste ready)                  ║
-- ║  Run this in Supabase SQL Editor to set up production DB   ║
-- ╚══════════════════════════════════════════════════════════════╝

-- ── Users ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username   TEXT UNIQUE NOT NULL,
    password   TEXT NOT NULL,  -- sha256 hash
    role       TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    email      TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ── Sessions ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ── Notebooks ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notebooks (
    id         BIGSERIAL PRIMARY KEY,
    owner_id   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL DEFAULT 'Untitled',
    content    TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- ── Workspaces ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workspaces (
    id         BIGSERIAL PRIMARY KEY,
    name       TEXT NOT NULL,
    owner_id   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ── Workspace Members ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS workspace_members (
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    permission   TEXT NOT NULL DEFAULT 'read' CHECK (permission IN ('read', 'write', 'append')),
    PRIMARY KEY (workspace_id, user_id)
);

-- ── Workspace Documents ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS workspace_docs (
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    doc_id       TEXT NOT NULL,
    doc_name     TEXT NOT NULL,
    enabled      BOOLEAN DEFAULT true,
    added_by     UUID NOT NULL REFERENCES users(id),
    PRIMARY KEY (workspace_id, doc_id)
);

-- ── Notebook Shares ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notebook_shares (
    notebook_id  BIGINT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    permission   TEXT NOT NULL DEFAULT 'read' CHECK (permission IN ('read', 'write', 'append')),
    PRIMARY KEY (notebook_id, workspace_id)
);

-- ── Chat Messages (workspace shared chat) ───────────────────
CREATE TABLE IF NOT EXISTS chat_messages (
    id           BIGSERIAL PRIMARY KEY,
    workspace_id BIGINT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id      UUID NOT NULL REFERENCES users(id),
    role         TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- ═══════════════════════════════════════════════════════════
-- RLS Policies
-- ═══════════════════════════════════════════════════════════

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE notebooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE workspace_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE workspace_docs ENABLE ROW LEVEL SECURITY;
ALTER TABLE notebook_shares ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- Users can read themselves
CREATE POLICY "users_read_self" ON users
    FOR SELECT USING (auth.uid() = id);

-- Users can read all usernames (for adding members)
CREATE POLICY "users_read_all_usernames" ON users
    FOR SELECT USING (true);

-- Notebooks: owner can CRUD
CREATE POLICY "notebooks_owner" ON notebooks
    FOR ALL USING (auth.uid() = owner_id);

-- Notebooks: workspace members can read shared notebooks
CREATE POLICY "notebooks_shared_read" ON notebooks
    FOR SELECT USING (
        id IN (
            SELECT ns.notebook_id FROM notebook_shares ns
            JOIN workspace_members wm ON ns.workspace_id = wm.workspace_id
            WHERE wm.user_id = auth.uid()
        )
    );

-- Workspaces: members can read
CREATE POLICY "workspaces_member_read" ON workspaces
    FOR SELECT USING (
        id IN (SELECT workspace_id FROM workspace_members WHERE user_id = auth.uid())
        OR owner_id = auth.uid()
    );

-- Workspaces: owner can manage
CREATE POLICY "workspaces_owner_manage" ON workspaces
    FOR ALL USING (owner_id = auth.uid());

-- Workspace members: members can read member list
CREATE POLICY "wm_read" ON workspace_members
    FOR SELECT USING (
        workspace_id IN (SELECT workspace_id FROM workspace_members WHERE user_id = auth.uid())
    );

-- Workspace members: workspace owner can manage
CREATE POLICY "wm_owner_manage" ON workspace_members
    FOR ALL USING (
        workspace_id IN (SELECT id FROM workspaces WHERE owner_id = auth.uid())
    );

-- Chat messages: workspace members can read and insert
CREATE POLICY "chat_member_read" ON chat_messages
    FOR SELECT USING (
        workspace_id IN (SELECT workspace_id FROM workspace_members WHERE user_id = auth.uid())
    );

CREATE POLICY "chat_member_insert" ON chat_messages
    FOR INSERT WITH CHECK (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = auth.uid() AND permission IN ('write', 'append')
        )
    );

-- ═══════════════════════════════════════════════════════════
-- Seed Data (run once after table creation)
-- ═══════════════════════════════════════════════════════════

-- Admin: Batman / Joker  (sha256 of 'Joker')
INSERT INTO users (username, password, role) VALUES
    ('Batman',  '729af84476dca4811e42ae7c0c74c96b7f8e6e20a34b25951be403bec2ec81f4', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Users: user1-user10 / 1234  (sha256 of '1234')
INSERT INTO users (username, password, role) VALUES
    ('user1',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user2',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user3',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user4',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user5',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user6',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user7',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user8',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user9',  '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user'),
    ('user10', '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4', 'user')
ON CONFLICT (username) DO NOTHING;
