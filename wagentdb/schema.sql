-- wagentdb schema
-- A compact, agent-friendly model for tracking training experiments.
--
-- Hierarchy:   project -> experiment -> run -> (metrics | logs | artifacts)
-- Graph:       edges link any node to any node with a typed relation
-- Reviews:     summaries hold agent-written notes/insights about any node
--
-- All timestamps are ISO-8601 UTC strings. JSON columns store serialized dicts/lists.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    metadata    TEXT,                 -- json object
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    hypothesis  TEXT,                 -- what the agent is trying to learn / prove
    metadata    TEXT,                 -- json object
    created_at  TEXT NOT NULL,
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
    name          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running',   -- running|finished|failed|crashed|killed
    config        TEXT,              -- json: hyperparameters / settings
    tags          TEXT,              -- json: list of strings
    notes         TEXT,
    created_by    TEXT,              -- agent id or name that owns the run
    git_commit    TEXT,
    git_remote    TEXT,
    system        TEXT,              -- json: host/gpu/python/library info
    summary       TEXT,              -- json: final / best metrics for quick scanning
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    heartbeat_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_project    ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);

-- Time-series metrics: one row per (run, key, step).
CREATE TABLE IF NOT EXISTS metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    step      INTEGER,
    value     REAL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key, step);

-- Free-form training logs (stdout lines, events, agent commentary).
CREATE TABLE IF NOT EXISTS logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    level     TEXT NOT NULL DEFAULT 'info',   -- debug|info|warning|error
    message   TEXT NOT NULL,
    step      INTEGER,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_run ON logs(run_id, id);

-- Artifacts: large blobs stored as objects in the object store (R2),
-- with their metadata indexed here.
CREATE TABLE IF NOT EXISTS artifacts (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    type        TEXT,              -- model|checkpoint|dataset|plot|config|file
    storage_key TEXT NOT NULL,     -- object key in the object store / R2 bucket
    size_bytes  INTEGER,
    checksum    TEXT,              -- sha256 hex
    metadata    TEXT,              -- json object
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);

-- The graph. A typed edge between any two nodes. This is what lets agents
-- trace lineage (forked_from, finetuned_from, resumed_from), compare runs,
-- and group runs into experiments.
CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src_type   TEXT NOT NULL,   -- run|experiment|artifact|project
    src_id     TEXT NOT NULL,
    dst_type   TEXT NOT NULL,
    dst_id     TEXT NOT NULL,
    relation   TEXT NOT NULL,   -- forked_from|resumed_from|finetuned_from|uses_artifact|produces|compared_with|part_of|...
    metadata   TEXT,            -- json object
    created_at TEXT NOT NULL,
    UNIQUE(src_type, src_id, dst_type, dst_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_type, dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation);

-- Agent-written summaries / reviews / insights about any node.
CREATE TABLE IF NOT EXISTS summaries (
    id           TEXT PRIMARY KEY,
    subject_type TEXT NOT NULL,   -- run|experiment|project
    subject_id   TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'summary',  -- summary|review|insight|hypothesis
    content      TEXT NOT NULL,
    created_by   TEXT,            -- which agent wrote it
    metadata     TEXT,            -- json object
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_subject ON summaries(subject_type, subject_id);

-- Internal key/value (schema version, etc.)
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
