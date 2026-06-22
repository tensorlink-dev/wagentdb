"""The Store: high-level experiment-tracking operations.

This is the heart of wagentdb. Both the FastAPI server and the embedded Python
client call into a ``Store``. It owns a :class:`Database` (sqlite metadata) and
an :class:`ObjectStore` (R2 blobs), and exposes operations in terms of the
pydantic models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import models as m
from .config import Settings
from .database import Database
from .objectstore import ObjectStore, build_object_store
from .utils import dumps, loads, new_id, sha256_bytes, utcnow

# Relations that describe how a run descends from / depends on another node.
LINEAGE_RELATIONS = {
    "forked_from",
    "resumed_from",
    "finetuned_from",
    "uses_artifact",
    "part_of",
    "compared_with",
    "produces",
}


class NotFound(Exception):
    pass


class Store:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        object_store: Optional[ObjectStore] = None,
        database: Optional[Database] = None,
    ):
        self.settings = settings or Settings()
        self.object_store = object_store or build_object_store(self.settings)
        self.db = database or Database(self.settings, self.object_store)

    def close(self) -> None:
        self.db.close()

    # ============================================================ projects
    def create_project(self, data: m.ProjectCreate) -> m.Project:
        existing = self.db.query_one("SELECT * FROM projects WHERE name = ?", (data.name,))
        if existing:
            return _project(existing)
        pid = new_id("proj")
        now = utcnow()
        self.db.write(
            "INSERT INTO projects(id, name, description, metadata, created_at) "
            "VALUES(?,?,?,?,?)",
            (pid, data.name, data.description, dumps(data.metadata), now),
        )
        return _project(self.db.query_one("SELECT * FROM projects WHERE id = ?", (pid,)))

    def get_project(self, ref: str) -> m.Project:
        row = self._project_row(ref)
        if not row:
            raise NotFound(f"project not found: {ref}")
        return _project(row)

    def _project_row(self, ref: str) -> Optional[Dict[str, Any]]:
        return self.db.query_one(
            "SELECT * FROM projects WHERE id = ? OR name = ?", (ref, ref)
        )

    def list_projects(self) -> List[m.Project]:
        return [_project(r) for r in self.db.query_all("SELECT * FROM projects ORDER BY created_at")]

    # ========================================================= experiments
    def create_experiment(self, data: m.ExperimentCreate) -> m.Experiment:
        proj = self.get_project(data.project)
        existing = self.db.query_one(
            "SELECT * FROM experiments WHERE project_id = ? AND name = ?",
            (proj.id, data.name),
        )
        if existing:
            return _experiment(existing)
        eid = new_id("exp")
        self.db.write(
            "INSERT INTO experiments(id, project_id, name, description, hypothesis, metadata, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (eid, proj.id, data.name, data.description, data.hypothesis, dumps(data.metadata), utcnow()),
        )
        return _experiment(self.db.query_one("SELECT * FROM experiments WHERE id = ?", (eid,)))

    def get_experiment(self, ref: str, project: Optional[str] = None) -> m.Experiment:
        row = self._experiment_row(ref, project)
        if not row:
            raise NotFound(f"experiment not found: {ref}")
        return _experiment(row)

    def _experiment_row(self, ref: str, project: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if project:
            proj = self._project_row(project)
            pid = proj["id"] if proj else project
            return self.db.query_one(
                "SELECT * FROM experiments WHERE (id = ? OR name = ?) AND project_id = ?",
                (ref, ref, pid),
            )
        return self.db.query_one(
            "SELECT * FROM experiments WHERE id = ? OR name = ?", (ref, ref)
        )

    def list_experiments(self, project: Optional[str] = None) -> List[m.Experiment]:
        if project:
            proj = self.get_project(project)
            rows = self.db.query_all(
                "SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at", (proj.id,)
            )
        else:
            rows = self.db.query_all("SELECT * FROM experiments ORDER BY created_at")
        return [_experiment(r) for r in rows]

    # ================================================================ runs
    def create_run(self, data: m.RunCreate) -> m.Run:
        proj = self.create_project(m.ProjectCreate(name=data.project)) if not self._project_row(
            data.project
        ) else self.get_project(data.project)

        exp_id = None
        if data.experiment:
            row = self._experiment_row(data.experiment, project=proj.id)
            if not row:
                exp = self.create_experiment(
                    m.ExperimentCreate(project=proj.id, name=data.experiment)
                )
                exp_id = exp.id
            else:
                exp_id = row["id"]

        rid = new_id("run")
        now = utcnow()
        name = data.name or rid
        self.db.write(
            "INSERT INTO runs(id, project_id, experiment_id, name, status, config, tags, notes, "
            "created_by, git_commit, git_remote, system, summary, created_at, started_at, heartbeat_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid, proj.id, exp_id, name, "running",
                dumps(data.config), dumps(data.tags), data.notes, data.created_by,
                data.git_commit, data.git_remote, dumps(data.system), dumps({}),
                now, now, now,
            ),
        )
        # If attached to an experiment, record the graph edge.
        if exp_id:
            self.add_edge(m.EdgeCreate(
                src_type="run", src_id=rid,
                dst_type="experiment", dst_id=exp_id, relation="part_of",
            ))
        return self.get_run(rid)

    def get_run(self, run_id: str) -> m.Run:
        row = self.db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not row:
            raise NotFound(f"run not found: {run_id}")
        return _run(row)

    def update_run(self, run_id: str, data: m.RunUpdate) -> m.Run:
        run = self.get_run(run_id)
        fields: Dict[str, Any] = {}
        if data.status is not None:
            fields["status"] = data.status
            if data.status in ("finished", "failed", "crashed", "killed"):
                fields["finished_at"] = utcnow()
        if data.notes is not None:
            fields["notes"] = data.notes
        if data.tags is not None:
            fields["tags"] = dumps(data.tags)
        if data.config is not None:
            merged = {**(run.config or {}), **data.config}
            fields["config"] = dumps(merged)
        if data.summary is not None:
            merged = {**(run.summary or {}), **data.summary}
            fields["summary"] = dumps(merged)
        if data.system is not None:
            merged = {**(run.system or {}), **data.system}
            fields["system"] = dumps(merged)
        fields["heartbeat_at"] = utcnow()

        if fields:
            cols = ", ".join(f"{k} = ?" for k in fields)
            self.db.write(
                f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id)
            )
        return self.get_run(run_id)

    def list_runs(
        self,
        project: Optional[str] = None,
        experiment: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> List[m.Run]:
        clauses: List[str] = []
        params: List[Any] = []
        if project:
            prow = self._project_row(project)
            clauses.append("project_id = ?")
            params.append(prow["id"] if prow else project)
        if experiment:
            erow = self._experiment_row(experiment, project=project)
            clauses.append("experiment_id = ?")
            params.append(erow["id"] if erow else experiment)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.db.query_all(
            f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ?", params
        )
        return [_run(r) for r in rows]

    def heartbeat(self, run_id: str) -> None:
        self.db.write("UPDATE runs SET heartbeat_at = ? WHERE id = ?", (utcnow(), run_id))

    # ============================================================= metrics
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None) -> int:
        self.get_run(run_id)  # validate
        now = utcnow()
        rows = [(run_id, key, step, float(val), now) for key, val in metrics.items()]
        self.db.write_many(
            "INSERT INTO metrics(run_id, key, step, value, timestamp) VALUES(?,?,?,?,?)", rows
        )
        # Keep the run summary current with the latest value of each metric.
        run = self.get_run(run_id)
        summary = dict(run.summary or {})
        summary.update({k: float(v) for k, v in metrics.items()})
        self.db.write("UPDATE runs SET summary = ?, heartbeat_at = ? WHERE id = ?",
                      (dumps(summary), now, run_id))
        return len(rows)

    def list_metric_keys(self, run_id: str) -> List[str]:
        rows = self.db.query_all(
            "SELECT DISTINCT key FROM metrics WHERE run_id = ? ORDER BY key", (run_id,)
        )
        return [r["key"] for r in rows]

    def get_metric_series(self, run_id: str, key: str) -> m.MetricSeries:
        rows = self.db.query_all(
            "SELECT step, value, timestamp FROM metrics WHERE run_id = ? AND key = ? "
            "ORDER BY id", (run_id, key),
        )
        return m.MetricSeries(
            key=key,
            steps=[r["step"] for r in rows],
            values=[r["value"] for r in rows],
            timestamps=[r["timestamp"] for r in rows],
        )

    def latest_metrics(self, run_id: str) -> Dict[str, float]:
        rows = self.db.query_all(
            "SELECT key, value FROM metrics WHERE run_id = ? AND id IN "
            "(SELECT MAX(id) FROM metrics WHERE run_id = ? GROUP BY key)",
            (run_id, run_id),
        )
        return {r["key"]: r["value"] for r in rows}

    # ================================================================ logs
    def log_message(self, run_id: str, message: str, level: str = "info",
                    step: Optional[int] = None) -> None:
        self.db.write(
            "INSERT INTO logs(run_id, level, message, step, timestamp) VALUES(?,?,?,?,?)",
            (run_id, level, message, step, utcnow()),
        )

    def get_logs(self, run_id: str, limit: int = 200, level: Optional[str] = None) -> List[Dict[str, Any]]:
        if level:
            rows = self.db.query_all(
                "SELECT level, message, step, timestamp FROM logs WHERE run_id = ? AND level = ? "
                "ORDER BY id DESC LIMIT ?", (run_id, level, limit),
            )
        else:
            rows = self.db.query_all(
                "SELECT level, message, step, timestamp FROM logs WHERE run_id = ? "
                "ORDER BY id DESC LIMIT ?", (run_id, limit),
            )
        return list(reversed(rows))

    # =========================================================== artifacts
    def log_artifact(
        self,
        run_id: str,
        name: str,
        data: bytes,
        type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> m.Artifact:
        self.get_run(run_id)
        aid = new_id("art")
        key = f"artifacts/{run_id}/{aid}/{name}"
        self.object_store.put_bytes(key, data)
        self.db.write(
            "INSERT INTO artifacts(id, run_id, name, type, storage_key, size_bytes, checksum, metadata, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (aid, run_id, name, type, key, len(data), sha256_bytes(data), dumps(metadata), utcnow()),
        )
        return _artifact(self.db.query_one("SELECT * FROM artifacts WHERE id = ?", (aid,)))

    def list_artifacts(self, run_id: str) -> List[m.Artifact]:
        rows = self.db.query_all(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at", (run_id,)
        )
        return [_artifact(r) for r in rows]

    def get_artifact(self, artifact_id: str) -> m.Artifact:
        row = self.db.query_one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        if not row:
            raise NotFound(f"artifact not found: {artifact_id}")
        return _artifact(row)

    def get_artifact_bytes(self, artifact_id: str) -> bytes:
        art = self.get_artifact(artifact_id)
        return self.object_store.get_bytes(art.storage_key)

    def artifact_url(self, artifact_id: str, expires: int = 3600) -> Optional[str]:
        art = self.get_artifact(artifact_id)
        return self.object_store.presigned_url(art.storage_key, expires=expires)

    # =============================================================== graph
    def add_edge(self, data: m.EdgeCreate) -> m.Edge:
        self.db.write(
            "INSERT OR IGNORE INTO edges(src_type, src_id, dst_type, dst_id, relation, metadata, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (data.src_type, data.src_id, data.dst_type, data.dst_id, data.relation,
             dumps(data.metadata), utcnow()),
        )
        row = self.db.query_one(
            "SELECT * FROM edges WHERE src_type=? AND src_id=? AND dst_type=? AND dst_id=? AND relation=?",
            (data.src_type, data.src_id, data.dst_type, data.dst_id, data.relation),
        )
        return _edge(row)

    def edges_from(self, node_id: str) -> List[m.Edge]:
        return [_edge(r) for r in self.db.query_all(
            "SELECT * FROM edges WHERE src_id = ? ORDER BY id", (node_id,))]

    def edges_to(self, node_id: str) -> List[m.Edge]:
        return [_edge(r) for r in self.db.query_all(
            "SELECT * FROM edges WHERE dst_id = ? ORDER BY id", (node_id,))]

    def lineage(self, node_id: str) -> Dict[str, List[m.Edge]]:
        """Direct in/out edges, split by direction (one hop)."""
        return {"out": self.edges_from(node_id), "in": self.edges_to(node_id)}

    def ancestry(self, node_id: str, max_depth: int = 25) -> List[m.Edge]:
        """Transitively follow outgoing lineage edges (forked_from, etc.)."""
        seen: set = set()
        out: List[m.Edge] = []
        frontier = [node_id]
        depth = 0
        while frontier and depth < max_depth:
            nxt: List[str] = []
            for nid in frontier:
                for edge in self.edges_from(nid):
                    if edge.relation not in LINEAGE_RELATIONS:
                        continue
                    key = (edge.src_id, edge.dst_id, edge.relation)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(edge)
                    nxt.append(edge.dst_id)
            frontier = nxt
            depth += 1
        return out

    def graph(self, project: Optional[str] = None) -> Dict[str, Any]:
        """Node + edge view for the whole db (or one project)."""
        runs = self.list_runs(project=project, limit=10_000)
        run_ids = {r.id for r in runs}
        nodes = [{"id": r.id, "type": "run", "label": r.name, "status": r.status} for r in runs]
        for e in self.list_experiments(project):
            nodes.append({"id": e.id, "type": "experiment", "label": e.name})
        edges = self.db.query_all("SELECT * FROM edges")
        if project:
            keep = run_ids | {e.id for e in self.list_experiments(project)}
            edges = [e for e in edges if e["src_id"] in keep or e["dst_id"] in keep]
        return {"nodes": nodes, "edges": [_edge(e).model_dump() for e in edges]}

    # =========================================================== summaries
    def add_summary(self, data: m.SummaryCreate) -> m.Summary:
        sid = new_id("sum")
        self.db.write(
            "INSERT INTO summaries(id, subject_type, subject_id, kind, content, created_by, metadata, created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (sid, data.subject_type, data.subject_id, data.kind, data.content,
             data.created_by, dumps(data.metadata), utcnow()),
        )
        return _summary(self.db.query_one("SELECT * FROM summaries WHERE id = ?", (sid,)))

    def list_summaries(self, subject_id: str) -> List[m.Summary]:
        rows = self.db.query_all(
            "SELECT * FROM summaries WHERE subject_id = ? ORDER BY created_at", (subject_id,)
        )
        return [_summary(r) for r in rows]

    # ============================================================== report
    def run_report(self, run_id: str, log_tail: int = 20) -> m.RunReport:
        run = self.get_run(run_id)
        project = _project(self.db.query_one("SELECT * FROM projects WHERE id = ?", (run.project_id,)))
        experiment = None
        if run.experiment_id:
            erow = self.db.query_one("SELECT * FROM experiments WHERE id = ?", (run.experiment_id,))
            experiment = _experiment(erow) if erow else None
        return m.RunReport(
            run=run,
            project=project,
            experiment=experiment,
            metric_keys=self.list_metric_keys(run_id),
            latest_metrics=self.latest_metrics(run_id),
            artifacts=self.list_artifacts(run_id),
            lineage=self.lineage(run_id),
            summaries=self.list_summaries(run_id),
            log_tail=self.get_logs(run_id, limit=log_tail),
        )


# ----------------------------------------------------------- row -> model
def _project(r: Dict[str, Any]) -> m.Project:
    return m.Project(id=r["id"], name=r["name"], description=r["description"],
                     metadata=loads(r["metadata"]), created_at=r["created_at"])


def _experiment(r: Dict[str, Any]) -> m.Experiment:
    return m.Experiment(id=r["id"], project_id=r["project_id"], name=r["name"],
                        description=r["description"], hypothesis=r["hypothesis"],
                        metadata=loads(r["metadata"]), created_at=r["created_at"])


def _run(r: Dict[str, Any]) -> m.Run:
    return m.Run(
        id=r["id"], project_id=r["project_id"], experiment_id=r["experiment_id"],
        name=r["name"], status=r["status"], config=loads(r["config"]), tags=loads(r["tags"]),
        notes=r["notes"], created_by=r["created_by"], git_commit=r["git_commit"],
        git_remote=r["git_remote"], system=loads(r["system"]), summary=loads(r["summary"]),
        created_at=r["created_at"], started_at=r["started_at"], finished_at=r["finished_at"],
        heartbeat_at=r["heartbeat_at"],
    )


def _artifact(r: Dict[str, Any]) -> m.Artifact:
    return m.Artifact(id=r["id"], run_id=r["run_id"], name=r["name"], type=r["type"],
                      storage_key=r["storage_key"], size_bytes=r["size_bytes"],
                      checksum=r["checksum"], metadata=loads(r["metadata"]),
                      created_at=r["created_at"])


def _edge(r: Dict[str, Any]) -> m.Edge:
    return m.Edge(id=r["id"], src_type=r["src_type"], src_id=r["src_id"],
                  dst_type=r["dst_type"], dst_id=r["dst_id"], relation=r["relation"],
                  metadata=loads(r["metadata"]), created_at=r["created_at"])


def _summary(r: Dict[str, Any]) -> m.Summary:
    return m.Summary(id=r["id"], subject_type=r["subject_type"], subject_id=r["subject_id"],
                     kind=r["kind"], content=r["content"], created_by=r["created_by"],
                     metadata=loads(r["metadata"]), created_at=r["created_at"])
