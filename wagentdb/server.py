"""FastAPI server exposing the Store over HTTP.

Run it with::

    wagentdb serve            # or: uvicorn wagentdb.server:app

Every endpoint is a thin wrapper over :class:`wagentdb.store.Store`, so the
HTTP API and the embedded Python API stay in lock-step. The OpenAPI schema at
``/docs`` doubles as machine-readable documentation for agents.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from . import models as m
from .config import Settings
from .store import NotFound, Store

_store: Optional[Store] = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(Settings.from_env())
    return _store


def set_store(store: Store) -> None:
    """Inject a Store (used by tests)."""
    global _store
    _store = store


def create_app(store: Optional[Store] = None) -> FastAPI:
    if store is not None:
        set_store(store)

    app = FastAPI(
        title="wagentdb",
        version="0.1.0",
        description="Agent-native experiment tracking. A wandb for AI agents, backed by R2 + SQLite.",
    )

    @app.exception_handler(NotFound)
    async def _not_found(_request, exc: NotFound):  # noqa: ANN001
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": get_store().settings.backend}

    # ----------------------------------------------------------- projects
    @app.post("/projects", response_model=m.Project)
    def create_project(data: m.ProjectCreate):
        return get_store().create_project(data)

    @app.get("/projects", response_model=list[m.Project])
    def list_projects():
        return get_store().list_projects()

    @app.get("/projects/{ref}", response_model=m.Project)
    def get_project(ref: str):
        return get_store().get_project(ref)

    # -------------------------------------------------------- experiments
    @app.post("/experiments", response_model=m.Experiment)
    def create_experiment(data: m.ExperimentCreate):
        return get_store().create_experiment(data)

    @app.get("/experiments", response_model=list[m.Experiment])
    def list_experiments(project: Optional[str] = None):
        return get_store().list_experiments(project)

    @app.get("/experiments/{ref}", response_model=m.Experiment)
    def get_experiment(ref: str, project: Optional[str] = None):
        return get_store().get_experiment(ref, project)

    # --------------------------------------------------------------- runs
    @app.post("/runs", response_model=m.Run)
    def create_run(data: m.RunCreate):
        return get_store().create_run(data)

    @app.get("/runs", response_model=list[m.Run])
    def list_runs(
        project: Optional[str] = None,
        experiment: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ):
        return get_store().list_runs(project, experiment, status, tag, limit)

    @app.get("/runs/{run_id}", response_model=m.Run)
    def get_run(run_id: str):
        return get_store().get_run(run_id)

    @app.patch("/runs/{run_id}", response_model=m.Run)
    def update_run(run_id: str, data: m.RunUpdate):
        return get_store().update_run(run_id, data)

    @app.post("/runs/{run_id}/heartbeat")
    def heartbeat(run_id: str):
        get_store().heartbeat(run_id)
        return {"ok": True}

    @app.get("/runs/{run_id}/report", response_model=m.RunReport)
    def run_report(run_id: str, log_tail: int = 20):
        return get_store().run_report(run_id, log_tail=log_tail)

    # ------------------------------------------------------------ metrics
    @app.post("/runs/{run_id}/metrics")
    def log_metrics(run_id: str, data: m.MetricLog):
        n = get_store().log_metrics(run_id, data.metrics, step=data.step)
        return {"logged": n}

    @app.get("/runs/{run_id}/metrics")
    def metric_keys(run_id: str):
        return {"keys": get_store().list_metric_keys(run_id)}

    @app.get("/runs/{run_id}/metrics/{key}", response_model=m.MetricSeries)
    def metric_series(run_id: str, key: str):
        return get_store().get_metric_series(run_id, key)

    # --------------------------------------------------------------- logs
    @app.post("/runs/{run_id}/logs")
    def log_message(run_id: str, data: m.LogLine):
        get_store().log_message(run_id, data.message, level=data.level, step=data.step)
        return {"ok": True}

    @app.get("/runs/{run_id}/logs")
    def get_logs(run_id: str, limit: int = 200, level: Optional[str] = None):
        return {"logs": get_store().get_logs(run_id, limit=limit, level=level)}

    # ---------------------------------------------------------- artifacts
    @app.post("/runs/{run_id}/artifacts", response_model=m.Artifact)
    def log_artifact(
        run_id: str,
        name: str = Query(...),
        type: Optional[str] = Query(None),
        body: bytes = Body(..., media_type="application/octet-stream"),
    ):
        return get_store().log_artifact(run_id, name, body, type=type)

    @app.get("/runs/{run_id}/artifacts", response_model=list[m.Artifact])
    def list_artifacts(run_id: str):
        return get_store().list_artifacts(run_id)

    @app.get("/artifacts/{artifact_id}", response_model=m.Artifact)
    def get_artifact(artifact_id: str):
        return get_store().get_artifact(artifact_id)

    @app.get("/artifacts/{artifact_id}/download")
    def download_artifact(artifact_id: str):
        store = get_store()
        art = store.get_artifact(artifact_id)
        # Prefer a presigned URL (R2); fall back to streaming the bytes (local).
        url = store.artifact_url(artifact_id)
        if url and url.startswith("http"):
            return JSONResponse({"url": url})
        data = store.get_artifact_bytes(artifact_id)
        return Response(content=data, media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{art.name}"'})

    # --------------------------------------------------------- graph/edges
    @app.post("/edges", response_model=m.Edge)
    def add_edge(data: m.EdgeCreate):
        return get_store().add_edge(data)

    @app.get("/nodes/{node_id}/lineage")
    def lineage(node_id: str):
        lin = get_store().lineage(node_id)
        return {k: [e.model_dump() for e in v] for k, v in lin.items()}

    @app.get("/nodes/{node_id}/ancestry", response_model=list[m.Edge])
    def ancestry(node_id: str):
        return get_store().ancestry(node_id)

    @app.get("/graph")
    def graph(project: Optional[str] = None):
        return get_store().graph(project)

    # ----------------------------------------------------------- summaries
    @app.post("/summaries", response_model=m.Summary)
    def add_summary(data: m.SummaryCreate):
        return get_store().add_summary(data)

    @app.get("/nodes/{node_id}/summaries", response_model=list[m.Summary])
    def list_summaries(node_id: str):
        return get_store().list_summaries(node_id)

    return app


# Default app for `uvicorn wagentdb.server:app`
app = create_app()
