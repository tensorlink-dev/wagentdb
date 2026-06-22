"""The wagentdb client: a small, wandb-like API for agents.

Two transport modes, identical surface:

* ``embedded`` - talk straight to a local :class:`Store` (R2 + sqlite). No
  server needed; perfect for an agent running a training job on a box.
* ``http``     - talk to a running wagentdb server over HTTP.

Logging::

    import wagentdb
    run = wagentdb.init(project="ssm-ds", name="designB-lr3e4",
                        config={"lr": 3e-4, "design": "B"})
    for step in range(1000):
        run.log({"loss": loss, "val/acc": acc}, step=step)
    run.log_artifact("model.pt", type="model")
    run.summary["best_acc"] = 0.91
    run.finish()

Reviewing (what an agent does afterwards)::

    db = wagentdb.connect(project="ssm-ds")
    for r in db.runs(status="finished"):
        print(r.name, r.summary)
    report = db.report(run_id)          # one-call digest of a run
    db.add_summary(run_id, "Design B overfits after step 600.", kind="review")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from . import models as m
from .config import Settings
from .store import Store


# ====================================================================== backends
class _EmbeddedBackend:
    def __init__(self, settings: Optional[Settings] = None, store: Optional[Store] = None):
        self.store = store or Store(settings or Settings.from_env())

    def request(self, method: str, path: str, **kw):  # pragma: no cover - not used
        raise NotImplementedError

    # The embedded backend just exposes the store directly.


class _HTTPBackend:
    def __init__(self, url: str, token: Optional[str] = None, client: Any = None):
        self.url = url.rstrip("/")
        if client is not None:
            self._client = client
        else:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "HTTP mode requires httpx. Install with `pip install wagentdb[client]`."
                ) from exc
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            self._client = httpx.Client(base_url=self.url, headers=headers, timeout=30.0)

    def json(self, method: str, path: str, **kw) -> Any:
        resp = self._client.request(method, path, **kw)
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return None


# ====================================================================== Run
class _Summary(dict):
    """A dict whose writes are pushed to the run summary on the next flush."""

    def __init__(self, run: "Run", initial: Optional[Dict[str, Any]] = None):
        super().__init__(initial or {})
        self._run = run

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._run._update(summary={key: value})


class Run:
    """A handle to a single training run."""

    def __init__(self, backend: Union[_EmbeddedBackend, _HTTPBackend], run: m.Run):
        self._backend = backend
        self._run = run
        self.summary = _Summary(self, run.summary or {})

    # convenience accessors
    @property
    def id(self) -> str:
        return self._run.id

    @property
    def name(self) -> str:
        return self._run.name

    @property
    def project_id(self) -> str:
        return self._run.project_id

    def __repr__(self) -> str:
        return f"<Run {self._run.id} name={self._run.name!r} status={self._run.status}>"

    # ---- logging ----
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics at an optional step."""
        if isinstance(self._backend, _EmbeddedBackend):
            self._backend.store.log_metrics(self.id, metrics, step=step)
        else:
            self._backend.json("POST", f"/runs/{self.id}/metrics",
                               json={"metrics": metrics, "step": step})

    def log_text(self, message: str, level: str = "info", step: Optional[int] = None) -> None:
        if isinstance(self._backend, _EmbeddedBackend):
            self._backend.store.log_message(self.id, message, level=level, step=step)
        else:
            self._backend.json("POST", f"/runs/{self.id}/logs",
                               json={"message": message, "level": level, "step": step})

    def log_artifact(
        self,
        path_or_bytes: Union[str, bytes],
        name: Optional[str] = None,
        type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> m.Artifact:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            data = bytes(path_or_bytes)
            name = name or "artifact.bin"
        else:
            import os
            with open(path_or_bytes, "rb") as fh:
                data = fh.read()
            name = name or os.path.basename(path_or_bytes)
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.log_artifact(self.id, name, data, type=type, metadata=metadata)
        params = {"name": name}
        if type:
            params["type"] = type
        result = self._backend.json("POST", f"/runs/{self.id}/artifacts",
                                    params=params, content=data)
        return m.Artifact(**result)

    def link(self, target: Union["Run", str], relation: str = "forked_from",
             target_type: str = "run", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a graph edge from this run to another node (e.g. ``forked_from``)."""
        target_id = target.id if isinstance(target, Run) else target
        edge = m.EdgeCreate(src_type="run", src_id=self.id, dst_type=target_type,
                            dst_id=target_id, relation=relation, metadata=metadata)
        if isinstance(self._backend, _EmbeddedBackend):
            self._backend.store.add_edge(edge)
        else:
            self._backend.json("POST", "/edges", json=edge.model_dump())

    def _update(self, **fields: Any) -> None:
        update = m.RunUpdate(**fields)
        if isinstance(self._backend, _EmbeddedBackend):
            self._run = self._backend.store.update_run(self.id, update)
        else:
            data = self._backend.json("PATCH", f"/runs/{self.id}",
                                      json=update.model_dump(exclude_none=True))
            self._run = m.Run(**data)

    def config_update(self, **kw: Any) -> None:
        self._update(config=kw)

    def finish(self, status: str = "finished") -> None:
        # Flush any summary that was set dict-style.
        self._update(status=status, summary=dict(self.summary))


# ====================================================================== read client
class DB:
    """Read/review interface — what an agent uses to study past experiments."""

    def __init__(self, backend: Union[_EmbeddedBackend, _HTTPBackend],
                 project: Optional[str] = None):
        self._backend = backend
        self.project = project

    def projects(self) -> List[m.Project]:
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.list_projects()
        return [m.Project(**p) for p in self._backend.json("GET", "/projects")]

    def experiments(self, project: Optional[str] = None) -> List[m.Experiment]:
        project = project or self.project
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.list_experiments(project)
        return [m.Experiment(**e) for e in
                self._backend.json("GET", "/experiments", params={"project": project})]

    def runs(self, project: Optional[str] = None, experiment: Optional[str] = None,
             status: Optional[str] = None, tag: Optional[str] = None,
             limit: int = 100) -> List[m.Run]:
        project = project or self.project
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.list_runs(project, experiment, status, tag, limit)
        params = {k: v for k, v in dict(project=project, experiment=experiment,
                                        status=status, tag=tag, limit=limit).items() if v is not None}
        return [m.Run(**r) for r in self._backend.json("GET", "/runs", params=params)]

    def run(self, run_id: str) -> m.Run:
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.get_run(run_id)
        return m.Run(**self._backend.json("GET", f"/runs/{run_id}"))

    def report(self, run_id: str, log_tail: int = 20) -> m.RunReport:
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.run_report(run_id, log_tail=log_tail)
        return m.RunReport(**self._backend.json("GET", f"/runs/{run_id}/report",
                                                params={"log_tail": log_tail}))

    def history(self, run_id: str, key: str) -> m.MetricSeries:
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.get_metric_series(run_id, key)
        return m.MetricSeries(**self._backend.json("GET", f"/runs/{run_id}/metrics/{key}"))

    def lineage(self, node_id: str) -> Dict[str, List[m.Edge]]:
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.lineage(node_id)
        raw = self._backend.json("GET", f"/nodes/{node_id}/lineage")
        return {k: [m.Edge(**e) for e in v] for k, v in raw.items()}

    def graph(self, project: Optional[str] = None) -> Dict[str, Any]:
        project = project or self.project
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.graph(project)
        return self._backend.json("GET", "/graph", params={"project": project} if project else {})

    def add_summary(self, subject_id: str, content: str, kind: str = "summary",
                    subject_type: str = "run", created_by: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> m.Summary:
        data = m.SummaryCreate(subject_type=subject_type, subject_id=subject_id,
                               content=content, kind=kind, created_by=created_by,
                               metadata=metadata)
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.add_summary(data)
        return m.Summary(**self._backend.json("POST", "/summaries", json=data.model_dump()))

    def summaries(self, node_id: str) -> List[m.Summary]:
        if isinstance(self._backend, _EmbeddedBackend):
            return self._backend.store.list_summaries(node_id)
        return [m.Summary(**s) for s in self._backend.json("GET", f"/nodes/{node_id}/summaries")]


# ====================================================================== entrypoints
def _make_backend(url: Optional[str], settings: Optional[Settings],
                  store: Optional[Store], token: Optional[str],
                  http_client: Any = None) -> Union[_EmbeddedBackend, _HTTPBackend]:
    if url:
        return _HTTPBackend(url, token=token, client=http_client)
    return _EmbeddedBackend(settings=settings, store=store)


def init(
    project: str,
    name: Optional[str] = None,
    experiment: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    created_by: Optional[str] = None,
    *,
    url: Optional[str] = None,
    token: Optional[str] = None,
    settings: Optional[Settings] = None,
    store: Optional[Store] = None,
    http_client: Any = None,
    fork_from: Optional[str] = None,
) -> Run:
    """Start a run. Returns a :class:`Run` handle. Mirrors ``wandb.init``."""
    backend = _make_backend(url, settings, store, token, http_client)
    create = m.RunCreate(project=project, name=name, experiment=experiment,
                         config=config, tags=tags, notes=notes, created_by=created_by)
    if isinstance(backend, _EmbeddedBackend):
        run_model = backend.store.create_run(create)
    else:
        run_model = m.Run(**backend.json("POST", "/runs", json=create.model_dump(exclude_none=True)))
    run = Run(backend, run_model)
    if fork_from:
        run.link(fork_from, relation="forked_from")
    return run


def connect(
    project: Optional[str] = None,
    *,
    url: Optional[str] = None,
    token: Optional[str] = None,
    settings: Optional[Settings] = None,
    store: Optional[Store] = None,
    http_client: Any = None,
) -> DB:
    """Open a read/review handle for studying experiments."""
    backend = _make_backend(url, settings, store, token, http_client)
    return DB(backend, project=project)
