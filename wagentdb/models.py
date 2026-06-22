"""Pydantic models: the public data shapes returned by the store and API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- create
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExperimentCreate(BaseModel):
    project: str = Field(description="Project id or name")
    name: str
    description: Optional[str] = None
    hypothesis: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RunCreate(BaseModel):
    project: str = Field(description="Project id or name")
    name: Optional[str] = None
    experiment: Optional[str] = Field(default=None, description="Experiment id or name")
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    created_by: Optional[str] = None
    git_commit: Optional[str] = None
    git_remote: Optional[str] = None
    system: Optional[Dict[str, Any]] = None


class RunUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    system: Optional[Dict[str, Any]] = None


class MetricPoint(BaseModel):
    key: str
    value: float
    step: Optional[int] = None


class MetricLog(BaseModel):
    metrics: Dict[str, float]
    step: Optional[int] = None


class LogLine(BaseModel):
    message: str
    level: str = "info"
    step: Optional[int] = None


class EdgeCreate(BaseModel):
    src_type: str
    src_id: str
    dst_type: str
    dst_id: str
    relation: str
    metadata: Optional[Dict[str, Any]] = None


class SummaryCreate(BaseModel):
    subject_type: str
    subject_id: str
    content: str
    kind: str = "summary"
    created_by: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------- read
class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


class Experiment(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    hypothesis: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


class Run(BaseModel):
    id: str
    project_id: str
    experiment_id: Optional[str] = None
    name: str
    status: str
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    created_by: Optional[str] = None
    git_commit: Optional[str] = None
    git_remote: Optional[str] = None
    system: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    heartbeat_at: Optional[str] = None


class Artifact(BaseModel):
    id: str
    run_id: str
    name: str
    type: Optional[str] = None
    storage_key: str
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


class Edge(BaseModel):
    id: int
    src_type: str
    src_id: str
    dst_type: str
    dst_id: str
    relation: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


class Summary(BaseModel):
    id: str
    subject_type: str
    subject_id: str
    kind: str
    content: str
    created_by: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


class MetricSeries(BaseModel):
    key: str
    steps: List[Optional[int]]
    values: List[float]
    timestamps: List[str]


class RunReport(BaseModel):
    """Everything an agent needs to review a run in one call."""

    run: Run
    project: Optional[Project] = None
    experiment: Optional[Experiment] = None
    metric_keys: List[str]
    latest_metrics: Dict[str, float]
    artifacts: List[Artifact]
    lineage: Dict[str, List[Edge]]
    summaries: List[Summary]
    log_tail: List[Dict[str, Any]]
