"""HTTP response contracts; experiment metadata retains extensible research fields."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str | None = None
    status: str = "unknown"
    tags: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    name: str
    size: int
    modified: float


class ExperimentDetail(BaseModel):
    experiment: ExperimentSummary
    config: dict[str, Any]
    results: list[dict[str, Any]]
    metadata: dict[str, Any]
    artifacts: list[Artifact]


class ExperimentPage(BaseModel):
    experiments: list[ExperimentSummary]
    total: int
    page: int
    total_pages: int
    limit: int
    has_next: bool
    has_prev: bool


class StatusSummary(BaseModel):
    total_experiments: int
    archived_experiments: int
    status_counts: dict[str, int]


class GraphNode(BaseModel):
    id: str
    name: str | None = None
    status: str
    script: str
    params: dict[str, Any]
    metrics: dict[str, Any]


class GraphEdge(BaseModel):
    source: str
    target: str
    slot: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
