"""
Experiment graph for dependency-aware access to experiment pipelines.

This module provides the ExperimentGraph class that represents a connected
pipeline of experiments, enabling graph-level artifact loading, parameter
access, comparison, and filtering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx

if TYPE_CHECKING:
    import pandas as pd

    from ..core.manager import ExperimentManager

from ..core.access_resolver import AccessResolver, parse_canonical_key
from ..core.dependency_graph import DependencyGraph
from ..core.filtering import ExperimentFilter
from ..utils.dict_utils import deep_merge, flatten_dict
from ..utils.exceptions import (
    AmbiguousArtifactError,
    ExperimentNotFoundError,
    KeyNotFoundError,
)
from .experiment import Experiment

logger = logging.getLogger(__name__)


class ExperimentGraph:
    """A dependency graph of experiments with convenience accessors.

    By default, includes only upstream (dependencies) and downstream (dependents)
    experiments — the causal lineage. With ``weakly_connected=True``, includes
    all experiments in the weakly connected component (including sibling branches
    that share a common ancestor but are independent of the starting experiment).

    Provides graph-level artifact loading, parameter access, comparison, and
    filtering using the same filter syntax as yr.get_experiments().

    Examples:
        >>> import yanex.results as yr
        >>> graph = yr.get_graph("abc123")              # upstream + downstream only
        >>> graph = yr.get_graph("abc123", weakly_connected=True)  # full component
        >>> graph.experiments           # all experiments in the graph
        >>> graph.roots                 # experiments with no dependencies
        >>> graph.filter(script_pattern="train.py")  # filter by script
        >>> graph.load_artifact("dataset.json")       # search entire graph
        >>> graph.compare(script_pattern="eval.py", include_dep_params=True)
    """

    def __init__(
        self,
        manager: ExperimentManager,
        experiments: dict[str, Experiment],
        digraph: nx.DiGraph,
    ):
        """Initialize ExperimentGraph.

        Use ExperimentGraph.build() to construct instances.

        Args:
            manager: ExperimentManager for storage access.
            experiments: Dict mapping experiment ID to Experiment object.
            digraph: NetworkX DiGraph with edges in data-flow direction.
        """
        self._manager = manager
        self._experiments = experiments
        self._digraph = digraph
        self._filter = ExperimentFilter(manager=manager)
        self._resolver: AccessResolver | None = None
        self._resolver_built = False

    @classmethod
    def build(
        cls,
        manager: ExperimentManager,
        experiment_id: str,
        *,
        weakly_connected: bool = False,
    ) -> ExperimentGraph:
        """Build an ExperimentGraph from a starting experiment.

        By default, returns only the upstream (dependencies) and downstream
        (dependents) experiments — the causal lineage of the starting experiment.

        With ``weakly_connected=True``, returns the full weakly connected
        component including sibling branches (experiments that share a common
        ancestor but are independent of the starting experiment).

        Args:
            manager: ExperimentManager instance.
            experiment_id: Starting experiment ID.
            weakly_connected: If True, include all experiments in the weakly
                connected component (including siblings). If False (default),
                include only upstream and downstream experiments.

        Returns:
            ExperimentGraph containing the selected experiments.

        Raises:
            ExperimentNotFoundError: If experiment_id doesn't exist.
        """
        dep_graph = DependencyGraph(manager.storage)

        if not dep_graph.experiment_exists(experiment_id):
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found")

        if weakly_connected:
            subgraph = dep_graph.get_connected_component(experiment_id)
        else:
            subgraph = dep_graph.get_lineage(experiment_id, max_depth=10_000)

        experiments = {}
        for node_id in subgraph.nodes():
            try:
                experiments[node_id] = Experiment(node_id, manager=manager)
            except ExperimentNotFoundError:
                logger.warning("Skipping experiment %s: not found", node_id)
                continue

        return cls(manager=manager, experiments=experiments, digraph=subgraph)

    # --- Properties ---

    @property
    def experiments(self) -> list[Experiment]:
        """All experiments in the graph."""
        return list(self._experiments.values())

    @property
    def roots(self) -> list[Experiment]:
        """Experiments with no dependencies (in-degree 0 in data-flow graph)."""
        return [
            self._experiments[node]
            for node in self._digraph.nodes()
            if self._digraph.in_degree(node) == 0 and node in self._experiments
        ]

    @property
    def leaves(self) -> list[Experiment]:
        """Experiments with no dependents (out-degree 0 in data-flow graph)."""
        return [
            self._experiments[node]
            for node in self._digraph.nodes()
            if self._digraph.out_degree(node) == 0 and node in self._experiments
        ]

    @property
    def digraph(self) -> nx.DiGraph:
        """The underlying NetworkX DiGraph (edges in data-flow direction)."""
        return self._digraph

    # --- Container protocol ---

    def __getitem__(self, experiment_id: str) -> Experiment:
        """Get an experiment by ID.

        Args:
            experiment_id: The experiment ID.

        Raises:
            KeyError: If experiment_id is not in the graph.
        """
        if experiment_id not in self._experiments:
            raise KeyError(
                f"Experiment '{experiment_id}' not in graph. "
                f"Graph contains {len(self._experiments)} experiments."
            )
        return self._experiments[experiment_id]

    def __len__(self) -> int:
        return len(self._experiments)

    def __iter__(self):
        return iter(self._experiments.values())

    def __contains__(self, experiment_id: str) -> bool:
        return experiment_id in self._experiments

    def __repr__(self) -> str:
        n = len(self._experiments)
        edges = self._digraph.number_of_edges()
        return f"ExperimentGraph(experiments={n}, edges={edges})"

    # --- Internal helpers ---

    def _get_resolver(self) -> AccessResolver | None:
        """Lazily build and cache an AccessResolver from graph experiments."""
        if not self._resolver_built:
            from .manager import _build_resolver_from_experiments

            self._resolver = _build_resolver_from_experiments(
                list(self._experiments.values())
            )
            self._resolver_built = True
        return self._resolver

    def _inject_graph_ids(self, filters: dict) -> dict:
        """Inject graph experiment IDs into filter kwargs."""
        filters["ids"] = list(self._experiments.keys())
        if "include_all" not in filters and "limit" not in filters:
            filters["include_all"] = True
        return filters

    # --- Filtering ---

    def filter(self, **filters) -> list[Experiment]:
        """Filter experiments in the graph.

        Accepts the same filter kwargs as yr.get_experiments() and yr.find(),
        including status, tags, name, script_pattern, time ranges, etc.

        Args:
            **filters: Filter arguments (same as ExperimentFilter).

        Returns:
            List of matching Experiment objects from this graph.

        Examples:
            >>> graph.filter(script_pattern="train.py")
            >>> graph.filter(status="completed", tags=["sweep"])
        """
        filters = self._inject_graph_ids(filters)
        metadata_list = self._filter.filter_experiments(**filters)

        # Return cached Experiment objects
        return [
            self._experiments[m["id"]]
            for m in metadata_list
            if m["id"] in self._experiments
        ]

    # --- Comparison and metrics ---

    def compare(
        self,
        params: str | list[str] = "auto",
        metrics: str | list[str] = "auto",
        meta: str | list[str] = "auto",
        include_dep_params: bool = False,
        **filters,
    ) -> pd.DataFrame:
        """Compare experiments in the graph as a pandas DataFrame.

        Accepts the same kwargs as ``yr.compare()``, scoped to experiments in
        this graph. Additional filter kwargs further narrow the selection.

        Args:
            params: Parameter columns — "auto", "all", "none", or list of names.
            metrics: Metric columns — "auto", "all", "none", or list of names.
            meta: Metadata columns — "auto", "all", "none", or list of names.
            include_dep_params: If True, include parameters from upstream
                dependencies (merged with local params, local takes precedence).
            **filters: Additional filter kwargs (script_pattern, status, etc.).

        Returns:
            pandas DataFrame with ``param:``, ``metric:``, ``meta:`` columns.

        Examples:
            >>> graph.compare()
            >>> graph.compare(script_pattern="eval.py", include_dep_params=True)
            >>> graph.compare(params=["lr"], metrics=["accuracy"], status="completed")
        """
        from .manager import ResultsManager

        rm = ResultsManager(storage_path=self._manager.experiments_dir)
        filters = self._inject_graph_ids(filters)
        return rm.compare_experiments(
            params=params,
            metrics=metrics,
            meta=meta,
            include_dep_params=include_dep_params,
            **filters,
        )

    def get_metrics(
        self,
        *,
        metrics: str | list[str] | None = None,
        params: list[str] | Literal["auto", "all", "none"] = "auto",
        meta: list[str] | None = None,
        include_dep_params: bool = False,
        **filters,
    ) -> pd.DataFrame:
        """Get time-series metrics from experiments in the graph.

        Returns long-format (tidy) DataFrame, identical to ``yr.get_metrics()``
        but scoped to experiments in this graph.

        Args:
            metrics: Which metrics — None (all), str, or list of names.
            params: Which param columns — "auto", "all", "none", or list.
            meta: Metadata columns for faceting (e.g., ["name"]).
            include_dep_params: If True, include upstream dependency params.
            **filters: Additional filter kwargs (script_pattern, status, etc.).

        Returns:
            DataFrame with columns [experiment_id, step, metric_name, value,
            <params>, <meta>].

        Examples:
            >>> graph.get_metrics()
            >>> graph.get_metrics(script_pattern="train.py", metrics=["loss"])
            >>> graph.get_metrics(include_dep_params=True, params="all")
        """
        from .manager import ResultsManager

        rm = ResultsManager(storage_path=self._manager.experiments_dir)
        filters = self._inject_graph_ids(filters)
        return rm.get_metrics(
            metrics=metrics,
            params=params,
            meta=meta,
            include_dep_params=include_dep_params,
            **filters,
        )

    # --- Graph-level data access ---

    def load_artifact(
        self,
        filename: str,
        loader: Any | None = None,
        format: str | None = None,
    ) -> Any | None:
        """Load an artifact, searching all experiments in the graph.

        Returns None if the artifact doesn't exist in any experiment.
        Raises AmbiguousArtifactError if found in multiple experiments.

        Args:
            filename: Name of artifact to load.
            loader: Optional custom loader function (path) -> object.
            format: Optional format name for explicit format selection.

        Returns:
            Loaded object, or None if not found.

        Raises:
            AmbiguousArtifactError: If artifact found in multiple experiments.
        """
        found_in: list[str] = []
        for exp_id in self._experiments:
            if self._manager.storage.artifact_exists(
                exp_id, filename, include_archived=True
            ):
                found_in.append(exp_id)

        if len(found_in) == 0:
            return None
        elif len(found_in) == 1:
            return self._manager.storage.load_artifact(
                found_in[0], filename, loader, include_archived=True, format=format
            )
        else:
            raise AmbiguousArtifactError(filename, found_in)

    def get_params(self) -> dict[str, Any]:
        """Get all parameters merged across all experiments in the graph.

        Returns a nested dict of all parameters. If the same parameter key
        exists in multiple experiments with the same value, it appears once.
        If the same key has different values, raises ValueError.

        Useful for linear pipelines where each experiment has distinct config.
        For fan-out pipelines, use ``graph.filter()`` + per-experiment access.

        Returns:
            Nested dict of merged parameters.

        Raises:
            ValueError: If any parameter key has conflicting values across
                experiments.

        Examples:
            >>> params = graph.get_params()
            >>> # {"dataset": "mnist", "learning_rate": 0.01, "epochs": 100}
        """
        # Pass 1: conflict detection on flattened keys
        seen: dict[str, tuple[str, Any]] = {}
        for exp_id, exp in self._experiments.items():
            for key, value in flatten_dict(exp.get_params()).items():
                if key in seen:
                    prev_id, prev_value = seen[key]
                    if prev_value != value:
                        raise ValueError(
                            f"Parameter '{key}' has conflicting values: "
                            f"{prev_id}={prev_value!r}, {exp_id}={value!r}. "
                            f"Use graph.filter() to select specific experiments."
                        )
                else:
                    seen[key] = (exp_id, value)

        # Pass 2: deep merge for nested structure (safe — no conflicts)
        merged: dict[str, Any] = {}
        for exp in self._experiments.values():
            merged = deep_merge(merged, exp.get_params())
        return merged

    def get_param(self, key: str) -> Any:
        """Get a parameter value, searching all experiments in the graph.

        Uses AccessResolver for sub-path resolution (e.g., ``"lr"`` resolves
        to ``"model.lr"`` if unambiguous).

        Returns the value if found in exactly one experiment (or if all
        experiments with the key have the same value).

        Args:
            key: Parameter key. Supports dot notation (``"model.lr"``),
                sub-path shorthand (``"lr"``), and canonical form
                (``"param:model.lr"``).

        Returns:
            Parameter value.

        Raises:
            KeyNotFoundError: If key not found in any experiment.
            AmbiguousKeyError: If key matches multiple parameter paths.
            ValueError: If parameter found in multiple experiments with
                different values.

        Examples:
            >>> graph.get_param("learning_rate")
            0.01
            >>> graph.get_param("lr")  # resolves to "model.lr" if unambiguous
            0.001
        """
        resolver = self._get_resolver()
        if resolver is None:
            raise KeyNotFoundError(key, scope="param")

        # Resolve key via AccessResolver — raises KeyNotFoundError or
        # AmbiguousKeyError if resolution fails
        canonical = resolver.resolve(key, scope="param")
        _, resolved_path = parse_canonical_key(canonical)

        # Search all experiments for the resolved key
        found: list[tuple[str, Any]] = []
        for exp_id, exp in self._experiments.items():
            params = flatten_dict(exp.get_params())
            if resolved_path in params:
                found.append((exp_id, params[resolved_path]))

        if not found:
            raise KeyNotFoundError(key, scope="param")

        values = [v for _, v in found]
        if all(v == values[0] for v in values):
            return values[0]

        details = ", ".join(f"{eid}={val!r}" for eid, val in found[:5])
        raise ValueError(
            f"Parameter '{key}' found in {len(found)} experiments with different "
            f"values: {details}. Use graph.filter() to select specific experiments."
        )

    def get_metric(self, name: str) -> Any:
        """Get a metric's final value, searching all experiments in the graph.

        Uses AccessResolver for sub-path resolution.

        Returns the value if found in exactly one experiment (or if all
        experiments with the metric have the same value).

        Args:
            name: Metric name.

        Returns:
            Metric value.

        Raises:
            KeyNotFoundError: If metric not found in any experiment.
            ValueError: If metric found in multiple experiments with
                different values.

        Examples:
            >>> graph.get_metric("accuracy")
            0.95
        """
        resolver = self._get_resolver()
        if resolver is None:
            raise KeyNotFoundError(name, scope="metric")

        # Try to resolve via AccessResolver for sub-path resolution
        try:
            canonical = resolver.resolve(name, scope="metric")
            _, resolved_name = parse_canonical_key(canonical)
        except KeyNotFoundError:
            # Fall back to direct name — might exist in metrics but not in
            # the resolver's index (e.g., logged after resolver was built)
            resolved_name = name

        # Search all experiments for the metric
        found: list[tuple[str, Any]] = []
        for exp_id, exp in self._experiments.items():
            value = exp.get_metric(resolved_name)
            if value is not None:
                found.append((exp_id, value))

        if not found:
            raise KeyNotFoundError(name, scope="metric")

        values = [v for _, v in found]
        if all(v == values[0] for v in values):
            return values[0]

        details = ", ".join(f"{eid}={val!r}" for eid, val in found[:5])
        raise ValueError(
            f"Metric '{name}' found in {len(found)} experiments with different "
            f"values: {details}. Use graph.filter() to select specific experiments."
        )
