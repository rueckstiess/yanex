"""
Experiment graph for dependency-aware access to experiment pipelines.

This module provides the ExperimentGraph class that represents a connected
pipeline of experiments, enabling graph-level artifact loading, parameter
access, and filtering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from ..core.manager import ExperimentManager

from ..core.dependency_graph import DependencyGraph
from ..core.filtering import ExperimentFilter
from ..utils.dict_utils import flatten_dict
from ..utils.exceptions import AmbiguousArtifactError, ExperimentNotFoundError
from .experiment import Experiment

logger = logging.getLogger(__name__)


class ExperimentGraph:
    """A dependency graph of experiments with convenience accessors.

    By default, includes only upstream (dependencies) and downstream (dependents)
    experiments — the causal lineage. With ``weakly_connected=True``, includes
    all experiments in the weakly connected component (including sibling branches
    that share a common ancestor but are independent of the starting experiment).

    Provides graph-level artifact loading, parameter access, and filtering
    using the same filter syntax as yr.get_experiments().

    Examples:
        >>> import yanex.results as yr
        >>> graph = yr.get_graph("abc123")              # upstream + downstream only
        >>> graph = yr.get_graph("abc123", weakly_connected=True)  # full component
        >>> graph.experiments           # all experiments in the graph
        >>> graph.roots                 # experiments with no dependencies
        >>> graph.filter(script_pattern="train.py")  # filter by script
        >>> graph.load_artifact("dataset.json")       # search entire graph
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
        # Constrain to experiments in this graph
        filters["ids"] = list(self._experiments.keys())
        if "include_all" not in filters and "limit" not in filters:
            filters["include_all"] = True

        metadata_list = self._filter.filter_experiments(**filters)

        # Return cached Experiment objects
        return [
            self._experiments[m["id"]]
            for m in metadata_list
            if m["id"] in self._experiments
        ]

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

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value, searching all experiments in the graph.

        Returns the value if found in exactly one experiment (or if all
        experiments with the key have the same value). Raises ValueError
        if found in multiple experiments with different values.

        Args:
            key: Parameter key (supports dot notation like "model.lr").
            default: Default value if key not found in any experiment.

        Returns:
            Parameter value, or default if not found.

        Raises:
            ValueError: If parameter found in multiple experiments with
                different values.
        """
        found: list[tuple[str, Any]] = []
        for exp_id, exp in self._experiments.items():
            params = flatten_dict(exp.get_params())
            if key in params:
                found.append((exp_id, params[key]))
            elif "." in key:
                # Try nested access via get_param for dot notation
                from ..utils.dict_utils import get_nested_value

                value = get_nested_value(exp.get_params(), key)
                if value is not None:
                    found.append((exp_id, value))

        if not found:
            return default

        # Check if all values are the same
        values = [v for _, v in found]
        if all(v == values[0] for v in values):
            return values[0]

        # Multiple different values — ambiguous
        details = ", ".join(f"{eid}={val!r}" for eid, val in found[:5])
        raise ValueError(
            f"Parameter '{key}' found in {len(found)} experiments with different values: "
            f"{details}. Use graph.filter() to select specific experiments."
        )

    def get_metric(self, name: str) -> Any | None:
        """Get a metric's final value, searching all experiments in the graph.

        Returns the value if found in exactly one experiment (or if all
        experiments with the metric have the same value). Raises ValueError
        if found in multiple experiments with different values.

        Args:
            name: Metric name.

        Returns:
            Metric value, or None if not found in any experiment.

        Raises:
            ValueError: If metric found in multiple experiments with
                different values.
        """
        found: list[tuple[str, Any]] = []
        for exp_id, exp in self._experiments.items():
            value = exp.get_metric(name)
            if value is not None:
                found.append((exp_id, value))

        if not found:
            return None

        # Check if all values are the same
        values = [v for _, v in found]
        if all(v == values[0] for v in values):
            return values[0]

        # Multiple different values — ambiguous
        details = ", ".join(f"{eid}={val!r}" for eid, val in found[:5])
        raise ValueError(
            f"Metric '{name}' found in {len(found)} experiments with different values: "
            f"{details}. Use graph.filter() to select specific experiments."
        )
