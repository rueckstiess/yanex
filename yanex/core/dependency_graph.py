"""Dependency graph for efficient lineage traversal.

This module provides a DependencyGraph class that loads all experiment
dependencies once and enables fast bidirectional traversal for lineage queries.
"""

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .storage_composition import CompositeExperimentStorage


class DependencyGraph:
    """Loads all experiment dependencies once, enables fast bidirectional traversal.

    Builds two graphs:
    - Forward graph: dependent -> dependency (for upstream queries)
    - Reverse graph: dependency -> dependent (for downstream queries)

    This avoids recursive file reads for each lineage query by loading
    all dependencies into memory upfront.
    """

    def __init__(self, storage: "CompositeExperimentStorage"):
        """Initialize dependency graph by loading all experiments.

        Args:
            storage: Composite storage instance to load data from.
        """
        self._storage = storage
        self._forward = nx.DiGraph()  # dependent -> dependency (for upstream)
        self._reverse = nx.DiGraph()  # dependency -> dependent (for downstream)
        self._exp_metadata: dict[str, dict[str, str]] = {}  # id -> {name, status}
        self._load_all()

    def _load_all(self) -> None:
        """Load ALL experiment dependencies and metadata in one pass."""
        # Include archived experiments - they're part of lineage history
        for exp_id in self._storage.list_experiments(include_archived=True):
            # Load metadata for node labels
            try:
                meta = self._storage.load_metadata(exp_id, include_archived=True)
                # Extract just the filename from the full script_path
                script_path = meta.get("script_path", "")
                script_name = Path(script_path).name if script_path else ""
                self._exp_metadata[exp_id] = {
                    "name": meta.get("name", ""),
                    "status": meta.get("status", "unknown"),
                    "script": script_name,
                }
            except Exception:
                # Experiment might be corrupted - use defaults
                self._exp_metadata[exp_id] = {
                    "name": "",
                    "status": "unknown",
                    "script": "",
                }

            # Load dependencies and build graphs
            try:
                dep_data = self._storage.dependency_storage.load_dependencies(
                    exp_id, include_archived=True
                )
                for slot, dep_id in dep_data.get("dependencies", {}).items():
                    # Forward: from dependent to dependency
                    self._forward.add_edge(exp_id, dep_id, slot=slot)
                    # Reverse: from dependency to dependent
                    self._reverse.add_edge(dep_id, exp_id, slot=slot)
            except Exception:
                # No dependencies or error loading - continue
                pass

    def get_upstream(self, exp_id: str, max_depth: int = 10) -> nx.DiGraph:
        """Get all dependencies (upstream) as a subgraph.

        Traverses the forward graph (dependent -> dependency) starting from exp_id.

        Args:
            exp_id: Experiment ID to get upstream dependencies for.
            max_depth: Maximum traversal depth.

        Returns:
            NetworkX DiGraph with edges pointing from dependent to dependency.
            Nodes have 'name' and 'status' attributes.
        """
        subgraph = self._bfs_subgraph(exp_id, self._forward, max_depth)
        self._add_node_metadata(subgraph)
        return subgraph

    def get_downstream(self, exp_id: str, max_depth: int = 10) -> nx.DiGraph:
        """Get all dependents (downstream) as a subgraph.

        Traverses the reverse graph (dependency -> dependent) starting from exp_id.

        Args:
            exp_id: Experiment ID to get downstream dependents for.
            max_depth: Maximum traversal depth.

        Returns:
            NetworkX DiGraph with edges pointing from dependency to dependent.
            Nodes have 'name' and 'status' attributes.
        """
        subgraph = self._bfs_subgraph(exp_id, self._reverse, max_depth)
        self._add_node_metadata(subgraph)
        return subgraph

    def get_lineage(self, exp_id: str, max_depth: int = 10) -> nx.DiGraph:
        """Get both upstream and downstream combined as unified DAG.

        Args:
            exp_id: Experiment ID to get lineage for.
            max_depth: Maximum traversal depth in each direction.

        Returns:
            NetworkX DiGraph with edges pointing from dependency to dependent
            (data-flow direction). Nodes have 'name' and 'status' attributes.
        """
        upstream = self._bfs_subgraph(exp_id, self._forward, max_depth)
        downstream = self._bfs_subgraph(exp_id, self._reverse, max_depth)

        # Combine into single graph with consistent edge direction (dep -> dependent)
        combined = nx.DiGraph()

        # Add upstream edges - reverse direction for display (dep -> dependent)
        for u, v, data in upstream.edges(data=True):
            combined.add_edge(v, u, **data)

        # Add downstream edges - already in correct direction (dep -> dependent)
        for u, v, data in downstream.edges(data=True):
            combined.add_edge(u, v, **data)

        # Ensure the queried experiment is in the graph even with no deps
        if exp_id not in combined:
            combined.add_node(exp_id)

        self._add_node_metadata(combined)
        return combined

    def get_multi_upstream(self, exp_ids: list[str], max_depth: int = 10) -> nx.DiGraph:
        """Get combined upstream dependencies for multiple experiments.

        Merges upstream graphs for all given experiments into one unified graph.
        Single experiment is just len(exp_ids) == 1.

        Args:
            exp_ids: List of experiment IDs to get combined upstream for.
            max_depth: Maximum traversal depth.

        Returns:
            NetworkX DiGraph with edges in dependency->dependent direction.
            Nodes have 'name' and 'status' attributes.
        """
        combined = nx.DiGraph()

        for exp_id in exp_ids:
            # get_upstream returns dependent->dependency direction
            # We need to reverse for consistent dep->dependent display
            raw = self._bfs_subgraph(exp_id, self._forward, max_depth)
            for u, v, data in raw.edges(data=True):
                if not combined.has_edge(v, u):
                    combined.add_edge(v, u, **data)

            # Ensure experiment node is present even with no deps
            if exp_id not in combined:
                combined.add_node(exp_id)

        self._add_node_metadata(combined)
        return combined

    def get_multi_downstream(
        self, exp_ids: list[str], max_depth: int = 10
    ) -> nx.DiGraph:
        """Get combined downstream dependents for multiple experiments.

        Merges downstream graphs for all given experiments into one unified graph.
        Single experiment is just len(exp_ids) == 1.

        Args:
            exp_ids: List of experiment IDs to get combined downstream for.
            max_depth: Maximum traversal depth.

        Returns:
            NetworkX DiGraph with edges in dependency->dependent direction.
            Nodes have 'name' and 'status' attributes.
        """
        combined = nx.DiGraph()

        for exp_id in exp_ids:
            downstream = self._bfs_subgraph(exp_id, self._reverse, max_depth)
            for u, v, data in downstream.edges(data=True):
                if not combined.has_edge(u, v):
                    combined.add_edge(u, v, **data)

            # Ensure experiment node is present even with no dependents
            if exp_id not in combined:
                combined.add_node(exp_id)

        self._add_node_metadata(combined)
        return combined

    def get_multi_lineage(self, exp_ids: list[str], max_depth: int = 10) -> nx.DiGraph:
        """Get combined lineage for multiple experiments.

        Merges lineage graphs for all given experiments into one unified graph.
        This is the generalized form - single experiment is just len(exp_ids) == 1.

        Args:
            exp_ids: List of experiment IDs to get combined lineage for.
            max_depth: Maximum traversal depth in each direction.

        Returns:
            NetworkX DiGraph with edges in dependency->dependent direction.
            Nodes have 'name' and 'status' attributes.
        """
        combined = nx.DiGraph()

        for exp_id in exp_ids:
            lineage = self.get_lineage(exp_id, max_depth=max_depth)

            # Merge edges
            for u, v, data in lineage.edges(data=True):
                if not combined.has_edge(u, v):
                    combined.add_edge(u, v, **data)

            # Merge nodes (including isolated nodes)
            for node in lineage.nodes():
                if node not in combined:
                    combined.add_node(node)

        self._add_node_metadata(combined)
        return combined

    def _bfs_subgraph(
        self, start: str, graph: nx.DiGraph, max_depth: int
    ) -> nx.DiGraph:
        """BFS traversal up to max_depth, returns subgraph.

        Args:
            start: Starting node for traversal.
            graph: Graph to traverse (forward or reverse).
            max_depth: Maximum depth to traverse.

        Returns:
            Subgraph containing all reachable nodes within max_depth.
        """
        subgraph = nx.DiGraph()
        visited = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Skip if node is not in the graph (no dependencies/dependents)
            if node not in graph:
                continue

            for neighbor in graph.successors(node):
                edge_data = graph.edges[node, neighbor]
                subgraph.add_edge(node, neighbor, **edge_data)

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return subgraph

    def _add_node_metadata(self, graph: nx.DiGraph) -> None:
        """Add cached metadata (name, status) to graph nodes.

        Args:
            graph: Graph to add metadata to (modified in place).
        """
        for node in graph.nodes():
            if node in self._exp_metadata:
                graph.nodes[node].update(self._exp_metadata[node])
            else:
                # Experiment was deleted or doesn't exist
                graph.nodes[node].update({"name": "[deleted]", "status": "deleted"})

    def get_metadata(self, exp_id: str) -> dict[str, str]:
        """Get cached metadata for an experiment.

        Args:
            exp_id: Experiment ID.

        Returns:
            Dict with 'name' and 'status' keys.
        """
        return self._exp_metadata.get(
            exp_id, {"name": "[deleted]", "status": "deleted"}
        )

    def experiment_exists(self, exp_id: str) -> bool:
        """Check if experiment exists in the loaded data.

        Args:
            exp_id: Experiment ID.

        Returns:
            True if experiment was loaded.
        """
        return exp_id in self._exp_metadata
