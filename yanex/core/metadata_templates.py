"""Parameter-only metadata templates, resolved before experiment creation."""

from collections.abc import Callable
from graphlib import CycleError, TopologicalSorter
from string import Formatter
from typing import TYPE_CHECKING, Any

from ..utils.exceptions import KeyNotFoundError, ValidationError
from ..utils.validation import (
    validate_description,
    validate_experiment_name,
    validate_tags,
)
from .access_resolver import AccessResolver
from .parameter_lookup import get_agreed_parameter

if TYPE_CHECKING:
    from .manager import ExperimentManager

_FORMATTER = Formatter()


class UpstreamParameterCache:
    """Invocation-scoped snapshots shared across sweep combinations.

    Cache only per-experiment data, never the union across combinations. Each
    lookup still resolves keys against precisely its selected upstream graph.
    """

    def __init__(self, manager: "ExperimentManager") -> None:
        self.manager = manager
        self.dependencies: dict[str, list[str]] = {}
        self.configs: dict[str, dict[str, Any]] = {}

    def lookup(self, dependencies: dict[str, str]) -> Callable[[str], Any]:
        """Return a lazy lookup for resolved, validated dependency IDs."""
        roots = tuple(dependencies.values())
        configs = None

        def lookup(key: str) -> Any:
            nonlocal configs
            if configs is None:
                graph = {}
                pending = list(roots)
                while pending:
                    exp_id = pending.pop()
                    if exp_id in graph:
                        continue
                    if exp_id not in self.dependencies:
                        data = (
                            self.manager.storage.dependency_storage.load_dependencies(
                                exp_id, include_archived=True, strict=True
                            )
                        )
                        self.dependencies[exp_id] = list(data["dependencies"].values())
                    graph[exp_id] = self.dependencies[exp_id]
                    pending.extend(graph[exp_id])
                try:
                    ids = list(TopologicalSorter(graph).static_order())
                except CycleError as exc:
                    raise ValidationError(
                        f"Circular upstream dependencies: {exc}"
                    ) from exc
                for exp_id in ids:
                    if exp_id not in self.configs:
                        self.configs[exp_id] = self.manager.storage.load_config(
                            exp_id, include_archived=True
                        )
                configs = {exp_id: self.configs[exp_id] for exp_id in sorted(ids)}
            return get_agreed_parameter(key, configs)

        return lookup


def upstream_parameter_lookup(
    manager: "ExperimentManager", dependencies: dict[str, str]
) -> Callable[[str], Any]:
    """Create an isolated lookup for a single experiment's ancestors."""
    return UpstreamParameterCache(manager).lookup(dependencies)


def has_placeholders(template: str) -> bool:
    """Distinguish replacement fields from escaped literal braces."""
    try:
        return any(field is not None for _, field, _, _ in _FORMATTER.parse(template))
    except ValueError as exc:
        raise ValidationError(f"Invalid metadata template: {exc}") from exc


def render_template(
    template: str,
    params: dict[str, Any],
    upstream_lookup: Callable[[str], Any] | None = None,
) -> str:
    """Render once, using parameter paths rather than Python expressions."""
    resolver = AccessResolver(params=params)
    parts = []
    try:
        for literal, key, spec, conversion in _FORMATTER.parse(template):
            parts.append(literal)
            if key is None:
                continue
            if not key or any(c in key for c in "[]():"):
                raise ValueError("Expected a parameter name or dotted parameter path")
            if conversion is not None:
                raise ValueError("Conversion flags are not supported")
            if "{" in spec or "}" in spec:
                raise ValueError("Nested format specifications are not supported")
            try:
                value = resolver.get_value(resolver.resolve(key, scope="param"))
            except KeyNotFoundError:
                if upstream_lookup is None:
                    raise
                value = upstream_lookup(key)
            parts.append(format(value, spec))
        return "".join(parts)
    except Exception as exc:
        raise ValidationError(f"Cannot render template {template!r}: {exc}") from exc


def render_metadata(
    name: str | None,
    tags: list[str],
    description: str | None,
    params: dict[str, Any],
    upstream_lookup: Callable[[str], Any] | None = None,
) -> tuple[str | None, list[str], str | None]:
    """Render and validate all user-facing metadata for a concrete configuration."""
    rendered_name = (
        render_template(name, params, upstream_lookup) if name is not None else None
    )
    rendered_tags = [render_template(tag, params, upstream_lookup) for tag in tags]
    rendered_description = (
        render_template(description, params, upstream_lookup)
        if description is not None
        else None
    )
    if rendered_name is not None:
        rendered_name = validate_experiment_name(rendered_name)
    if any(not tag for tag in rendered_tags):
        raise ValidationError("Tags cannot be empty")
    rendered_tags = validate_tags(rendered_tags)
    if rendered_description is not None:
        validate_description(rendered_description)
    return rendered_name, rendered_tags, rendered_description
