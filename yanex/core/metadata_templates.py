"""Parameter-only metadata templates, resolved before experiment creation."""

from collections.abc import Callable
from string import Formatter
from typing import TYPE_CHECKING, Any

from ..utils.exceptions import KeyNotFoundError, ValidationError
from ..utils.validation import (
    validate_description,
    validate_experiment_name,
    validate_tags,
)
from .access_resolver import AccessResolver
from .dependencies import DependencyResolver
from .parameter_lookup import get_agreed_parameter

if TYPE_CHECKING:
    from .manager import ExperimentManager

_FORMATTER = Formatter()


def upstream_parameter_lookup(
    manager: "ExperimentManager", dependencies: dict[str, str]
) -> Callable[[str], Any]:
    """Lazily load the selected ancestors once, including archived experiments.

    Dependencies must already contain resolved, validated IDs. Traversal follows
    dependency edges only, and shared ancestors are deduplicated by ID.
    """
    configs = None

    def lookup(key: str) -> Any:
        nonlocal configs
        if configs is None:
            resolver = DependencyResolver(manager)
            ids = set()
            for dep_id in dependencies.values():
                ids.update(
                    resolver.get_transitive_dependencies(
                        dep_id, include_self=True, include_archived=True
                    )
                )
            configs = {
                exp_id: manager.storage.load_config(exp_id, include_archived=True)
                for exp_id in sorted(ids)
            }
        return get_agreed_parameter(key, configs)

    return lookup


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
