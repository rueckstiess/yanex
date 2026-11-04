"""Parameter-only metadata templates, resolved before experiment creation."""

from string import Formatter
from typing import Any

from ..utils.exceptions import ValidationError
from ..utils.validation import (
    validate_description,
    validate_experiment_name,
    validate_tags,
)
from .access_resolver import AccessResolver

_FORMATTER = Formatter()


def has_placeholders(template: str) -> bool:
    """Distinguish replacement fields from escaped literal braces."""
    try:
        return any(field is not None for _, field, _, _ in _FORMATTER.parse(template))
    except ValueError as exc:
        raise ValidationError(f"Invalid metadata template: {exc}") from exc


def render_template(template: str, params: dict[str, Any]) -> str:
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
            value = resolver.get_value(resolver.resolve(key, scope="param"))
            parts.append(format(value, spec))
        return "".join(parts)
    except Exception as exc:
        raise ValidationError(f"Cannot render template {template!r}: {exc}") from exc


def render_metadata(
    name: str | None, tags: list[str], description: str | None, params: dict[str, Any]
) -> tuple[str | None, list[str], str | None]:
    """Render and validate all user-facing metadata for a concrete configuration."""
    rendered_name = render_template(name, params) if name is not None else None
    rendered_tags = [render_template(tag, params) for tag in tags]
    rendered_description = (
        render_template(description, params) if description is not None else None
    )
    if rendered_name is not None:
        rendered_name = validate_experiment_name(rendered_name)
    if any(not tag for tag in rendered_tags):
        raise ValidationError("Tags cannot be empty")
    rendered_tags = validate_tags(rendered_tags)
    if rendered_description is not None:
        validate_description(rendered_description)
    return rendered_name, rendered_tags, rendered_description
