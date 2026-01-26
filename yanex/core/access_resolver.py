"""
Access resolver for unified parameter, metric, and metadata access.

This module provides the AccessResolver class which resolves unqualified keys
to their canonical group:path format, with support for pattern matching.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Literal

from yanex.utils.exceptions import (
    AmbiguousKeyError,
    InvalidGroupError,
    KeyNotFoundError,
)

# Valid group prefixes
GROUPS = ("param", "metric", "meta")

# Metadata fields that fall under meta: group
META_FIELDS = {
    "id",
    "name",
    "status",
    "description",
    "tags",
    "created_at",
    "started_at",
    "completed_at",
    "script_path",
    "git.branch",
    "git.commit_hash",
    "git.dirty",
}

# Default meta fields shown with "auto" mode
AUTO_META_FIELDS = ["id", "name", "status"]


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary using dot notation.

    Args:
        d: Dictionary to flatten
        prefix: Prefix to prepend to keys

    Returns:
        Flattened dictionary with dot-notation keys
    """
    items: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            items.update(flatten_dict(value, new_key))
        else:
            items[new_key] = value
    return items


def parse_canonical_key(key: str) -> tuple[str | None, str]:
    """Parse a key into (group, path) tuple.

    Args:
        key: Key to parse, with or without group prefix

    Returns:
        Tuple of (group, path). group is None if no prefix.

    Examples:
        >>> parse_canonical_key("param:model.lr")
        ("param", "model.lr")
        >>> parse_canonical_key("model.lr")
        (None, "model.lr")
    """
    for group in GROUPS:
        prefix = f"{group}:"
        if key.startswith(prefix):
            return (group, key[len(prefix) :])
    return (None, key)


def build_canonical_key(group: str, path: str) -> str:
    """Build a canonical key from group and path.

    Args:
        group: Group name (param, metric, meta)
        path: Dot-notation path

    Returns:
        Canonical key like "param:model.lr"
    """
    return f"{group}:{path}"


class AccessResolver:
    """Resolves unqualified keys to canonical group:key format.

    The resolver builds an index of all available canonical keys from
    the provided params, metrics, and meta data, and then resolves
    unqualified keys by matching against leaf keys and partial paths.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize AccessResolver with available data.

        Args:
            params: Parameter dictionary (can be nested)
            metrics: Metrics dictionary (can be nested)
            meta: Metadata dictionary
        """
        self._params = params or {}
        self._metrics = metrics or {}
        self._meta = meta or {}

        # Build index of all canonical keys
        self._canonical_keys: dict[str, str] = {}  # group:path -> group
        self._build_index()

    def _build_index(self) -> None:
        """Build index of all canonical keys from available data."""
        # Flatten and add params
        flat_params = flatten_dict(self._params)
        for path in flat_params:
            self._canonical_keys[build_canonical_key("param", path)] = "param"

        # Flatten and add metrics
        flat_metrics = flatten_dict(self._metrics)
        for path in flat_metrics:
            self._canonical_keys[build_canonical_key("metric", path)] = "metric"

        # Add meta fields (use flat meta dict structure)
        flat_meta = flatten_dict(self._meta)
        for path in flat_meta:
            self._canonical_keys[build_canonical_key("meta", path)] = "meta"

    def get_all_keys(
        self, scope: Literal["param", "metric", "meta"] | None = None
    ) -> list[str]:
        """Get all canonical keys, optionally filtered by scope.

        Args:
            scope: If specified, only return keys from this group

        Returns:
            Sorted list of canonical keys
        """
        if scope:
            return sorted(
                key for key, group in self._canonical_keys.items() if group == scope
            )
        return sorted(self._canonical_keys.keys())

    def get_paths(
        self, scope: Literal["param", "metric", "meta"] | None = None
    ) -> list[str]:
        """Get all paths (without group prefix), optionally filtered by scope.

        Args:
            scope: If specified, only return paths from this group

        Returns:
            Sorted list of paths
        """
        keys = self.get_all_keys(scope)
        return [parse_canonical_key(key)[1] for key in keys]

    def _find_matches(
        self,
        query: str,
        scope: Literal["param", "metric", "meta"] | None = None,
    ) -> list[str]:
        """Find all canonical keys that match a query.

        A query matches if:
        1. It equals the path exactly
        2. It equals the path suffix (e.g., "lr" matches "model.lr")

        Args:
            query: Query string to match
            scope: If specified, only search this group

        Returns:
            List of matching canonical keys
        """
        matches = []

        for canonical_key, group in self._canonical_keys.items():
            if scope and group != scope:
                continue

            _, path = parse_canonical_key(canonical_key)

            # Exact match
            if path == query:
                matches.append(canonical_key)
                continue

            # Suffix match: query matches end of path
            # "lr" matches "model.lr", "head.lr" matches "advisor.head.lr"
            if path.endswith(f".{query}"):
                matches.append(canonical_key)

        return matches

    def resolve(
        self,
        key: str,
        scope: Literal["param", "metric", "meta"] | None = None,
    ) -> str:
        """Resolve a key to its canonical form.

        Args:
            key: The key to resolve (with or without group prefix)
            scope: If specified, only search this group

        Returns:
            Canonical key like "param:model.lr"

        Raises:
            AmbiguousKeyError: If key matches multiple canonical keys
            InvalidGroupError: If key has wrong group prefix for scope
            KeyNotFoundError: If key cannot be resolved
        """
        # Check if already a canonical key
        parsed_group, path = parse_canonical_key(key)

        if parsed_group:
            # Key has a group prefix
            canonical = build_canonical_key(parsed_group, path)

            # Validate scope if specified
            if scope and parsed_group != scope:
                raise InvalidGroupError(key, scope, parsed_group)

            # Check if the canonical key exists
            if canonical in self._canonical_keys:
                return canonical

            # Try to resolve the path within the specified group
            matches = self._find_matches(path, scope=parsed_group)
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                raise AmbiguousKeyError(key, matches)
            else:
                raise KeyNotFoundError(key, parsed_group)

        # Key has no group prefix - resolve across groups
        matches = self._find_matches(path, scope=scope)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise AmbiguousKeyError(key, matches)
        else:
            raise KeyNotFoundError(key, scope)

    def resolve_pattern(
        self,
        pattern: str,
        scope: Literal["param", "metric", "meta"] | None = None,
    ) -> list[str]:
        """Resolve a glob pattern to matching canonical keys.

        Supports glob patterns like:
        - "*.lr" - all keys ending in .lr
        - "train.*" - all keys starting with train.
        - "model.*.lr" - nested patterns

        Args:
            pattern: Glob pattern to match
            scope: If specified, only search this group

        Returns:
            Sorted list of matching canonical keys
        """
        # Check if pattern has group prefix
        parsed_group, path_pattern = parse_canonical_key(pattern)

        if parsed_group:
            # Pattern has explicit group prefix
            if scope and parsed_group != scope:
                raise InvalidGroupError(pattern, scope, parsed_group)
            search_scope = parsed_group
        else:
            search_scope = scope

        matches = []
        for canonical_key, group in self._canonical_keys.items():
            if search_scope and group != search_scope:
                continue

            _, path = parse_canonical_key(canonical_key)

            # Use fnmatch for glob pattern matching
            if fnmatch.fnmatch(path, path_pattern):
                matches.append(canonical_key)

        return sorted(matches)

    def validate_group(self, key: str, expected_group: str) -> None:
        """Validate that a prefixed key belongs to expected group.

        Args:
            key: Key with group prefix
            expected_group: Expected group name

        Raises:
            InvalidGroupError: If key belongs to different group
        """
        parsed_group, _ = parse_canonical_key(key)

        if parsed_group and parsed_group != expected_group:
            raise InvalidGroupError(key, expected_group, parsed_group)

    def is_pattern(self, value: str) -> bool:
        """Check if a value contains glob pattern characters.

        Args:
            value: Value to check

        Returns:
            True if value contains *, ?, or [] patterns
        """
        return any(c in value for c in "*?[]")

    def resolve_or_pattern(
        self,
        value: str,
        scope: Literal["param", "metric", "meta"] | None = None,
    ) -> list[str]:
        """Resolve a value as either a single key or a pattern.

        If the value contains glob characters, treat it as a pattern.
        Otherwise, try to resolve it as a single key.

        Args:
            value: Value to resolve
            scope: If specified, only search this group

        Returns:
            List of matching canonical keys (single-element for non-patterns)

        Raises:
            AmbiguousKeyError: If non-pattern key is ambiguous
            KeyNotFoundError: If non-pattern key is not found
        """
        if self.is_pattern(value):
            return self.resolve_pattern(value, scope)
        else:
            return [self.resolve(value, scope)]

    def resolve_list(
        self,
        values: list[str],
        scope: Literal["param", "metric", "meta"] | None = None,
    ) -> list[str]:
        """Resolve a list of values (keys or patterns) to canonical keys.

        Args:
            values: List of keys or patterns
            scope: If specified, only search this group

        Returns:
            Sorted list of unique canonical keys
        """
        result = set()
        for value in values:
            matches = self.resolve_or_pattern(value, scope)
            result.update(matches)
        return sorted(result)

    def get_value(self, canonical_key: str) -> Any:
        """Get the value for a canonical key.

        Args:
            canonical_key: Canonical key like "param:model.lr"

        Returns:
            Value from the underlying data

        Raises:
            KeyNotFoundError: If key is not found
        """
        group, path = parse_canonical_key(canonical_key)

        if group == "param":
            data = self._params
        elif group == "metric":
            data = self._metrics
        elif group == "meta":
            data = self._meta
        else:
            raise KeyNotFoundError(canonical_key)

        # Navigate nested path
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyNotFoundError(canonical_key)

        return value
