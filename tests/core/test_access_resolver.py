"""Tests for the AccessResolver utility."""

import pytest

from yanex.core.access_resolver import (
    AccessResolver,
    build_canonical_key,
    flatten_dict,
    parse_canonical_key,
)
from yanex.utils.exceptions import (
    AmbiguousKeyError,
    InvalidGroupError,
    KeyNotFoundError,
)


class TestFlattenDict:
    """Tests for flatten_dict function."""

    def test_flat_dict(self) -> None:
        """Flat dict returns same keys."""
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self) -> None:
        """Nested dict flattens with dot notation."""
        d = {"model": {"lr": 0.01, "layers": 5}}
        assert flatten_dict(d) == {"model.lr": 0.01, "model.layers": 5}

    def test_deeply_nested(self) -> None:
        """Deeply nested dict flattens correctly."""
        d = {"a": {"b": {"c": 1}}}
        assert flatten_dict(d) == {"a.b.c": 1}

    def test_mixed_nesting(self) -> None:
        """Mixed flat and nested keys work."""
        d = {"x": 1, "y": {"z": 2}}
        assert flatten_dict(d) == {"x": 1, "y.z": 2}

    def test_empty_dict(self) -> None:
        """Empty dict returns empty dict."""
        assert flatten_dict({}) == {}


class TestParseCanonicalKey:
    """Tests for parse_canonical_key function."""

    def test_param_prefix(self) -> None:
        """Parses param: prefix correctly."""
        assert parse_canonical_key("param:model.lr") == ("param", "model.lr")

    def test_metric_prefix(self) -> None:
        """Parses metric: prefix correctly."""
        assert parse_canonical_key("metric:train.loss") == ("metric", "train.loss")

    def test_meta_prefix(self) -> None:
        """Parses meta: prefix correctly."""
        assert parse_canonical_key("meta:status") == ("meta", "status")

    def test_no_prefix(self) -> None:
        """No prefix returns None for group."""
        assert parse_canonical_key("model.lr") == (None, "model.lr")

    def test_colon_in_value(self) -> None:
        """Colon in value (not prefix) handled correctly."""
        # "unknown:" is not a valid group, so treated as no prefix
        assert parse_canonical_key("unknown:value") == (None, "unknown:value")


class TestBuildCanonicalKey:
    """Tests for build_canonical_key function."""

    def test_build_param(self) -> None:
        """Builds param key correctly."""
        assert build_canonical_key("param", "model.lr") == "param:model.lr"

    def test_build_metric(self) -> None:
        """Builds metric key correctly."""
        assert build_canonical_key("metric", "accuracy") == "metric:accuracy"


class TestAccessResolver:
    """Tests for AccessResolver class."""

    @pytest.fixture
    def sample_resolver(self) -> AccessResolver:
        """Create a resolver with sample data."""
        return AccessResolver(
            params={
                "epochs": 100,
                "advisor": {
                    "head": {"lr": 0.01, "layers": 3},
                    "backbone": {"lr": 0.001, "layers": 10},
                },
            },
            metrics={
                "train": {"loss": 0.5, "accuracy": 0.9},
                "test": {"loss": 0.6, "accuracy": 0.85},
            },
            meta={
                "id": "abc123",
                "name": "test-experiment",
                "status": "completed",
                "git": {"branch": "main", "commit_hash": "abc123"},
            },
        )

    def test_get_all_keys(self, sample_resolver: AccessResolver) -> None:
        """Gets all canonical keys."""
        keys = sample_resolver.get_all_keys()
        assert "param:epochs" in keys
        assert "param:advisor.head.lr" in keys
        assert "metric:train.loss" in keys
        assert "meta:status" in keys

    def test_get_all_keys_scoped(self, sample_resolver: AccessResolver) -> None:
        """Gets keys for specific scope."""
        param_keys = sample_resolver.get_all_keys(scope="param")
        assert all(k.startswith("param:") for k in param_keys)
        assert "param:epochs" in param_keys

        metric_keys = sample_resolver.get_all_keys(scope="metric")
        assert all(k.startswith("metric:") for k in metric_keys)

    def test_get_paths(self, sample_resolver: AccessResolver) -> None:
        """Gets paths without group prefix."""
        paths = sample_resolver.get_paths(scope="param")
        assert "epochs" in paths
        assert "advisor.head.lr" in paths
        # Should not have prefix
        assert not any(p.startswith("param:") for p in paths)

    def test_resolve_exact_match(self, sample_resolver: AccessResolver) -> None:
        """Resolves exact path match."""
        assert sample_resolver.resolve("epochs") == "param:epochs"
        assert sample_resolver.resolve("status") == "meta:status"

    def test_resolve_suffix_match(self, sample_resolver: AccessResolver) -> None:
        """Resolves suffix match when unambiguous."""
        # "head.lr" should match "advisor.head.lr"
        assert sample_resolver.resolve("head.lr") == "param:advisor.head.lr"

    def test_resolve_ambiguous(self, sample_resolver: AccessResolver) -> None:
        """Raises AmbiguousKeyError for ambiguous keys."""
        # "lr" matches both advisor.head.lr and advisor.backbone.lr
        with pytest.raises(AmbiguousKeyError) as exc_info:
            sample_resolver.resolve("lr")

        assert exc_info.value.key == "lr"
        assert "param:advisor.head.lr" in exc_info.value.matches
        assert "param:advisor.backbone.lr" in exc_info.value.matches

    def test_resolve_ambiguous_across_groups(
        self, sample_resolver: AccessResolver
    ) -> None:
        """Raises AmbiguousKeyError for keys matching across groups."""
        # "accuracy" matches both train.accuracy and test.accuracy in metrics
        with pytest.raises(AmbiguousKeyError) as exc_info:
            sample_resolver.resolve("accuracy")

        assert "metric:train.accuracy" in exc_info.value.matches
        assert "metric:test.accuracy" in exc_info.value.matches

    def test_resolve_with_scope(self, sample_resolver: AccessResolver) -> None:
        """Resolves within specific scope."""
        # "train.accuracy" in metric scope
        result = sample_resolver.resolve("train.accuracy", scope="metric")
        assert result == "metric:train.accuracy"

    def test_resolve_not_found(self, sample_resolver: AccessResolver) -> None:
        """Raises KeyNotFoundError for unknown keys."""
        with pytest.raises(KeyNotFoundError):
            sample_resolver.resolve("nonexistent")

    def test_resolve_with_group_prefix(self, sample_resolver: AccessResolver) -> None:
        """Resolves keys with explicit group prefix."""
        assert sample_resolver.resolve("param:epochs") == "param:epochs"
        assert sample_resolver.resolve("meta:status") == "meta:status"

    def test_resolve_wrong_group_prefix(self, sample_resolver: AccessResolver) -> None:
        """Raises InvalidGroupError for wrong group prefix in scope."""
        with pytest.raises(InvalidGroupError) as exc_info:
            sample_resolver.resolve("metric:accuracy", scope="param")

        assert exc_info.value.expected_group == "param"
        assert exc_info.value.actual_group == "metric"

    def test_resolve_pattern_wildcard(self, sample_resolver: AccessResolver) -> None:
        """Resolves glob pattern with wildcard."""
        matches = sample_resolver.resolve_pattern("*.lr")
        assert "param:advisor.head.lr" in matches
        assert "param:advisor.backbone.lr" in matches

    def test_resolve_pattern_prefix(self, sample_resolver: AccessResolver) -> None:
        """Resolves glob pattern matching prefix."""
        matches = sample_resolver.resolve_pattern("train.*", scope="metric")
        assert "metric:train.loss" in matches
        assert "metric:train.accuracy" in matches
        assert "metric:test.loss" not in matches

    def test_resolve_pattern_nested(self, sample_resolver: AccessResolver) -> None:
        """Resolves nested glob patterns."""
        matches = sample_resolver.resolve_pattern("advisor.*.lr")
        assert "param:advisor.head.lr" in matches
        assert "param:advisor.backbone.lr" in matches

    def test_resolve_pattern_with_scope(self, sample_resolver: AccessResolver) -> None:
        """Pattern respects scope."""
        matches = sample_resolver.resolve_pattern("*", scope="meta")
        assert all(k.startswith("meta:") for k in matches)

    def test_is_pattern(self, sample_resolver: AccessResolver) -> None:
        """Detects pattern characters."""
        assert sample_resolver.is_pattern("*.lr")
        assert sample_resolver.is_pattern("train.*")
        assert sample_resolver.is_pattern("model[12].lr")
        assert sample_resolver.is_pattern("model?.lr")
        assert not sample_resolver.is_pattern("model.lr")
        assert not sample_resolver.is_pattern("epochs")

    def test_resolve_or_pattern_key(self, sample_resolver: AccessResolver) -> None:
        """resolve_or_pattern handles regular keys."""
        result = sample_resolver.resolve_or_pattern("epochs")
        assert result == ["param:epochs"]

    def test_resolve_or_pattern_pattern(self, sample_resolver: AccessResolver) -> None:
        """resolve_or_pattern handles patterns."""
        result = sample_resolver.resolve_or_pattern("*.lr")
        assert "param:advisor.head.lr" in result
        assert "param:advisor.backbone.lr" in result

    def test_resolve_list(self, sample_resolver: AccessResolver) -> None:
        """Resolves list of keys and patterns."""
        result = sample_resolver.resolve_list(["epochs", "train.*"], scope=None)
        assert "param:epochs" in result
        assert "metric:train.loss" in result
        assert "metric:train.accuracy" in result

    def test_get_value(self, sample_resolver: AccessResolver) -> None:
        """Gets values for canonical keys."""
        assert sample_resolver.get_value("param:epochs") == 100
        assert sample_resolver.get_value("param:advisor.head.lr") == 0.01
        assert sample_resolver.get_value("meta:status") == "completed"
        assert sample_resolver.get_value("metric:train.loss") == 0.5

    def test_get_value_not_found(self, sample_resolver: AccessResolver) -> None:
        """Raises KeyNotFoundError for unknown canonical key."""
        with pytest.raises(KeyNotFoundError):
            sample_resolver.get_value("param:nonexistent")

    def test_empty_resolver(self) -> None:
        """Resolver with no data works."""
        resolver = AccessResolver()
        assert resolver.get_all_keys() == []
        with pytest.raises(KeyNotFoundError):
            resolver.resolve("anything")

    def test_validate_group_correct(self, sample_resolver: AccessResolver) -> None:
        """validate_group passes for correct group."""
        # Should not raise
        sample_resolver.validate_group("param:epochs", "param")

    def test_validate_group_wrong(self, sample_resolver: AccessResolver) -> None:
        """validate_group raises for wrong group."""
        with pytest.raises(InvalidGroupError):
            sample_resolver.validate_group("metric:loss", "param")

    def test_validate_group_no_prefix(self, sample_resolver: AccessResolver) -> None:
        """validate_group passes for keys without prefix."""
        # No prefix means no validation needed
        sample_resolver.validate_group("epochs", "param")
