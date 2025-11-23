"""Tests for parameter tracking utilities."""

import pytest

from yanex.core.param_tracking import (
    _get_value_by_path,
    _set_value_by_path,
    deduplicate_paths,
    extract_accessed_params,
)
from yanex.core.tracked_dict import TrackedDict


class TestGetValueByPath:
    """Test _get_value_by_path helper."""

    def test_simple_path(self):
        """Test getting value with simple path."""
        data = {"a": 1, "b": 2}
        assert _get_value_by_path(data, "a") == 1
        assert _get_value_by_path(data, "b") == 2

    def test_nested_path(self):
        """Test getting value with nested path."""
        data = {"model": {"train": {"lr": 0.01, "epochs": 20}}}
        assert _get_value_by_path(data, "model.train.lr") == 0.01
        assert _get_value_by_path(data, "model.train.epochs") == 20

    def test_nested_dict_value(self):
        """Test getting nested dict as value."""
        data = {"model": {"train": {"lr": 0.01}}}
        result = _get_value_by_path(data, "model.train")
        assert result == {"lr": 0.01}

    def test_missing_key_raises_error(self):
        """Test that missing key raises KeyError."""
        data = {"a": 1}
        with pytest.raises(KeyError):
            _get_value_by_path(data, "b")

    def test_missing_nested_key_raises_error(self):
        """Test that missing nested key raises KeyError."""
        data = {"model": {"train": {"lr": 0.01}}}
        with pytest.raises(KeyError):
            _get_value_by_path(data, "model.test.lr")

    def test_deeply_nested_path(self):
        """Test deeply nested path."""
        data = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        assert _get_value_by_path(data, "a.b.c.d.e") == 42


class TestSetValueByPath:
    """Test _set_value_by_path helper."""

    def test_simple_path(self):
        """Test setting value with simple path."""
        data = {}
        _set_value_by_path(data, "a", 1)
        assert data == {"a": 1}

    def test_nested_path_creates_structure(self):
        """Test that nested path creates intermediate dicts."""
        data = {}
        _set_value_by_path(data, "model.train.lr", 0.01)
        assert data == {"model": {"train": {"lr": 0.01}}}

    def test_multiple_values_same_parent(self):
        """Test setting multiple values under same parent."""
        data = {}
        _set_value_by_path(data, "model.train.lr", 0.01)
        _set_value_by_path(data, "model.train.epochs", 20)
        assert data == {"model": {"train": {"lr": 0.01, "epochs": 20}}}

    def test_multiple_top_level_keys(self):
        """Test setting multiple top-level keys."""
        data = {}
        _set_value_by_path(data, "model.lr", 0.01)
        _set_value_by_path(data, "seed", 42)
        assert data == {"model": {"lr": 0.01}, "seed": 42}

    def test_deeply_nested_path(self):
        """Test deeply nested path creation."""
        data = {}
        _set_value_by_path(data, "a.b.c.d.e", 42)
        assert data == {"a": {"b": {"c": {"d": {"e": 42}}}}}

    def test_tracked_dict_converted_to_dict(self):
        """Test that TrackedDict values are converted to plain dict."""
        data = {}
        tracked = TrackedDict({"lr": 0.01, "epochs": 20})
        _set_value_by_path(data, "model.train", tracked)

        # Should be plain dict, not TrackedDict
        assert isinstance(data["model"]["train"], dict)
        assert not isinstance(data["model"]["train"], TrackedDict)
        assert data["model"]["train"] == {"lr": 0.01, "epochs": 20}


class TestExtractAccessedParams:
    """Test extract_accessed_params function."""

    def test_no_accesses_returns_empty(self):
        """Test that no accesses returns empty dict."""
        tracked = TrackedDict({"a": 1, "b": 2})
        result = extract_accessed_params(tracked)
        assert result == {}

    def test_single_top_level_access(self):
        """Test extracting single top-level param."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = tracked["a"]
        result = extract_accessed_params(tracked)
        assert result == {"a": 1}

    def test_multiple_top_level_accesses(self):
        """Test extracting multiple top-level params."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = tracked["a"]
        _ = tracked["c"]
        result = extract_accessed_params(tracked)
        assert result == {"a": 1, "c": 3}

    def test_nested_access(self):
        """Test extracting nested params."""
        tracked = TrackedDict({"model": {"lr": 0.01, "layers": 5}, "seed": 42})
        _ = tracked["model"]["lr"]
        result = extract_accessed_params(tracked)

        # Should include only accessed nested value
        assert result == {"model": {"lr": 0.01}}

    def test_partial_nested_access(self):
        """Test extracting some but not all nested params."""
        tracked = TrackedDict(
            {
                "model": {"train": {"lr": 0.01, "epochs": 20}, "arch": {"layers": 5}},
                "seed": 42,
            }
        )

        _ = tracked["model"]["train"]["lr"]
        _ = tracked["seed"]
        result = extract_accessed_params(tracked)

        assert result == {"model": {"train": {"lr": 0.01}}, "seed": 42}

    def test_whole_nested_dict_access(self):
        """Test accessing entire nested dict preserves structure."""
        tracked = TrackedDict({"model": {"lr": 0.01, "layers": 5}})

        # Access the whole nested dict
        model = tracked["model"]
        # Then access a value within it
        _ = model["lr"]

        result = extract_accessed_params(tracked)
        # Should have model.lr but not model.layers
        assert result == {"model": {"lr": 0.01}}

    def test_deeply_nested_access(self):
        """Test extracting deeply nested params."""
        tracked = TrackedDict({"a": {"b": {"c": {"d": {"e": 42}}}}})
        _ = tracked["a"]["b"]["c"]["d"]["e"]
        result = extract_accessed_params(tracked)

        assert result == {"a": {"b": {"c": {"d": {"e": 42}}}}}

    def test_mixed_depth_accesses(self):
        """Test extracting params at different nesting levels."""
        tracked = TrackedDict(
            {
                "top": 1,
                "nested": {"mid": 2},
                "deep": {"path": {"to": {"value": 3}}},
            }
        )

        _ = tracked["top"]
        _ = tracked["nested"]["mid"]
        _ = tracked["deep"]["path"]["to"]["value"]

        result = extract_accessed_params(tracked)
        assert result == {
            "top": 1,
            "nested": {"mid": 2},
            "deep": {"path": {"to": {"value": 3}}},
        }

    def test_iteration_marks_all(self):
        """Test that iteration extracts all params."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = list(tracked.keys())  # Marks all as accessed

        result = extract_accessed_params(tracked)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_parent_and_child_both_accessed(self):
        """Test when both parent dict and child value are accessed."""
        tracked = TrackedDict({"model": {"lr": 0.01, "layers": 5}})

        # Access parent first
        _ = tracked["model"]
        # Then access specific child
        _ = tracked["model"]["lr"]

        result = extract_accessed_params(tracked)
        # Should only include the specific accessed child
        assert result == {"model": {"lr": 0.01}}


class TestDeduplicatePaths:
    """Test deduplicate_paths function."""

    def test_empty_set(self):
        """Test empty set returns empty set."""
        assert deduplicate_paths(set()) == set()

    def test_no_redundancy(self):
        """Test paths with no parent-child relationships."""
        paths = {"a", "b", "c"}
        result = deduplicate_paths(paths)
        assert result == {"a", "b", "c"}

    def test_removes_parent_when_child_exists(self):
        """Test that parent paths are removed when children exist."""
        paths = {"model", "model.lr"}
        result = deduplicate_paths(paths)
        assert result == {"model.lr"}

    def test_multiple_children_same_parent(self):
        """Test multiple children of same parent."""
        paths = {"model", "model.lr", "model.layers"}
        result = deduplicate_paths(paths)
        assert result == {"model.lr", "model.layers"}

    def test_deep_hierarchy(self):
        """Test deep parent-child hierarchy."""
        paths = {"a", "a.b", "a.b.c", "a.b.c.d"}
        result = deduplicate_paths(paths)
        assert result == {"a.b.c.d"}

    def test_separate_hierarchies(self):
        """Test multiple separate hierarchies."""
        paths = {"model", "model.lr", "data", "data.path"}
        result = deduplicate_paths(paths)
        assert result == {"model.lr", "data.path"}

    def test_mixed_depths(self):
        """Test mixed depth accesses in different branches."""
        paths = {"a", "a.b.c", "x.y", "p.q.r.s"}
        result = deduplicate_paths(paths)
        assert result == {"a.b.c", "x.y", "p.q.r.s"}

    def test_preserves_leaf_nodes(self):
        """Test that only leaf nodes are preserved."""
        paths = {"model", "model.train", "model.train.lr", "seed"}
        result = deduplicate_paths(paths)
        assert result == {"model.train.lr", "seed"}
