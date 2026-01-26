"""Tests for TrackedDict class."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from yanex.core.tracked_dict import _MISSING, TrackedDict
from yanex.utils.exceptions import ParameterConflictError


class TestTrackedDictBasics:
    """Test basic TrackedDict functionality."""

    def test_init_empty(self):
        """Test initialization with no data."""
        tracked = TrackedDict()
        assert len(tracked) == 0
        assert tracked.get_accessed_paths() == set()

    def test_init_with_data(self):
        """Test initialization with data."""
        data = {"a": 1, "b": 2}
        tracked = TrackedDict(data)
        assert len(tracked) == 2
        assert tracked.get_accessed_paths() == set()  # Not accessed yet

    def test_getitem_tracks_access(self):
        """Test that __getitem__ tracks access."""
        tracked = TrackedDict({"a": 1, "b": 2})
        _ = tracked["a"]
        assert "a" in tracked.get_accessed_paths()
        assert "b" not in tracked.get_accessed_paths()

    def test_get_tracks_access_when_key_exists(self):
        """Test that get() tracks access when key exists."""
        tracked = TrackedDict({"a": 1, "b": 2})
        value = tracked.get("a")
        assert value == 1
        assert "a" in tracked.get_accessed_paths()

    def test_get_does_not_track_when_key_missing(self):
        """Test that get() doesn't track when key doesn't exist."""
        tracked = TrackedDict({"a": 1})
        value = tracked.get("b", default=999)
        assert value == 999
        assert "b" not in tracked.get_accessed_paths()

    def test_contains_does_not_track(self):
        """Test that 'in' operator doesn't track access."""
        tracked = TrackedDict({"a": 1})
        assert "a" in tracked
        assert tracked.get_accessed_paths() == set()

    def test_len_does_not_track(self):
        """Test that len() doesn't track access."""
        tracked = TrackedDict({"a": 1, "b": 2})
        assert len(tracked) == 2
        assert tracked.get_accessed_paths() == set()


class TestTrackedDictNested:
    """Test nested dictionary tracking."""

    def test_nested_dict_wrapping(self):
        """Test that nested dicts are automatically wrapped."""
        data = {"model": {"lr": 0.01, "layers": 5}}
        tracked = TrackedDict(data)

        # Access nested dict
        model = tracked["model"]
        assert isinstance(model, TrackedDict)
        assert "model" in tracked.get_accessed_paths()

    def test_nested_access_tracking(self):
        """Test tracking of nested access paths."""
        data = {"model": {"train": {"lr": 0.01, "epochs": 20}}}
        tracked = TrackedDict(data)

        # Access nested value
        lr = tracked["model"]["train"]["lr"]
        assert lr == 0.01

        # Check all paths are tracked
        accessed = tracked.get_accessed_paths()
        assert "model" in accessed
        assert "model.train" in accessed
        assert "model.train.lr" in accessed

    def test_nested_dict_shares_tracking(self):
        """Test that nested dicts share the same tracking set."""
        data = {"model": {"lr": 0.01}, "seed": 42}
        tracked = TrackedDict(data)

        # Access nested value
        _ = tracked["model"]["lr"]
        # Access top-level value
        _ = tracked["seed"]

        # Both should be in the same set
        accessed = tracked.get_accessed_paths()
        assert "model.lr" in accessed
        assert "seed" in accessed
        assert len(accessed) == 3  # "model", "model.lr", "seed"

    def test_deeply_nested_tracking(self):
        """Test deeply nested structure tracking."""
        data = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        tracked = TrackedDict(data)

        value = tracked["a"]["b"]["c"]["d"]["e"]
        assert value == 42

        accessed = tracked.get_accessed_paths()
        assert "a" in accessed
        assert "a.b" in accessed
        assert "a.b.c" in accessed
        assert "a.b.c.d" in accessed
        assert "a.b.c.d.e" in accessed


class TestTrackedDictIterations:
    """Test iteration tracking."""

    def test_keys_marks_all_accessed(self):
        """Test that keys() marks all keys as accessed."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = list(tracked.keys())

        accessed = tracked.get_accessed_paths()
        assert "a" in accessed
        assert "b" in accessed
        assert "c" in accessed

    def test_values_marks_all_accessed(self):
        """Test that values() marks all keys as accessed."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = list(tracked.values())

        accessed = tracked.get_accessed_paths()
        assert "a" in accessed
        assert "b" in accessed
        assert "c" in accessed

    def test_items_marks_all_accessed(self):
        """Test that items() marks all keys as accessed."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = list(tracked.items())

        accessed = tracked.get_accessed_paths()
        assert "a" in accessed
        assert "b" in accessed
        assert "c" in accessed

    def test_iteration_over_dict(self):
        """Test iteration over dict marks all accessed."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        for _key in tracked:
            pass  # Just iterate

        # Iteration uses keys(), which marks all
        accessed = tracked.get_accessed_paths()
        assert len(accessed) == 3


class TestTrackedDictThreadSafety:
    """Test thread safety of access tracking."""

    def test_concurrent_access(self):
        """Test concurrent access from multiple threads."""
        data = {f"key_{i}": i for i in range(100)}
        tracked = TrackedDict(data)

        def access_keys(start, end):
            for i in range(start, end):
                _ = tracked[f"key_{i}"]

        # Access different keys from multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(access_keys, i * 10, (i + 1) * 10)
                futures.append(future)

            for future in futures:
                future.result()

        # All 100 keys should be tracked
        accessed = tracked.get_accessed_paths()
        assert len(accessed) == 100
        for i in range(100):
            assert f"key_{i}" in accessed

    def test_concurrent_nested_access(self):
        """Test concurrent access to nested dicts."""
        data = {"model": {f"param_{i}": i for i in range(50)}}
        tracked = TrackedDict(data)

        def access_nested(start, end):
            for i in range(start, end):
                _ = tracked["model"][f"param_{i}"]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(access_nested, i * 10, (i + 1) * 10)
                futures.append(future)

            for future in futures:
                future.result()

        accessed = tracked.get_accessed_paths()
        # Should have "model" + 50 "model.param_X" entries
        assert "model" in accessed
        assert len(accessed) == 51


class TestTrackedDictUtilities:
    """Test utility methods."""

    def test_clear_accessed_paths(self):
        """Test clearing tracked accesses."""
        tracked = TrackedDict({"a": 1, "b": 2})
        _ = tracked["a"]
        assert len(tracked.get_accessed_paths()) > 0

        tracked.clear_accessed_paths()
        assert len(tracked.get_accessed_paths()) == 0

    def test_get_accessed_paths_returns_copy(self):
        """Test that get_accessed_paths returns a copy."""
        tracked = TrackedDict({"a": 1})
        _ = tracked["a"]

        paths1 = tracked.get_accessed_paths()
        paths2 = tracked.get_accessed_paths()

        # Should be equal but not the same object
        assert paths1 == paths2
        assert paths1 is not paths2

        # Modifying returned set shouldn't affect tracking
        paths1.add("fake")
        assert "fake" not in tracked.get_accessed_paths()


class TestTrackedDictEdgeCases:
    """Test edge cases and limitations."""

    def test_copying_creates_untracked_dict(self):
        """Test that dict() conversion loses tracking but iteration marks all accessed."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})

        # Converting to dict() uses iteration, which marks all as accessed
        copied = dict(tracked)

        # All keys should be marked accessed due to iteration
        accessed_before_copy_access = tracked.get_accessed_paths()
        assert "a" in accessed_before_copy_access
        assert "b" in accessed_before_copy_access
        assert "c" in accessed_before_copy_access

        # But the copied dict is plain dict, not tracked
        assert not isinstance(copied, TrackedDict)
        assert isinstance(copied, dict)

        # Accessing copied dict doesn't add any NEW tracking
        _ = copied["a"]  # Doesn't affect tracking
        assert tracked.get_accessed_paths() == accessed_before_copy_access

    def test_none_values(self):
        """Test tracking with None values."""
        tracked = TrackedDict({"a": None, "b": 0, "c": ""})
        assert tracked["a"] is None
        assert tracked["b"] == 0
        assert tracked["c"] == ""

        accessed = tracked.get_accessed_paths()
        assert "a" in accessed
        assert "b" in accessed
        assert "c" in accessed

    def test_empty_nested_dict(self):
        """Test tracking with empty nested dicts."""
        tracked = TrackedDict({"a": {}, "b": {"c": {}}})
        empty = tracked["a"]
        assert isinstance(empty, TrackedDict)
        assert len(empty) == 0

        nested_empty = tracked["b"]["c"]
        assert isinstance(nested_empty, TrackedDict)
        assert len(nested_empty) == 0

        accessed = tracked.get_accessed_paths()
        assert "a" in accessed
        assert "b" in accessed
        assert "b.c" in accessed

    def test_list_values_not_wrapped(self):
        """Test that list values are not wrapped (only dicts are)."""
        tracked = TrackedDict({"items": [1, 2, 3], "nested": {"list": [4, 5]}})
        items = tracked["items"]
        assert isinstance(items, list)
        assert items == [1, 2, 3]

        nested_list = tracked["nested"]["list"]
        assert isinstance(nested_list, list)
        assert nested_list == [4, 5]

    def test_mixed_types(self):
        """Test tracking with mixed value types."""
        tracked = TrackedDict(
            {
                "int": 42,
                "float": 3.14,
                "str": "hello",
                "bool": True,
                "none": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            }
        )

        # Access all values
        _ = tracked["int"]
        _ = tracked["float"]
        _ = tracked["str"]
        _ = tracked["bool"]
        _ = tracked["none"]
        _ = tracked["list"]
        nested = tracked["dict"]["nested"]

        accessed = tracked.get_accessed_paths()
        assert "int" in accessed
        assert "float" in accessed
        assert "str" in accessed
        assert "bool" in accessed
        assert "none" in accessed
        assert "list" in accessed
        assert "dict" in accessed
        assert "dict.nested" in accessed
        assert nested == "value"


class TestTrackedDictConflictDetection:
    """Test dependency conflict detection."""

    def _create_mock_experiment(self, exp_id: str, params: dict) -> MagicMock:
        """Create a mock Experiment with get_param() support."""
        mock_exp = MagicMock()
        mock_exp.id = exp_id

        def mock_get_param(key, default=None):
            """Navigate nested keys and return value or default."""
            keys = key.split(".")
            value = params
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

        mock_exp.get_param = mock_get_param
        return mock_exp

    def test_no_conflict_without_dependencies(self):
        """Test no conflict check when no dependencies."""
        tracked = TrackedDict({"lr": 0.01})
        # Should not raise
        assert tracked["lr"] == 0.01

    def test_no_conflict_same_value(self):
        """Test no conflict when dependency has same value."""
        mock_dep = self._create_mock_experiment("abc12345", {"lr": 0.01})
        tracked = TrackedDict({"lr": 0.01}, dependencies={"model": mock_dep})

        # Should not raise - values match
        assert tracked["lr"] == 0.01

    def test_no_conflict_dep_missing_param(self):
        """Test no conflict when dependency doesn't have the param."""
        mock_dep = self._create_mock_experiment("abc12345", {"other": "value"})
        tracked = TrackedDict({"lr": 0.01}, dependencies={"model": mock_dep})

        # Should not raise - only local has the param
        assert tracked["lr"] == 0.01

    def test_conflict_different_values(self):
        """Test conflict raises error when values differ."""
        mock_dep = self._create_mock_experiment("abc12345", {"lr": 0.001})
        tracked = TrackedDict({"lr": 0.01}, dependencies={"model": mock_dep})

        with pytest.raises(ParameterConflictError) as exc_info:
            _ = tracked["lr"]

        assert "lr" in str(exc_info.value)
        assert "0.01" in str(exc_info.value)
        assert "0.001" in str(exc_info.value)
        assert "model" in str(exc_info.value)
        assert "abc12345" in str(exc_info.value)

    def test_conflict_multiple_deps_disagree(self):
        """Test conflict when multiple deps have different values."""
        mock_dep1 = self._create_mock_experiment("abc12345", {"lr": 0.001})
        mock_dep2 = self._create_mock_experiment("def67890", {"lr": 0.0001})
        tracked = TrackedDict(
            {"lr": 0.01}, dependencies={"model1": mock_dep1, "model2": mock_dep2}
        )

        with pytest.raises(ParameterConflictError) as exc_info:
            _ = tracked["lr"]

        # Should mention all conflicting sources
        error_str = str(exc_info.value)
        assert "lr" in error_str
        assert "config" in error_str

    def test_conflict_deps_agree_but_differ_from_config(self):
        """Test conflict when deps agree but differ from config."""
        mock_dep1 = self._create_mock_experiment("abc12345", {"lr": 0.001})
        mock_dep2 = self._create_mock_experiment("def67890", {"lr": 0.001})
        tracked = TrackedDict(
            {"lr": 0.01},  # Config differs
            dependencies={"model1": mock_dep1, "model2": mock_dep2},
        )

        with pytest.raises(ParameterConflictError):
            _ = tracked["lr"]

    def test_no_conflict_all_sources_agree(self):
        """Test no conflict when all sources have same value."""
        mock_dep1 = self._create_mock_experiment("abc12345", {"lr": 0.01})
        mock_dep2 = self._create_mock_experiment("def67890", {"lr": 0.01})
        tracked = TrackedDict(
            {"lr": 0.01}, dependencies={"model1": mock_dep1, "model2": mock_dep2}
        )

        # Should not raise - all agree
        assert tracked["lr"] == 0.01

    def test_nested_param_conflict(self):
        """Test conflict detection for nested parameters."""
        mock_dep = self._create_mock_experiment(
            "abc12345", {"model": {"lr": 0.001, "layers": 3}}
        )
        tracked = TrackedDict(
            {"model": {"lr": 0.01, "layers": 3}}, dependencies={"dep": mock_dep}
        )

        # layers matches, should not raise
        assert tracked["model"]["layers"] == 3

        # lr differs, should raise
        with pytest.raises(ParameterConflictError) as exc_info:
            _ = tracked["model"]["lr"]

        assert "model.lr" in str(exc_info.value)

    def test_nested_no_conflict_different_keys(self):
        """Test no conflict when nested keys are different."""
        mock_dep = self._create_mock_experiment(
            "abc12345", {"model": {"batch_size": 32}}
        )
        tracked = TrackedDict({"model": {"lr": 0.01}}, dependencies={"dep": mock_dep})

        # Different keys - no conflict
        assert tracked["model"]["lr"] == 0.01

    def test_complex_value_comparison_list(self):
        """Test conflict detection with list values."""
        mock_dep = self._create_mock_experiment("abc12345", {"layers": [64, 128, 256]})
        tracked = TrackedDict(
            {"layers": [64, 128, 256]}, dependencies={"dep": mock_dep}
        )

        # Same list - no conflict
        assert tracked["layers"] == [64, 128, 256]

    def test_complex_value_conflict_list(self):
        """Test conflict with different list values."""
        mock_dep = self._create_mock_experiment("abc12345", {"layers": [32, 64]})
        tracked = TrackedDict({"layers": [64, 128]}, dependencies={"dep": mock_dep})

        with pytest.raises(ParameterConflictError):
            _ = tracked["layers"]

    def test_complex_value_comparison_dict(self):
        """Test conflict detection with dict values."""
        mock_dep = self._create_mock_experiment(
            "abc12345", {"optimizer": {"type": "adam", "lr": 0.01}}
        )
        tracked = TrackedDict(
            {"optimizer": {"type": "adam", "lr": 0.01}}, dependencies={"dep": mock_dep}
        )

        # Same dict - no conflict
        assert tracked["optimizer"] == {"type": "adam", "lr": 0.01}

    def test_dependency_loading_error_ignored(self):
        """Test that errors loading dependency params are gracefully handled."""
        mock_dep = MagicMock()
        mock_dep.id = "abc12345"
        mock_dep.get_param.side_effect = Exception("Storage error")

        tracked = TrackedDict({"lr": 0.01}, dependencies={"dep": mock_dep})

        # Should not raise - dependency error is silently ignored
        assert tracked["lr"] == 0.01

    def test_shared_state_across_nested(self):
        """Test that shared state is correctly propagated across nested TrackedDicts."""
        mock_dep = self._create_mock_experiment("abc12345", {"model": {"lr": 0.001}})
        tracked = TrackedDict({"model": {"lr": 0.01}}, dependencies={"dep": mock_dep})

        # Access nested dict
        model_dict = tracked["model"]

        # Verify shared state is propagated (no root reference)
        assert model_dict._dependencies is tracked._dependencies
        assert model_dict._accessed_paths is tracked._accessed_paths
        assert model_dict._lock is tracked._lock

        # Verify nested dict doesn't have access to root TrackedDict object
        # (this is important for safety - prevents accidental root iteration)
        assert not hasattr(model_dict, "_root")

        # Conflict should still be detected via shared dependencies
        with pytest.raises(ParameterConflictError):
            _ = model_dict["lr"]


class TestMakeHashable:
    """Test _make_hashable utility method."""

    def test_simple_values(self):
        """Test hashable conversion for simple types."""
        assert TrackedDict._make_hashable(42) == 42
        assert TrackedDict._make_hashable(3.14) == 3.14
        assert TrackedDict._make_hashable("hello") == "hello"
        assert TrackedDict._make_hashable(True) is True
        assert TrackedDict._make_hashable(None) is None

    def test_list_to_tuple(self):
        """Test list converted to tuple."""
        result = TrackedDict._make_hashable([1, 2, 3])
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_nested_list(self):
        """Test nested list conversion."""
        result = TrackedDict._make_hashable([[1, 2], [3, 4]])
        assert result == ((1, 2), (3, 4))

    def test_dict_to_sorted_tuple(self):
        """Test dict converted to sorted tuple of tuples."""
        result = TrackedDict._make_hashable({"b": 2, "a": 1})
        assert result == (("a", 1), ("b", 2))

    def test_nested_dict(self):
        """Test nested dict conversion."""
        result = TrackedDict._make_hashable({"outer": {"inner": 1}})
        assert result == (("outer", (("inner", 1),)),)

    def test_set_to_frozenset(self):
        """Test set converted to frozenset."""
        result = TrackedDict._make_hashable({1, 2, 3})
        assert result == frozenset({1, 2, 3})


class TestMissingSentinel:
    """Test _MISSING sentinel behavior."""

    def test_missing_is_singleton(self):
        """Test _MISSING is a consistent sentinel."""
        from yanex.core.tracked_dict import _MISSING, _MissingSentinel

        assert isinstance(_MISSING, _MissingSentinel)

    def test_missing_not_equal_to_none(self):
        """Test _MISSING is distinguishable from None."""
        assert _MISSING is not None
        assert _MISSING != None  # noqa: E711


class TestTrackedDictPop:
    """Test pop() method tracking."""

    def test_pop_tracks_access(self):
        """Test that pop() tracks the accessed key."""
        tracked = TrackedDict({"a": 1, "b": 2})
        value = tracked.pop("a")
        assert value == 1
        assert "a" in tracked.get_accessed_paths()
        assert "b" not in tracked.get_accessed_paths()

    def test_pop_removes_key(self):
        """Test that pop() removes the key from dict."""
        tracked = TrackedDict({"a": 1, "b": 2})
        tracked.pop("a")
        assert "a" not in tracked
        assert "b" in tracked

    def test_pop_with_default(self):
        """Test pop() with default for missing key."""
        tracked = TrackedDict({"a": 1})
        value = tracked.pop("missing", "default")
        assert value == "default"
        # Missing key should not be tracked
        assert "missing" not in tracked.get_accessed_paths()

    def test_pop_missing_raises_keyerror(self):
        """Test pop() raises KeyError for missing key without default."""
        tracked = TrackedDict({"a": 1})
        with pytest.raises(KeyError):
            tracked.pop("missing")

    def test_pop_nested_tracks_path(self):
        """Test pop() on nested TrackedDict tracks full path."""
        tracked = TrackedDict({"model": {"lr": 0.01, "epochs": 10}})
        model = tracked["model"]
        lr = model.pop("lr")
        assert lr == 0.01
        assert "model.lr" in tracked.get_accessed_paths()
        assert "lr" not in model  # Removed from nested dict

    def _create_mock_experiment(self, exp_id: str, params: dict) -> MagicMock:
        """Create a mock Experiment with get_param() support."""
        mock_exp = MagicMock()
        mock_exp.id = exp_id

        def mock_get_param(key, default=None):
            keys = key.split(".")
            value = params
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

        mock_exp.get_param = mock_get_param
        return mock_exp

    def test_pop_with_conflict_raises(self):
        """Test pop() raises ParameterConflictError on conflict."""
        mock_dep = self._create_mock_experiment("abc12345", {"lr": 0.001})
        tracked = TrackedDict({"lr": 0.01}, dependencies={"dep": mock_dep})

        with pytest.raises(ParameterConflictError):
            tracked.pop("lr")


class TestTrackedDictSetdefault:
    """Test setdefault() method tracking."""

    def test_setdefault_existing_key_tracks(self):
        """Test setdefault() tracks access for existing key."""
        tracked = TrackedDict({"a": 1, "b": 2})
        value = tracked.setdefault("a", 999)
        assert value == 1  # Original value
        assert "a" in tracked.get_accessed_paths()

    def test_setdefault_missing_key_tracks_and_sets(self):
        """Test setdefault() tracks and sets missing key."""
        tracked = TrackedDict({"a": 1})
        value = tracked.setdefault("b", 999)
        assert value == 999
        assert tracked["b"] == 999
        assert "b" in tracked.get_accessed_paths()

    def test_setdefault_default_none(self):
        """Test setdefault() with default None."""
        tracked = TrackedDict({"a": 1})
        value = tracked.setdefault("b")
        assert value is None
        assert tracked["b"] is None

    def test_setdefault_nested_wraps_dict(self):
        """Test setdefault() wraps nested dict values."""
        tracked = TrackedDict({"model": {"lr": 0.01}})
        model = tracked.setdefault("model", {})
        assert isinstance(model, TrackedDict)
        assert model["lr"] == 0.01

    def _create_mock_experiment(self, exp_id: str, params: dict) -> MagicMock:
        """Create a mock Experiment with get_param() support."""
        mock_exp = MagicMock()
        mock_exp.id = exp_id

        def mock_get_param(key, default=None):
            keys = key.split(".")
            value = params
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

        mock_exp.get_param = mock_get_param
        return mock_exp

    def test_setdefault_with_conflict_raises(self):
        """Test setdefault() raises ParameterConflictError on conflict."""
        mock_dep = self._create_mock_experiment("abc12345", {"lr": 0.001})
        tracked = TrackedDict({"lr": 0.01}, dependencies={"dep": mock_dep})

        with pytest.raises(ParameterConflictError):
            tracked.setdefault("lr", 0.1)


class TestTrackedDictPopitem:
    """Test popitem() method tracking."""

    def test_popitem_tracks_and_removes(self):
        """Test popitem() tracks the removed key."""
        tracked = TrackedDict({"a": 1})
        key, value = tracked.popitem()
        assert key == "a"
        assert value == 1
        assert "a" in tracked.get_accessed_paths()
        assert len(tracked) == 0

    def test_popitem_empty_raises(self):
        """Test popitem() raises KeyError on empty dict."""
        tracked = TrackedDict({})
        with pytest.raises(KeyError):
            tracked.popitem()

    def test_popitem_multiple(self):
        """Test multiple popitem() calls track all removed keys."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        items = []
        for _ in range(3):
            items.append(tracked.popitem())

        accessed = tracked.get_accessed_paths()
        assert len(accessed) == 3
        assert "a" in accessed
        assert "b" in accessed
        assert "c" in accessed


class TestTrackedDictCopy:
    """Test copy() method."""

    def test_copy_returns_tracked_dict(self):
        """Test copy() returns a TrackedDict, not plain dict."""
        tracked = TrackedDict({"a": 1, "b": 2})
        copied = tracked.copy()
        assert isinstance(copied, TrackedDict)

    def test_copy_shares_tracking(self):
        """Test copy() shares tracking state with original."""
        tracked = TrackedDict({"a": 1, "b": 2})
        copied = tracked.copy()

        # Access on copy should be tracked in original's paths
        _ = copied["a"]
        assert "a" in tracked.get_accessed_paths()
        assert "a" in copied.get_accessed_paths()

    def test_copy_is_shallow(self):
        """Test copy() is a shallow copy of data."""
        tracked = TrackedDict({"a": 1, "b": 2})
        copied = tracked.copy()

        # Modify copy doesn't affect original
        copied["c"] = 3
        assert "c" in copied
        assert "c" not in tracked

    def test_copy_nested_tracking(self):
        """Test copy() preserves nested tracking capability."""
        tracked = TrackedDict({"model": {"lr": 0.01, "epochs": 10}})
        copied = tracked.copy()

        # Access nested value on copy
        _ = copied["model"]["lr"]

        # Should be tracked in shared state
        accessed = tracked.get_accessed_paths()
        assert "model" in accessed
        assert "model.lr" in accessed

    def test_copy_preserves_dependencies(self):
        """Test copy() preserves dependency checking."""
        mock_dep = MagicMock()
        mock_dep.id = "abc12345"
        mock_dep.get_param.return_value = 0.001  # Different value

        tracked = TrackedDict({"lr": 0.01}, dependencies={"dep": mock_dep})
        copied = tracked.copy()

        # Conflict should still be detected on copy
        with pytest.raises(ParameterConflictError):
            _ = copied["lr"]

    def test_copy_does_not_mark_all_accessed(self):
        """Test copy() itself doesn't mark all keys as accessed."""
        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = tracked.copy()

        # copy() should not mark keys as accessed (unlike dict(tracked))
        assert tracked.get_accessed_paths() == set()


class TestTrackedDictPickle:
    """Test pickle support for TrackedDict."""

    def test_pickle_roundtrip(self):
        """Test TrackedDict can be pickled and unpickled."""
        import pickle

        tracked = TrackedDict({"a": 1, "b": 2, "model": {"lr": 0.01}})
        _ = tracked["a"]  # Access to track

        # Pickle and unpickle
        pickled = pickle.dumps(tracked)
        restored = pickle.loads(pickled)

        # Should have same data
        assert restored["a"] == 1
        assert restored["b"] == 2
        assert restored["model"]["lr"] == 0.01

    def test_pickle_preserves_data(self):
        """Test pickling preserves all dictionary data."""
        import pickle

        data = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "list": [1, 2, 3],
            "nested": {"deep": {"value": True}},
        }
        tracked = TrackedDict(data)

        restored = pickle.loads(pickle.dumps(tracked))

        assert dict(restored) == data

    def test_pickle_preserves_accessed_paths(self):
        """Test pickling preserves the accessed paths set."""
        import pickle

        tracked = TrackedDict({"a": 1, "b": 2, "c": 3})
        _ = tracked["a"]
        _ = tracked["b"]

        restored = pickle.loads(pickle.dumps(tracked))

        # Accessed paths should be preserved
        assert "a" in restored.get_accessed_paths()
        assert "b" in restored.get_accessed_paths()
        assert "c" not in restored.get_accessed_paths()

    def test_pickle_restores_lock(self):
        """Test unpickling creates a new functional lock."""
        import pickle

        tracked = TrackedDict({"a": 1})
        restored = pickle.loads(pickle.dumps(tracked))

        # Lock should work after unpickling
        with restored._lock:
            pass  # Should not raise

        # Tracking should still work (uses the lock)
        _ = restored["a"]
        assert "a" in restored.get_accessed_paths()

    def test_pickle_nested_tracking_works(self):
        """Test nested tracking works after unpickling."""
        import pickle

        tracked = TrackedDict({"model": {"train": {"lr": 0.01}}})
        restored = pickle.loads(pickle.dumps(tracked))

        # Access nested value
        _ = restored["model"]["train"]["lr"]

        # Should track the full path
        accessed = restored.get_accessed_paths()
        assert "model" in accessed
        assert "model.train" in accessed
        assert "model.train.lr" in accessed

    def test_pickle_preserves_path(self):
        """Test pickling preserves the path prefix."""
        import pickle

        tracked = TrackedDict({"a": 1}, path="root.nested")
        restored = pickle.loads(pickle.dumps(tracked))

        assert restored._path == "root.nested"

    def test_torch_save_compatible(self):
        """Test TrackedDict works with torch.save-like operations.

        torch.save uses pickle internally. This test verifies that TrackedDict
        can be pickled when embedded in another object (simulating a model
        that stores its config as a TrackedDict).
        """
        import pickle

        # Simulate what happens when a model stores TrackedDict config:
        # 1. Create a TrackedDict with parameters
        config = TrackedDict({"lr": 0.01, "epochs": 10})
        _ = config["lr"]  # Access param (triggers tracking)

        # 2. Store it in a dict (simulating model.__dict__ or a checkpoint dict)
        checkpoint = {"config": config, "weights": [1.0, 2.0, 3.0]}

        # 3. This should not raise "cannot pickle '_thread.lock' object"
        pickled = pickle.dumps(checkpoint)
        restored = pickle.loads(pickled)

        # 4. Verify restored config works correctly
        assert restored["config"]["lr"] == 0.01
        assert restored["config"]["epochs"] == 10
        assert isinstance(restored["config"], TrackedDict)
