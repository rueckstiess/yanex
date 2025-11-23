"""Tests for TrackedDict class."""

from concurrent.futures import ThreadPoolExecutor

from yanex.core.tracked_dict import TrackedDict


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
