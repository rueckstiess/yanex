"""Tests for dictionary utility functions."""

from yanex.utils.dict_utils import deep_merge, flatten_dict, unflatten_dict


class TestFlattenDict:
    """Test cases for flatten_dict function."""

    def test_flatten_simple_dict(self):
        """Test flattening a simple flat dictionary."""
        input_dict = {"a": 1, "b": 2, "c": 3}
        expected = {"a": 1, "b": 2, "c": 3}
        assert flatten_dict(input_dict) == expected

    def test_flatten_nested_dict(self):
        """Test flattening a nested dictionary."""
        input_dict = {"a": {"b": 1, "c": 2}, "d": 3}
        expected = {"a.b": 1, "a.c": 2, "d": 3}
        assert flatten_dict(input_dict) == expected

    def test_flatten_deeply_nested_dict(self):
        """Test flattening a deeply nested dictionary."""
        input_dict = {"a": {"b": {"c": {"d": 1}}}, "e": 2}
        expected = {"a.b.c.d": 1, "e": 2}
        assert flatten_dict(input_dict) == expected

    def test_flatten_dict_with_list_values(self):
        """Test flattening a dict with list values (lists are not flattened)."""
        input_dict = {"a": [1, 2, 3], "b": {"c": [4, 5]}}
        expected = {"a": [1, 2, 3], "b.c": [4, 5]}
        assert flatten_dict(input_dict) == expected

    def test_flatten_empty_dict(self):
        """Test flattening an empty dictionary."""
        assert flatten_dict({}) == {}

    def test_flatten_dict_with_empty_nested_dict(self):
        """Test flattening a dict containing an empty dict."""
        input_dict = {"a": {}, "b": 1}
        expected = {"a": {}, "b": 1}
        assert flatten_dict(input_dict) == expected

    def test_flatten_config_example(self):
        """Test flattening a realistic configuration dictionary."""
        config = {
            "model": {"learning_rate": 0.001, "hidden_size": 128, "dropout": 0.1},
            "training": {"epochs": 10, "batch_size": 32, "optimizer": "adam"},
            "data": {"dataset": "mnist", "train_split": 0.8, "validation_split": 0.2},
        }
        expected = {
            "model.learning_rate": 0.001,
            "model.hidden_size": 128,
            "model.dropout": 0.1,
            "training.epochs": 10,
            "training.batch_size": 32,
            "training.optimizer": "adam",
            "data.dataset": "mnist",
            "data.train_split": 0.8,
            "data.validation_split": 0.2,
        }
        assert flatten_dict(config) == expected

    def test_flatten_custom_separator(self):
        """Test flattening with a custom separator."""
        input_dict = {"a": {"b": 1, "c": 2}, "d": 3}
        expected = {"a/b": 1, "a/c": 2, "d": 3}
        assert flatten_dict(input_dict, separator="/") == expected


class TestUnflattenDict:
    """Test cases for unflatten_dict function."""

    def test_unflatten_simple_dict(self):
        """Test unflattening a simple flat dictionary."""
        input_dict = {"a": 1, "b": 2, "c": 3}
        expected = {"a": 1, "b": 2, "c": 3}
        assert unflatten_dict(input_dict) == expected

    def test_unflatten_nested_dict(self):
        """Test unflattening a dictionary with dot notation."""
        input_dict = {"a.b": 1, "a.c": 2, "d": 3}
        expected = {"a": {"b": 1, "c": 2}, "d": 3}
        assert unflatten_dict(input_dict) == expected

    def test_unflatten_deeply_nested_dict(self):
        """Test unflattening a deeply nested dictionary."""
        input_dict = {"a.b.c.d": 1, "e": 2}
        expected = {"a": {"b": {"c": {"d": 1}}}, "e": 2}
        assert unflatten_dict(input_dict) == expected

    def test_unflatten_empty_dict(self):
        """Test unflattening an empty dictionary."""
        assert unflatten_dict({}) == {}

    def test_unflatten_config_example(self):
        """Test unflattening a realistic flattened configuration."""
        flat_config = {
            "model.learning_rate": 0.001,
            "model.hidden_size": 128,
            "model.dropout": 0.1,
            "training.epochs": 10,
            "training.batch_size": 32,
            "training.optimizer": "adam",
            "data.dataset": "mnist",
            "data.train_split": 0.8,
            "data.validation_split": 0.2,
        }
        expected = {
            "model": {"learning_rate": 0.001, "hidden_size": 128, "dropout": 0.1},
            "training": {"epochs": 10, "batch_size": 32, "optimizer": "adam"},
            "data": {"dataset": "mnist", "train_split": 0.8, "validation_split": 0.2},
        }
        assert unflatten_dict(flat_config) == expected

    def test_unflatten_custom_separator(self):
        """Test unflattening with a custom separator."""
        input_dict = {"a/b": 1, "a/c": 2, "d": 3}
        expected = {"a": {"b": 1, "c": 2}, "d": 3}
        assert unflatten_dict(input_dict, separator="/") == expected


class TestRoundTrip:
    """Test round-trip conversion (flatten -> unflatten)."""

    def test_roundtrip_nested_dict(self):
        """Test that flatten and unflatten are inverse operations."""
        original = {"a": {"b": {"c": 1}}, "d": 2}
        flattened = flatten_dict(original)
        unflattened = unflatten_dict(flattened)
        assert unflattened == original

    def test_roundtrip_config_example(self):
        """Test round-trip with a realistic configuration."""
        original = {
            "model": {"learning_rate": 0.001, "hidden_size": 128},
            "training": {"epochs": 10, "batch_size": 32},
        }
        flattened = flatten_dict(original)
        unflattened = unflatten_dict(flattened)
        assert unflattened == original

    def test_roundtrip_with_mixed_values(self):
        """Test round-trip with various value types."""
        original = {
            "a": {"b": 1, "c": "string"},
            "d": [1, 2, 3],
            "e": {"f": {"g": True}},
        }
        flattened = flatten_dict(original)
        unflattened = unflatten_dict(flattened)
        assert unflattened == original


class TestDeepMerge:
    """Test cases for deep_merge function."""

    def test_merge_simple_dicts(self):
        """Test merging simple flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        result = deep_merge(base, override)
        assert result == {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}

    def test_merge_deeply_nested_dicts(self):
        """Test merging deeply nested dictionaries."""
        base = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}
        override = {"a": {"b": {"c": 10, "f": 4}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 10, "d": 2, "f": 4}}, "e": 3}

    def test_merge_empty_base(self):
        """Test merging with empty base dictionary."""
        base = {}
        override = {"a": 1, "b": {"c": 2}}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_merge_empty_override(self):
        """Test merging with empty override dictionary."""
        base = {"a": 1, "b": {"c": 2}}
        override = {}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_merge_both_empty(self):
        """Test merging two empty dictionaries."""
        result = deep_merge({}, {})
        assert result == {}

    def test_merge_does_not_modify_inputs(self):
        """Test that deep_merge does not modify input dictionaries."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        base_copy = {"a": {"b": 1}}
        override_copy = {"a": {"c": 2}}

        deep_merge(base, override)

        assert base == base_copy
        assert override == override_copy

    def test_merge_override_replaces_non_dict_with_dict(self):
        """Test that override dict replaces base non-dict value."""
        base = {"a": 1}
        override = {"a": {"b": 2}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": 2}}

    def test_merge_override_replaces_dict_with_non_dict(self):
        """Test that override non-dict replaces base dict value."""
        base = {"a": {"b": 2}}
        override = {"a": 1}
        result = deep_merge(base, override)
        assert result == {"a": 1}

    def test_merge_config_example(self):
        """Test merging realistic configuration dictionaries."""
        # Dependency params
        dep_config = {
            "model": {"lr": 0.001, "layers": 3, "n_embd": 128},
            "training": {"epochs": 10},
        }
        # Local params (override lr but keep layers and n_embd from dep)
        local_config = {
            "model": {"lr": 0.01},
            "training": {"epochs": 20, "batch_size": 32},
        }
        result = deep_merge(dep_config, local_config)
        expected = {
            "model": {"lr": 0.01, "layers": 3, "n_embd": 128},
            "training": {"epochs": 20, "batch_size": 32},
        }
        assert result == expected

    def test_merge_with_list_values(self):
        """Test that lists are replaced (not merged element-wise)."""
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"a": [4, 5]}
