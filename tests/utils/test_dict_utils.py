"""Tests for dictionary utility functions."""

from yanex.utils.dict_utils import flatten_dict, unflatten_dict


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
