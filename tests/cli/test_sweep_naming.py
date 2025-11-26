"""
Tests for parameter sweep naming functionality.
"""

import pytest

from yanex.cli.commands.run import _generate_sweep_experiment_name


class TestSweepExperimentNaming:
    """Test sweep experiment naming functionality."""

    @pytest.mark.parametrize(
        "base_name,config,sweep_param_paths,dependency_id,expected_name",
        [
            # No sweep params or dependencies - name unchanged
            ("train-model", {"lr": 0.001}, None, None, "train-model"),
            ("model", {"batch_size": 32}, [], None, "model"),
            ("test", {"epochs": 100}, None, None, "test"),
            # No base name - defaults to "sweep"
            (None, {"lr": 0.001}, None, None, "sweep"),
            # Single sweep parameter
            ("model", {"lr": 0.01}, ["lr"], None, "model-0-01"),
            ("train", {"batch_size": 32}, ["batch_size"], None, "train-32"),
            (None, {"lr": 0.001}, ["lr"], None, "sweep-0-001"),
            # Multiple sweep parameters
            (
                "model",
                {"lr": 0.01, "batch_size": 32},
                ["lr", "batch_size"],
                None,
                "model-0-01-32",
            ),
            (
                "train",
                {"lr": 0.1, "momentum": 0.9, "epochs": 100},
                ["lr", "momentum"],
                None,
                "train-0-1-0-9",
            ),
            # Dependency sweep (no params)
            ("model", {}, None, "abc12345", "model-abc12345"),
            (None, {}, None, "def67890", "sweep-def67890"),
            # Dependency + parameter sweep
            (
                "model",
                {"lr": 0.01},
                ["lr"],
                "abc12345",
                "model-abc12345-0-01",
            ),
            (
                "train",
                {"lr": 0.1, "batch_size": 32},
                ["lr", "batch_size"],
                "xyz98765",
                "train-xyz98765-0-1-32",
            ),
        ],
    )
    def test_basic_naming_patterns(
        self, base_name, config, sweep_param_paths, dependency_id, expected_name
    ):
        """Test basic parameter sweep naming patterns."""
        result = _generate_sweep_experiment_name(
            base_name, config, sweep_param_paths, dependency_id
        )
        assert result == expected_name

    def test_nested_parameter_naming(self):
        """Test naming with nested parameters - only leaf values used."""
        config = {
            "model": {"lr": 0.01, "hidden_size": 128},
            "training": {"epochs": 100},
        }
        # Only sweep on model.lr
        sweep_paths = ["model.lr"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-0-01"

        # Sweep on multiple nested params
        sweep_paths = ["model.lr", "training.epochs"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-0-01-100"

    def test_scientific_notation_formatting(self):
        """Test that scientific notation is formatted correctly."""
        config = {"lr": 1e-4, "weight_decay": 1e-5}
        sweep_paths = ["lr", "weight_decay"]
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        # Scientific notation converted to string then sanitized
        assert result == "model-0-0001-1e-05"

    def test_string_parameter_formatting(self):
        """Test string parameter formatting."""
        config = {"sequence_order": "ORDERED"}
        sweep_paths = ["sequence_order"]
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        # String converted to lowercase
        assert result == "model-ordered"

        config = {"sequence_order": "SHUFFLED"}
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        assert result == "model-shuffled"

    def test_special_characters_in_values(self):
        """Test that special characters are replaced with dashes."""
        config = {"dataset": "COCO 2017"}
        sweep_paths = ["dataset"]
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        # Spaces and special chars replaced with dash, lowercase
        assert result == "model-coco-2017"

        config = {"model_type": "ResNet-50"}
        sweep_paths = ["model_type"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-resnet-50"

    def test_list_parameter_formatting(self):
        """Test list parameter formatting."""
        config = {"model_types": ["cnn", "rnn"]}
        sweep_paths = ["model_types"]
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        # List elements joined with dash
        assert result == "model-cnn-rnn"

        config = {"layers": [128, 64, 32]}
        sweep_paths = ["layers"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-128-64-32"

    def test_dict_parameter_formatting(self):
        """Test dict parameter formatting."""
        config = {"optimizer": {"type": "adam", "lr": 0.01}}
        sweep_paths = ["optimizer"]
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        # Dict keys and values interleaved
        assert result == "model-type-adam-lr-0-01"

        config = {"config": {"a": 1, "b": 2}}
        sweep_paths = ["config"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-a-1-b-2"

    def test_boolean_parameter_formatting(self):
        """Test boolean parameter formatting."""
        config = {"use_dropout": True}
        sweep_paths = ["use_dropout"]
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        assert result == "model-true"

        config = {"use_batch_norm": False}
        sweep_paths = ["use_batch_norm"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-false"

    def test_zero_and_negative_values(self):
        """Test zero and negative value formatting."""
        config = {"lr": 0, "momentum": -0.1}
        sweep_paths = ["lr", "momentum"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-0-0-1"

    def test_very_long_names(self):
        """Test that very long names are not truncated."""
        config = {
            "lr": 0.001,
            "batch_size": 32,
            "hidden_size": 512,
            "dropout": 0.1,
        }
        sweep_paths = ["lr", "batch_size", "hidden_size", "dropout"]
        result = _generate_sweep_experiment_name("model", config, sweep_paths)
        # All values included, no truncation
        assert result == "model-0-001-32-512-0-1"

    def test_base_name_with_spaces(self):
        """Test base name with spaces is preserved."""
        config = {"lr": 0.01}
        sweep_paths = ["lr"]
        # Note: base name is not sanitized, only parameter values are
        result = _generate_sweep_experiment_name("Model Training", config, sweep_paths)
        assert result == "Model Training-0-01"

    def test_multiple_dependency_sweep(self):
        """Test naming with dependency sweep."""
        config = {}
        # Different dependency IDs
        dep_id_1 = "abc12345"
        dep_id_2 = "def67890"

        result_1 = _generate_sweep_experiment_name("model", config, None, dep_id_1)
        assert result_1 == "model-abc12345"

        result_2 = _generate_sweep_experiment_name("model", config, None, dep_id_2)
        assert result_2 == "model-def67890"

    def test_cartesian_sweep(self):
        """Test naming for cartesian product sweep (dependency × parameter)."""
        config_1 = {"lr": 0.01}
        config_2 = {"lr": 0.1}
        sweep_paths = ["lr"]

        dep_id_1 = "abc12345"
        dep_id_2 = "def67890"

        # Should generate: model-abc12345-0.01, model-abc12345-0.1, model-def67890-0.01, model-def67890-0.1
        result_1_1 = _generate_sweep_experiment_name(
            "model", config_1, sweep_paths, dep_id_1
        )
        assert result_1_1 == "model-abc12345-0-01"

        result_1_2 = _generate_sweep_experiment_name(
            "model", config_2, sweep_paths, dep_id_1
        )
        assert result_1_2 == "model-abc12345-0-1"

        result_2_1 = _generate_sweep_experiment_name(
            "model", config_1, sweep_paths, dep_id_2
        )
        assert result_2_1 == "model-def67890-0-01"

        result_2_2 = _generate_sweep_experiment_name(
            "model", config_2, sweep_paths, dep_id_2
        )
        assert result_2_2 == "model-def67890-0-1"

    def test_empty_sweep_paths(self):
        """Test behavior with empty sweep paths list."""
        config = {"lr": 0.01, "batch_size": 32}
        result = _generate_sweep_experiment_name("test", config, [])
        # No sweep params, name unchanged
        assert result == "test"

    def test_sweep_parameter_filtering(self):
        """Test that only sweep parameters are included in name."""
        config = {
            "lr": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "model": "resnet",
        }
        # Only lr is sweep parameter
        sweep_paths = ["lr"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-0-01"

        # lr and batch_size are sweep parameters
        sweep_paths = ["lr", "batch_size"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-0-01-32"

    def test_no_base_name_with_sweeps(self):
        """Test default 'sweep' name with parameter sweeps."""
        config = {"lr": 0.01}
        sweep_paths = ["lr"]
        result = _generate_sweep_experiment_name(None, config, sweep_paths)
        assert result == "sweep-0-01"

    def test_no_base_name_with_dependency(self):
        """Test default 'sweep' name with dependency sweep."""
        config = {}
        dep_id = "abc12345"
        result = _generate_sweep_experiment_name(None, config, None, dep_id)
        assert result == "sweep-abc12345"

    def test_complex_nested_structure(self):
        """Test with complex nested parameter structure."""
        config = {
            "model": {
                "architecture": {
                    "type": "transformer",
                    "layers": 12,
                },
                "dropout": 0.1,
            }
        }
        sweep_paths = ["model.architecture.type", "model.dropout"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        assert result == "test-transformer-0-1"

    def test_underscore_and_dash_handling(self):
        """Test that underscores and dashes in values are preserved."""
        config = {"model_type": "bert_base"}
        sweep_paths = ["model_type"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        # Underscores are special chars, replaced with dash
        assert result == "test-bert-base"

    def test_unicode_characters(self):
        """Test that unicode characters are handled."""
        config = {"model": "model_α"}
        sweep_paths = ["model"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        # Non-alphanumeric replaced with dash
        assert result == "test-model"

    def test_multiple_consecutive_special_chars(self):
        """Test that multiple consecutive special chars are collapsed."""
        config = {"path": "/path//to///file"}
        sweep_paths = ["path"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        # Multiple special chars replaced with single dash
        assert result == "test-path-to-file"

    def test_leading_trailing_special_chars(self):
        """Test that leading/trailing special chars are removed."""
        config = {"tag": "_test_"}
        sweep_paths = ["tag"]
        result = _generate_sweep_experiment_name("test", config, sweep_paths)
        # Leading/trailing dashes removed
        assert result == "test-test"
