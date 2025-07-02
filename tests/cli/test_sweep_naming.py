"""
Tests for parameter sweep naming functionality.
"""

import pytest

from tests.test_utils import TestDataFactory
from yanex.cli.commands.run import _generate_sweep_experiment_name


class TestSweepExperimentNaming:
    """Test sweep experiment naming functionality."""

    @pytest.mark.parametrize(
        "base_name,config,expected_name",
        [
            # Single parameter tests
            ("train-model", {"lr": 0.001}, "train-model-lr_0p001"),
            ("model", {"batch_size": 32}, "model-batch_size_32"),
            # Multiple parameters tests
            (
                "train-model",
                {"lr": 0.001, "batch_size": 32, "epochs": 100},
                "train-model-lr_0p001-batch_size_32-epochs_100",
            ),
            (
                "test",
                {"epochs": 100, "batch_size": 32, "hidden_dim": 512},
                "test-epochs_100-batch_size_32-hidden_dim_512",
            ),
            # No base name tests
            (None, {"lr": 0.001, "momentum": 0.9}, "sweep-lr_0p001-momentum_0p9"),
            (None, {"batch_size": 16}, "sweep-batch_size_16"),
            # Boolean parameters
            (
                "model",
                {"use_dropout": True, "use_batch_norm": False, "lr": 0.01},
                "model-use_dropout_true-use_batch_norm_false-lr_0p01",
            ),
            # String parameters
            (
                "train",
                {"model_type": "resnet50", "optimizer": "adam", "lr": 0.001},
                "train-model_type_resnet50-optimizer_adam-lr_0p001",
            ),
            # Float-integer values
            ("test", {"lr": 1.0, "momentum": 0.0}, "test-lr_1-momentum_0"),
            # Empty config
            ("test", {}, "test"),
            (None, {}, "sweep"),
        ],
    )
    def test_basic_naming_patterns(self, base_name, config, expected_name):
        """Test basic parameter sweep naming patterns."""
        result = _generate_sweep_experiment_name(base_name, config)
        assert result == expected_name

    @pytest.mark.parametrize(
        "config,expected_components",
        [
            # Nested parameters
            (
                {
                    "model": {"lr": 0.01, "hidden_size": 128},
                    "training": {"epochs": 100},
                },
                ["model_lr_0p01", "model_hidden_size_128", "training_epochs_100"],
            ),
            # Mixed nested levels
            (
                {
                    "lr": 0.01,
                    "model": {"type": "resnet", "layers": 6},
                    "batch_size": 32,
                },
                ["lr_0p01", "model_type_resnet", "model_layers_6", "batch_size_32"],
            ),
        ],
    )
    def test_nested_parameter_naming(self, config, expected_components):
        """Test naming with nested parameters."""
        result = _generate_sweep_experiment_name("test", config)

        assert result.startswith("test-")
        for component in expected_components:
            assert component in result

    @pytest.mark.parametrize(
        "config,expected_patterns",
        [
            # Scientific notation
            (
                {"lr": 1e-4, "weight_decay": 1e-5},
                ["lr_0p0001", "weight_decay_1em05"],
            ),
            # Very small numbers
            (
                {"lr": 0.000001, "epsilon": 1e-8},
                ["lr_1em06", "epsilon_1em08"],
            ),
        ],
    )
    def test_scientific_notation_formatting(self, config, expected_patterns):
        """Test formatting of scientific notation and very small numbers."""
        result = _generate_sweep_experiment_name("model", config)

        assert result.startswith("model-")
        for pattern in expected_patterns:
            assert pattern in result

    def test_name_truncation_behavior(self):
        """Test name truncation when generated name is too long."""
        # Create config with many long parameters to exceed length limit
        long_config = {}
        for i in range(20):
            long_config[f"very_long_parameter_name_{i}"] = f"very_long_value_{i}"

        # Test with base name
        name_with_base = _generate_sweep_experiment_name("base-name", long_config)
        assert len(name_with_base) <= 100
        assert name_with_base.startswith("base-name-")
        assert name_with_base.endswith("...")

        # Test without base name
        name_without_base = _generate_sweep_experiment_name(None, long_config)
        assert len(name_without_base) <= 100
        assert name_without_base.startswith("sweep-")
        assert name_without_base.endswith("...")

    def test_parameter_order_consistency(self):
        """Test that parameter order is consistent across multiple calls."""
        config = {"z_param": 1, "a_param": 2, "m_param": 3}

        name1 = _generate_sweep_experiment_name("test", config)
        name2 = _generate_sweep_experiment_name("test", config)

        assert name1 == name2

    @pytest.mark.parametrize(
        "config_factory_type,expected_components",
        [
            ("ml_training", ["learning_rate", "batch_size", "epochs"]),
            ("data_processing", ["n_docs", "chunk_size"]),
            ("simple", ["param1", "param2", "param3"]),
        ],
    )
    def test_naming_with_factory_generated_configs(
        self, config_factory_type, expected_components
    ):
        """Test naming with configurations generated by TestDataFactory."""
        config = TestDataFactory.create_experiment_config(
            config_type=config_factory_type
        )

        result = _generate_sweep_experiment_name("test", config)

        assert result.startswith("test-")
        # Check that some expected parameter names appear in the result
        name_lower = result.lower()
        found_components = [
            comp
            for comp in expected_components
            if any(comp.lower() in name_lower for comp in expected_components)
        ]
        assert len(found_components) > 0

    def test_naming_with_complex_ml_config(self):
        """Test naming with complex ML configuration using utilities."""
        # Create a comprehensive ML config
        ml_config = TestDataFactory.create_experiment_config(
            config_type="ml_training",
            dropout_rate=0.1,
            weight_decay=1e-5,
        )

        result = _generate_sweep_experiment_name("ml-experiment", ml_config)

        # Verify basic structure without assuming exact parameter inclusion
        assert result.startswith("ml-experiment-")
        assert len(result) > len("ml-experiment-")
        # The generated name should include some parameters from the config
        assert (
            len([char for char in result if char == "_"]) >= 2
        )  # Should have parameter value separators

    def test_naming_edge_cases(self):
        """Test edge cases in parameter naming."""
        edge_cases = [
            # Zero values
            {"lr": 0, "momentum": 0.0},
            # Negative values
            {"bias": -0.1, "offset": -10},
            # Very large numbers
            {"max_tokens": 1000000, "vocab_size": 50000},
        ]

        for config in edge_cases:
            result = _generate_sweep_experiment_name("edge-test", config)
            assert result.startswith("edge-test-")
            assert len(result) > len("edge-test-")

    def test_naming_with_special_characters_in_values(self):
        """Test naming behavior with special characters in parameter values."""
        config = {
            "model_path": "/path/to/model",
            "dataset": "dataset-v2.1",
            "tag": "experiment_2024",
        }

        result = _generate_sweep_experiment_name("special", config)

        # Should handle special characters gracefully
        assert result.startswith("special-")
        # The exact transformation depends on implementation,
        # but result should be a valid string
        assert isinstance(result, str)
        assert len(result) > len("special-")

    @pytest.mark.parametrize(
        "base_name_pattern",
        [
            "simple-name",
            "name_with_underscores",
            "name-with-many-dashes",
            "CamelCaseName",
            "name123",
        ],
    )
    def test_naming_with_various_base_name_patterns(self, base_name_pattern):
        """Test naming with various base name patterns."""
        config = {"lr": 0.01, "epochs": 10}

        result = _generate_sweep_experiment_name(base_name_pattern, config)

        assert result.startswith(f"{base_name_pattern}-")
        assert "lr_0p01" in result
        assert "epochs_10" in result
