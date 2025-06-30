"""
Tests for parameter sweep naming functionality.
"""

from yanex.cli.commands.run import _generate_sweep_experiment_name


class TestSweepExperimentNaming:
    """Test sweep experiment naming functionality."""

    def test_naming_with_base_name_single_param(self):
        """Test naming with base name and single parameter."""
        config = {"lr": 0.001}
        name = _generate_sweep_experiment_name("train-model", config)
        assert name == "train-model-lr_0p001"

    def test_naming_with_base_name_multiple_params(self):
        """Test naming with base name and multiple parameters."""
        config = {"lr": 0.001, "batch_size": 32, "epochs": 100}
        name = _generate_sweep_experiment_name("train-model", config)
        assert name == "train-model-lr_0p001-batch_size_32-epochs_100"

    def test_naming_without_base_name(self):
        """Test naming without base name."""
        config = {"lr": 0.001, "momentum": 0.9}
        name = _generate_sweep_experiment_name(None, config)
        assert name == "sweep-lr_0p001-momentum_0p9"

    def test_naming_with_nested_parameters(self):
        """Test naming with nested parameters."""
        config = {
            "model": {"lr": 0.01, "hidden_size": 128},
            "training": {"epochs": 100}
        }
        name = _generate_sweep_experiment_name("test", config)
        assert name == "test-model_lr_0p01-model_hidden_size_128-training_epochs_100"

    def test_naming_with_boolean_parameters(self):
        """Test naming with boolean parameters."""
        config = {"use_dropout": True, "use_batch_norm": False, "lr": 0.01}
        name = _generate_sweep_experiment_name("model", config)
        assert name == "model-use_dropout_true-use_batch_norm_false-lr_0p01"

    def test_naming_with_string_parameters(self):
        """Test naming with string parameters."""
        config = {"model_type": "resnet50", "optimizer": "adam", "lr": 0.001}
        name = _generate_sweep_experiment_name("train", config)
        assert name == "train-model_type_resnet50-optimizer_adam-lr_0p001"

    def test_naming_with_integer_parameters(self):
        """Test naming with integer parameters."""
        config = {"epochs": 100, "batch_size": 32, "hidden_dim": 512}
        name = _generate_sweep_experiment_name("test", config)
        assert name == "test-epochs_100-batch_size_32-hidden_dim_512"

    def test_naming_with_scientific_notation(self):
        """Test naming with scientific notation values."""
        config = {"lr": 1e-4, "weight_decay": 1e-5}
        name = _generate_sweep_experiment_name("model", config)
        assert name == "model-lr_0p0001-weight_decay_1em05"

    def test_naming_with_very_small_numbers(self):
        """Test naming with very small numbers."""
        config = {"lr": 0.000001, "epsilon": 1e-8}
        name = _generate_sweep_experiment_name("test", config)
        assert name == "test-lr_1em06-epsilon_1em08"

    def test_naming_with_float_integers(self):
        """Test naming with floats that are actually integers."""
        config = {"lr": 1.0, "momentum": 0.0}
        name = _generate_sweep_experiment_name("test", config)
        assert name == "test-lr_1-momentum_0"

    def test_naming_truncation_with_base_name(self):
        """Test name truncation when too long with base name."""
        # Create a config that would result in a very long name
        config = {}
        for i in range(20):
            config[f"very_long_parameter_name_{i}"] = f"very_long_value_{i}"
        
        name = _generate_sweep_experiment_name("base-name", config)
        assert len(name) <= 100
        assert name.startswith("base-name-")
        assert name.endswith("...")

    def test_naming_truncation_without_base_name(self):
        """Test name truncation when too long without base name."""
        # Create a config that would result in a very long name
        config = {}
        for i in range(20):
            config[f"very_long_parameter_name_{i}"] = f"very_long_value_{i}"
        
        name = _generate_sweep_experiment_name(None, config)
        assert len(name) <= 100
        assert name.startswith("sweep-")
        assert name.endswith("...")

    def test_naming_with_empty_config(self):
        """Test naming with empty configuration."""
        config = {}
        name = _generate_sweep_experiment_name("test", config)
        assert name == "test"
        
        name = _generate_sweep_experiment_name(None, config)
        assert name == "sweep"

    def test_naming_parameter_order_consistency(self):
        """Test that parameter order is consistent."""
        config = {"z_param": 1, "a_param": 2, "m_param": 3}
        name1 = _generate_sweep_experiment_name("test", config)
        name2 = _generate_sweep_experiment_name("test", config)
        assert name1 == name2
        # Note: We can't guarantee alphabetical order without sorting,
        # but we can ensure consistency

    def test_naming_with_mixed_nested_levels(self):
        """Test naming with mixed nested parameter levels."""
        config = {
            "lr": 0.01,
            "model": {
                "type": "resnet",
                "layers": 6
            },
            "batch_size": 32
        }
        name = _generate_sweep_experiment_name("test", config)
        # Check that it contains the expected components
        assert name.startswith("test-")
        assert "lr_0p01" in name
        assert "model_type_resnet" in name
        assert "model_layers_6" in name
        assert "batch_size_32" in name