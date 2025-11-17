"""
Tests for yanex.core.config module.
"""

from unittest.mock import patch

import pytest
import yaml

from tests.test_utils import TestDataFactory, TestFileHelpers
from yanex.core.config import (
    load_yaml_config,
    merge_configs,
    parse_param_overrides,
    resolve_config,
    save_yaml_config,
)
from yanex.utils.exceptions import ConfigError


class TestLoadYamlConfig:
    """Test load_yaml_config function."""

    @pytest.mark.parametrize(
        "config_type,expected_keys",
        [
            ("ml_training", ["learning_rate", "batch_size", "epochs"]),
            ("data_processing", ["n_docs", "chunk_size"]),
            ("simple", ["param1", "param2", "param3"]),
        ],
    )
    def test_load_valid_config(self, temp_dir, config_type, expected_keys):
        """Test loading valid YAML configuration with various config types."""
        config_path = temp_dir / "config.yaml"
        config_data = TestDataFactory.create_experiment_config(config_type=config_type)

        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        result = load_yaml_config(config_path)
        assert result == config_data

        # Verify expected keys are present
        for key in expected_keys:
            assert key in result

    @pytest.mark.parametrize(
        "file_content,expected_result",
        [
            ("", {}),
            ("# Just comments\n", {}),
            ("null", {}),
        ],
    )
    def test_load_empty_config(self, temp_dir, file_content, expected_result):
        """Test loading empty or null YAML files."""
        config_path = temp_dir / "empty.yaml"
        TestFileHelpers.create_test_file(config_path, file_content)

        result = load_yaml_config(config_path)
        assert result == expected_result

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading nonexistent file raises ConfigError."""
        config_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_yaml_config(config_path)

    def test_load_directory_instead_of_file(self, temp_dir):
        """Test loading directory raises ConfigError."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        with pytest.raises(ConfigError, match="Configuration path is not a file"):
            load_yaml_config(config_dir)

    @pytest.mark.parametrize(
        "invalid_content,error_pattern",
        [
            ("invalid: yaml: content: [unclosed", "Failed to parse YAML config"),
            ("- item1\n- item2", "Configuration must be a dictionary"),
            ("just a string", "Configuration must be a dictionary"),
            ("123", "Configuration must be a dictionary"),
        ],
    )
    def test_load_invalid_yaml(self, temp_dir, invalid_content, error_pattern):
        """Test loading invalid YAML raises ConfigError."""
        config_path = temp_dir / "invalid.yaml"
        TestFileHelpers.create_test_file(config_path, invalid_content)

        with pytest.raises(ConfigError, match=error_pattern):
            load_yaml_config(config_path)


class TestSaveYamlConfig:
    """Test save_yaml_config function."""

    @pytest.mark.parametrize(
        "config_type",
        [
            "ml_training",
            "data_processing",
            "simple",
        ],
    )
    def test_save_valid_config(self, temp_dir, config_type):
        """Test saving valid configuration with various config types."""
        config_path = temp_dir / "output.yaml"
        config_data = TestDataFactory.create_experiment_config(config_type=config_type)

        save_yaml_config(config_data, config_path)

        assert config_path.exists()

        # Load back and verify
        with config_path.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded == config_data

    def test_save_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        config_path = temp_dir / "subdir" / "config.yaml"
        config_data = TestDataFactory.create_experiment_config(config_type="simple")

        save_yaml_config(config_data, config_path)

        assert config_path.exists()
        assert config_path.parent.is_dir()

    @pytest.mark.parametrize(
        "invalid_data,error_pattern",
        [
            ("not a dict", "Configuration must be a dictionary"),
            (123, "Configuration must be a dictionary"),
            ([], "Configuration must be a dictionary"),
            (None, "Configuration must be a dictionary"),
        ],
    )
    def test_save_invalid_config_data(self, temp_dir, invalid_data, error_pattern):
        """Test saving invalid config data raises ValidationError."""
        config_path = temp_dir / "output.yaml"

        from yanex.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match=error_pattern):
            save_yaml_config(invalid_data, config_path)


class TestParseParamOverrides:
    """Test parse_param_overrides function."""

    @pytest.mark.parametrize(
        "param_strings,expected_result",
        [
            (
                ["param1=value1", "param2=123", "param3=true"],
                {"param1": "value1", "param2": 123, "param3": True},
            ),
            (
                ["learning_rate=0.01", "epochs=100"],
                {"learning_rate": 0.01, "epochs": 100},
            ),
            (["batch_size=32", "dropout=0.5"], {"batch_size": 32, "dropout": 0.5}),
        ],
    )
    def test_parse_simple_params(self, param_strings, expected_result):
        """Test parsing simple parameter overrides."""
        result = parse_param_overrides(param_strings)
        assert result == expected_result

    @pytest.mark.parametrize(
        "param_strings,expected_structure",
        [
            (
                ["model.learning_rate=0.01", "model.batch_size=32"],
                {"model": {"learning_rate": 0.01, "batch_size": 32}},
            ),
            (
                ["data.train.path=/train", "data.test.path=/test"],
                {"data": {"train": {"path": "/train"}, "test": {"path": "/test"}}},
            ),
        ],
    )
    def test_parse_nested_params(self, param_strings, expected_structure):
        """Test parsing nested parameter overrides."""
        result = parse_param_overrides(param_strings)
        assert result == expected_structure

    @pytest.mark.parametrize(
        "param_strings,expected_result",
        [
            (["flag1=true", "flag2=false"], {"flag1": True, "flag2": False}),
            (["flag3=yes", "flag4=no"], {"flag3": True, "flag4": False}),
            (["flag5=1", "flag6=0"], {"flag5": True, "flag6": False}),
            (["flag7=on", "flag8=off"], {"flag7": True, "flag8": False}),
        ],
    )
    def test_parse_boolean_values(self, param_strings, expected_result):
        """Test parsing boolean parameter values."""
        result = parse_param_overrides(param_strings)
        assert result == expected_result

    @pytest.mark.parametrize(
        "param_strings,expected_result",
        [
            (["int_param=42"], {"int_param": 42}),
            (["float_param=3.14"], {"float_param": 3.14}),
            (["exp_param=1e-5"], {"exp_param": 1e-5}),
            (["negative=-123"], {"negative": -123}),
            (["zero=0"], {"zero": 0}),
        ],
    )
    def test_parse_numeric_values(self, param_strings, expected_result):
        """Test parsing numeric parameter values."""
        result = parse_param_overrides(param_strings)
        assert result == expected_result

    @pytest.mark.parametrize(
        "param_strings,expected_result",
        [
            (["param1=null"], {"param1": None}),
            (["param2=none"], {"param2": None}),
            (["param3=~"], {"param3": None}),
        ],
    )
    def test_parse_null_values(self, param_strings, expected_result):
        """Test parsing null/none values."""
        result = parse_param_overrides(param_strings)
        assert result == expected_result

    @pytest.mark.parametrize(
        "param_strings,expected_result",
        [
            (["list1=[1,2,3]"], {"list1": [1, 2, 3]}),
            (["list2=[a,b,c]"], {"list2": ["a", "b", "c"]}),
            (["empty_list=[]"], {"empty_list": []}),
            (["mixed=[1,test,true]"], {"mixed": [1, "test", True]}),
        ],
    )
    def test_parse_list_values(self, param_strings, expected_result):
        """Test parsing list parameter values."""
        result = parse_param_overrides(param_strings)
        assert result == expected_result

    @pytest.mark.parametrize(
        "param_strings,expected_result",
        [
            (["empty_param="], {"empty_param": ""}),
            (
                ["url=http://example.com?param=value"],
                {"url": "http://example.com?param=value"},
            ),
            (["equation=x=y+z"], {"equation": "x=y+z"}),
        ],
    )
    def test_parse_special_values(self, param_strings, expected_result):
        """Test parsing special values including empty and values with equals."""
        result = parse_param_overrides(param_strings)
        assert result == expected_result

    @pytest.mark.parametrize(
        "invalid_params,error_pattern",
        [
            (["invalid_param_without_equals"], "Invalid parameter format"),
            (["=value"], "Empty parameter key"),
            ([""], "Invalid parameter format"),
        ],
    )
    def test_invalid_param_format(self, invalid_params, error_pattern):
        """Test invalid parameter format raises ConfigError."""
        with pytest.raises(ConfigError, match=error_pattern):
            parse_param_overrides(invalid_params)


class TestMergeConfigs:
    """Test merge_configs function."""

    @pytest.mark.parametrize(
        "base_config_type,override_params,expected_overrides",
        [
            (
                "ml_training",
                {"learning_rate": 0.001, "new_param": "test"},
                {"learning_rate": 0.001, "new_param": "test"},
            ),
            (
                "simple",
                {"param1": "overridden", "param4": "new"},
                {"param1": "overridden", "param4": "new"},
            ),
        ],
    )
    def test_merge_simple_configs(
        self, base_config_type, override_params, expected_overrides
    ):
        """Test merging simple configurations."""
        base = TestDataFactory.create_experiment_config(config_type=base_config_type)
        override = override_params

        result = merge_configs(base, override)

        # Check that all override values are present
        for key, value in expected_overrides.items():
            assert result[key] == value

        # Check that non-overridden base values are preserved
        for key, value in base.items():
            if key not in override:
                assert result[key] == value

    def test_merge_nested_configs(self):
        """Test merging nested configurations."""
        base = {
            "model": {"learning_rate": 0.01, "batch_size": 32},
            "data": {"path": "/data"},
        }
        override = {
            "model": {"learning_rate": 0.001},  # partial override
            "training": {"epochs": 100},  # new section
        }

        result = merge_configs(base, override)

        assert result == {
            "model": {"learning_rate": 0.001, "batch_size": 32},
            "data": {"path": "/data"},
            "training": {"epochs": 100},
        }

    def test_merge_does_not_modify_originals(self):
        """Test that merge doesn't modify original configs."""
        base = TestDataFactory.create_experiment_config(config_type="simple")
        override = {"new_param": "value"}

        original_base = base.copy()
        original_override = override.copy()

        result = merge_configs(base, override)

        assert base == original_base
        assert override == original_override
        assert "new_param" in result
        assert result["new_param"] == "value"

    @pytest.mark.parametrize(
        "base_section,override_value,expected_type",
        [
            ({"param1": "value1", "param2": "value2"}, "simple_value", str),
            ({"nested": {"deep": "value"}}, 123, int),
            ({"complex": {"structure": [1, 2, 3]}}, None, type(None)),
        ],
    )
    def test_merge_override_nested_with_non_dict(
        self, base_section, override_value, expected_type
    ):
        """Test overriding nested dict with non-dict value."""
        base = {"section": base_section}
        override = {"section": override_value}

        result = merge_configs(base, override)

        assert result == {"section": override_value}
        assert isinstance(result["section"], expected_type)


class TestResolveConfig:
    """Test resolve_config function."""

    @pytest.mark.parametrize(
        "config_type",
        [
            "ml_training",
            "data_processing",
            "simple",
        ],
    )
    def test_resolve_with_config_file(self, temp_dir, config_type):
        """Test resolving config with specified file."""
        config_path = temp_dir / "config.yaml"
        config_data = TestDataFactory.create_experiment_config(config_type=config_type)
        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        experiment_config, cli_defaults = resolve_config(config_paths=(config_path,))

        assert experiment_config == config_data
        assert cli_defaults == {}

    def test_resolve_with_default_config(self, temp_dir):
        """Test resolving config with default config file."""
        config_data = TestDataFactory.create_experiment_config(
            config_type="ml_training"
        )
        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        with patch("yanex.core.config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir
            experiment_config, cli_defaults = resolve_config()

        assert experiment_config == config_data
        assert cli_defaults == {}

    def test_resolve_no_config_file(self, temp_dir):
        """Test resolving config when no file exists."""
        with patch("yanex.core.config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir
            experiment_config, cli_defaults = resolve_config()

        assert experiment_config == {}
        assert cli_defaults == {}

    @pytest.mark.parametrize(
        "base_config_type,param_overrides,expected_overrides",
        [
            (
                "ml_training",
                ["learning_rate=0.001", "new_param=test"],
                {"learning_rate": 0.001, "new_param": "test"},
            ),
            (
                "simple",
                ["param1=overridden", "param4=new"],
                {"param1": "overridden", "param4": "new"},
            ),
        ],
    )
    def test_resolve_with_param_overrides(
        self, temp_dir, base_config_type, param_overrides, expected_overrides
    ):
        """Test resolving config with parameter overrides."""
        config_path = temp_dir / "config.yaml"
        config_data = TestDataFactory.create_experiment_config(
            config_type=base_config_type
        )
        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        experiment_config, cli_defaults = resolve_config(
            config_paths=(config_path,), param_overrides=param_overrides
        )

        # Check that overrides are applied
        for key, value in expected_overrides.items():
            assert experiment_config[key] == value

        # Check that non-overridden values are preserved
        for key, value in config_data.items():
            if key not in expected_overrides:
                assert experiment_config[key] == value

        assert cli_defaults == {}

    @pytest.mark.parametrize(
        "param_overrides,expected_result",
        [
            (["param1=value1", "param2=123"], {"param1": "value1", "param2": 123}),
            (
                ["learning_rate=0.01", "epochs=100"],
                {"learning_rate": 0.01, "epochs": 100},
            ),
        ],
    )
    def test_resolve_only_param_overrides(self, param_overrides, expected_result):
        """Test resolving config with only parameter overrides."""
        experiment_config, cli_defaults = resolve_config(
            param_overrides=param_overrides
        )
        assert experiment_config == expected_result
        assert cli_defaults == {}

    @pytest.mark.parametrize(
        "custom_name,config_type",
        [
            ("custom.yaml", "ml_training"),
            ("experiment.yml", "data_processing"),
            ("settings.yaml", "simple"),
        ],
    )
    def test_resolve_custom_default_config_name(
        self, temp_dir, custom_name, config_type
    ):
        """Test resolving with custom default config name."""
        config_data = TestDataFactory.create_experiment_config(config_type=config_type)
        TestFileHelpers.create_config_file(temp_dir, config_data, custom_name)

        with patch("yanex.core.config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir
            experiment_config, cli_defaults = resolve_config(
                default_config_name=custom_name
            )

        assert experiment_config == config_data
        assert cli_defaults == {}

    def test_resolve_comprehensive_workflow(self, temp_dir):
        """Test comprehensive config resolution workflow."""
        # Create base config
        base_config = TestDataFactory.create_experiment_config(
            config_type="ml_training",
            learning_rate=0.01,
            epochs=100,
        )
        config_path = temp_dir / "experiment.yaml"
        TestFileHelpers.create_config_file(temp_dir, base_config, "experiment.yaml")

        # Apply overrides
        param_overrides = [
            "learning_rate=0.001",  # Override existing
            "batch_size=64",  # Add new
            "model.type=transformer",  # Add nested
        ]

        experiment_config, cli_defaults = resolve_config(
            config_paths=(config_path,), param_overrides=param_overrides
        )

        # Verify overrides applied
        assert experiment_config["learning_rate"] == 0.001
        assert experiment_config["batch_size"] == 64
        assert experiment_config["model"]["type"] == "transformer"

        # Verify original values preserved where not overridden
        assert experiment_config["epochs"] == 100

        # Verify structure is complete
        assert isinstance(experiment_config, dict)
        assert len(experiment_config) >= 4  # At least the added/modified keys
        assert cli_defaults == {}

    def test_resolve_with_cli_defaults(self, temp_dir):
        """Test resolving config with yanex CLI defaults section."""
        config_data = {
            "learning_rate": 0.01,
            "epochs": 100,
            "yanex": {
                "name": "test-experiment",
                "tag": ["ml", "testing"],
                "description": "Test description",
                "dry_run": False,
                "stage": False,
            },
        }
        config_path = temp_dir / "config.yaml"
        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        experiment_config, cli_defaults = resolve_config(config_paths=(config_path,))

        # Verify experiment config doesn't contain yanex section
        expected_experiment_config = {"learning_rate": 0.01, "epochs": 100}
        assert experiment_config == expected_experiment_config

        # Verify CLI defaults extracted correctly
        expected_cli_defaults = {
            "name": "test-experiment",
            "tag": ["ml", "testing"],
            "description": "Test description",
            "dry_run": False,
            "stage": False,
        }
        assert cli_defaults == expected_cli_defaults

    def test_resolve_with_cli_defaults_and_overrides(self, temp_dir):
        """Test config with CLI defaults and parameter overrides."""
        config_data = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "yanex": {"name": "config-experiment", "tag": "config"},
        }
        config_path = temp_dir / "config.yaml"
        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        param_overrides = ["learning_rate=0.001", "epochs=200"]

        experiment_config, cli_defaults = resolve_config(
            config_paths=(config_path,), param_overrides=param_overrides
        )

        # Verify experiment config has overrides applied, no yanex section
        expected_experiment_config = {
            "learning_rate": 0.001,  # Overridden
            "batch_size": 32,  # Original
            "epochs": 200,  # New
        }
        assert experiment_config == expected_experiment_config

        # Verify CLI defaults unchanged by param overrides
        expected_cli_defaults = {"name": "config-experiment", "tag": "config"}
        assert cli_defaults == expected_cli_defaults

    def test_resolve_empty_yanex_section(self, temp_dir):
        """Test config with empty yanex section."""
        config_data = {
            "learning_rate": 0.01,
            "yanex": {},
        }
        config_path = temp_dir / "config.yaml"
        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        experiment_config, cli_defaults = resolve_config(config_paths=(config_path,))

        assert experiment_config == {"learning_rate": 0.01}
        assert cli_defaults == {}

    def test_resolve_with_multiple_config_files(self, temp_dir):
        """Test resolving config with multiple config files merged in order."""
        # Create first config file (data config)
        data_config = {
            "data": {
                "filename": "my-data.jsonl",
                "split": 0.2,
            }
        }
        data_config_path = temp_dir / "data-config.yaml"
        TestFileHelpers.create_config_file(temp_dir, data_config, "data-config.yaml")

        # Create second config file (model config)
        model_config = {
            "model": {
                "epochs": 1000,
                "learning_rate": 0.01,
            }
        }
        model_config_path = temp_dir / "model-config.yaml"
        TestFileHelpers.create_config_file(temp_dir, model_config, "model-config.yaml")

        # Resolve with both config files
        experiment_config, cli_defaults = resolve_config(
            config_paths=(data_config_path, model_config_path)
        )

        # Verify both configs are merged
        assert experiment_config["data"]["filename"] == "my-data.jsonl"
        assert experiment_config["data"]["split"] == 0.2
        assert experiment_config["model"]["epochs"] == 1000
        assert experiment_config["model"]["learning_rate"] == 0.01
        assert cli_defaults == {}

    def test_resolve_with_multiple_configs_later_takes_precedence(self, temp_dir):
        """Test that later config files override earlier ones."""
        # Create first config
        config1 = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
        }
        config1_path = temp_dir / "config1.yaml"
        TestFileHelpers.create_config_file(temp_dir, config1, "config1.yaml")

        # Create second config that overrides some values
        config2 = {
            "learning_rate": 0.001,  # Override
            "optimizer": "adam",  # New value
        }
        config2_path = temp_dir / "config2.yaml"
        TestFileHelpers.create_config_file(temp_dir, config2, "config2.yaml")

        # Resolve with both configs
        experiment_config, cli_defaults = resolve_config(
            config_paths=(config1_path, config2_path)
        )

        # Verify later config takes precedence
        assert experiment_config["learning_rate"] == 0.001  # From config2
        assert experiment_config["batch_size"] == 32  # From config1
        assert experiment_config["epochs"] == 100  # From config1
        assert experiment_config["optimizer"] == "adam"  # From config2
        assert cli_defaults == {}

    def test_resolve_multiple_configs_with_cli_defaults(self, temp_dir):
        """Test multiple configs with yanex CLI defaults sections."""
        # First config with CLI defaults
        config1 = {
            "learning_rate": 0.01,
            "yanex": {
                "name": "experiment1",
                "tag": ["ml"],
            },
        }
        config1_path = temp_dir / "config1.yaml"
        TestFileHelpers.create_config_file(temp_dir, config1, "config1.yaml")

        # Second config with different CLI defaults
        config2 = {
            "batch_size": 64,
            "yanex": {
                "name": "experiment2",  # Override name
                "description": "Test experiment",  # Add description
            },
        }
        config2_path = temp_dir / "config2.yaml"
        TestFileHelpers.create_config_file(temp_dir, config2, "config2.yaml")

        # Resolve with both configs
        experiment_config, cli_defaults = resolve_config(
            config_paths=(config1_path, config2_path)
        )

        # Verify experiment config contains both parameters
        assert experiment_config["learning_rate"] == 0.01
        assert experiment_config["batch_size"] == 64
        assert "yanex" not in experiment_config

        # Verify CLI defaults merged, later takes precedence
        assert cli_defaults["name"] == "experiment2"  # From config2
        assert cli_defaults["tag"] == ["ml"]  # From config1
        assert cli_defaults["description"] == "Test experiment"  # From config2

    def test_resolve_multiple_configs_with_param_overrides(self, temp_dir):
        """Test multiple configs with parameter overrides."""
        # First config
        config1 = {
            "learning_rate": 0.01,
            "batch_size": 32,
        }
        config1_path = temp_dir / "config1.yaml"
        TestFileHelpers.create_config_file(temp_dir, config1, "config1.yaml")

        # Second config
        config2 = {
            "epochs": 100,
            "optimizer": "adam",
        }
        config2_path = temp_dir / "config2.yaml"
        TestFileHelpers.create_config_file(temp_dir, config2, "config2.yaml")

        # Parameter overrides
        param_overrides = ["learning_rate=0.001", "new_param=test"]

        # Resolve with both configs and overrides
        experiment_config, cli_defaults = resolve_config(
            config_paths=(config1_path, config2_path),
            param_overrides=param_overrides,
        )

        # Verify all values merged and overrides applied
        assert experiment_config["learning_rate"] == 0.001  # Overridden
        assert experiment_config["batch_size"] == 32  # From config1
        assert experiment_config["epochs"] == 100  # From config2
        assert experiment_config["optimizer"] == "adam"  # From config2
        assert experiment_config["new_param"] == "test"  # From override
        assert cli_defaults == {}

    def test_resolve_three_config_files(self, temp_dir):
        """Test resolving with three config files."""
        # Create three config files
        config1 = {"data": {"filename": "data.jsonl"}}
        config2 = {"yanex": {"scripts": [{"name": "train.py"}]}}
        config3 = {"model": {"epochs": 1000, "learning_rate": 0.01}}

        config1_path = temp_dir / "data.yaml"
        config2_path = temp_dir / "scripts.yaml"
        config3_path = temp_dir / "model.yaml"

        TestFileHelpers.create_config_file(temp_dir, config1, "data.yaml")
        TestFileHelpers.create_config_file(temp_dir, config2, "scripts.yaml")
        TestFileHelpers.create_config_file(temp_dir, config3, "model.yaml")

        # Resolve with all three configs
        experiment_config, cli_defaults = resolve_config(
            config_paths=(config1_path, config2_path, config3_path)
        )

        # Verify all configs merged
        assert experiment_config["data"]["filename"] == "data.jsonl"
        assert experiment_config["model"]["epochs"] == 1000
        assert experiment_config["model"]["learning_rate"] == 0.01
        assert cli_defaults["scripts"] == [{"name": "train.py"}]

    def test_normalize_tags_helper(self):
        """Test _normalize_tags helper function."""
        from yanex.cli.commands.run import _normalize_tags

        # Test string input
        assert _normalize_tags("single-tag") == ["single-tag"]

        # Test list input
        assert _normalize_tags(["tag1", "tag2"]) == ["tag1", "tag2"]

        # Test empty input
        assert _normalize_tags([]) == []
        assert _normalize_tags(None) == []

        # Test mixed types in list
        assert _normalize_tags([1, "tag", True]) == ["1", "tag", "True"]
