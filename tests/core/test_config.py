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

        result = resolve_config(config_path=config_path)

        assert result == config_data

    def test_resolve_with_default_config(self, temp_dir):
        """Test resolving config with default config file."""
        config_data = TestDataFactory.create_experiment_config(
            config_type="ml_training"
        )
        TestFileHelpers.create_config_file(temp_dir, config_data, "config.yaml")

        with patch("yanex.core.config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir
            result = resolve_config()

        assert result == config_data

    def test_resolve_no_config_file(self, temp_dir):
        """Test resolving config when no file exists."""
        with patch("yanex.core.config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir
            result = resolve_config()

        assert result == {}

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

        result = resolve_config(
            config_path=config_path, param_overrides=param_overrides
        )

        # Check that overrides are applied
        for key, value in expected_overrides.items():
            assert result[key] == value

        # Check that non-overridden values are preserved
        for key, value in config_data.items():
            if key not in expected_overrides:
                assert result[key] == value

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
        result = resolve_config(param_overrides=param_overrides)
        assert result == expected_result

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
            result = resolve_config(default_config_name=custom_name)

        assert result == config_data

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

        result = resolve_config(
            config_path=config_path, param_overrides=param_overrides
        )

        # Verify overrides applied
        assert result["learning_rate"] == 0.001
        assert result["batch_size"] == 64
        assert result["model"]["type"] == "transformer"

        # Verify original values preserved where not overridden
        assert result["epochs"] == 100

        # Verify structure is complete
        assert isinstance(result, dict)
        assert len(result) >= 4  # At least the added/modified keys
