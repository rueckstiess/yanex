"""
Tests for yanex.core.config module.
"""

from unittest.mock import patch

import pytest
import yaml

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

    def test_load_valid_config(self, temp_dir):
        """Test loading valid YAML configuration."""
        config_path = temp_dir / "config.yaml"
        config_data = {"param1": "value1", "param2": 123, "nested": {"param3": True}}

        config_path.write_text(yaml.safe_dump(config_data))

        result = load_yaml_config(config_path)
        assert result == config_data

    def test_load_empty_config(self, temp_dir):
        """Test loading empty YAML file."""
        config_path = temp_dir / "empty.yaml"
        config_path.write_text("")

        result = load_yaml_config(config_path)
        assert result == {}

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

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML raises ConfigError."""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(ConfigError, match="Failed to parse YAML config"):
            load_yaml_config(config_path)

    def test_load_non_dict_yaml(self, temp_dir):
        """Test loading non-dictionary YAML raises ConfigError."""
        config_path = temp_dir / "list.yaml"
        config_path.write_text("- item1\n- item2")

        with pytest.raises(ConfigError, match="Configuration must be a dictionary"):
            load_yaml_config(config_path)


class TestSaveYamlConfig:
    """Test save_yaml_config function."""

    def test_save_valid_config(self, temp_dir):
        """Test saving valid configuration."""
        config_path = temp_dir / "output.yaml"
        config_data = {"param1": "value1", "param2": 123, "nested": {"param3": True}}

        save_yaml_config(config_data, config_path)

        assert config_path.exists()

        # Load back and verify
        with config_path.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded == config_data

    def test_save_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        config_path = temp_dir / "subdir" / "config.yaml"
        config_data = {"test": "value"}

        save_yaml_config(config_data, config_path)

        assert config_path.exists()
        assert config_path.parent.is_dir()

    def test_save_invalid_config_data(self, temp_dir):
        """Test saving invalid config data raises ValidationError."""
        config_path = temp_dir / "output.yaml"

        from yanex.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Configuration must be a dictionary"):
            save_yaml_config("not a dict", config_path)


class TestParseParamOverrides:
    """Test parse_param_overrides function."""

    def test_parse_simple_params(self):
        """Test parsing simple parameter overrides."""
        param_strings = ["param1=value1", "param2=123", "param3=true"]

        result = parse_param_overrides(param_strings)

        assert result == {
            "param1": "value1",
            "param2": 123,
            "param3": True,
        }

    def test_parse_nested_params(self):
        """Test parsing nested parameter overrides."""
        param_strings = ["model.learning_rate=0.01", "model.batch_size=32"]

        result = parse_param_overrides(param_strings)

        assert result == {
            "model": {
                "learning_rate": 0.01,
                "batch_size": 32,
            }
        }

    def test_parse_boolean_values(self):
        """Test parsing boolean parameter values."""
        param_strings = [
            "flag1=true",
            "flag2=false",
            "flag3=yes",
            "flag4=no",
            "flag5=1",
            "flag6=0",
            "flag7=on",
            "flag8=off",
        ]

        result = parse_param_overrides(param_strings)

        assert result == {
            "flag1": True,
            "flag2": False,
            "flag3": True,
            "flag4": False,
            "flag5": True,
            "flag6": False,
            "flag7": True,
            "flag8": False,
        }

    def test_parse_numeric_values(self):
        """Test parsing numeric parameter values."""
        param_strings = ["int_param=42", "float_param=3.14", "exp_param=1e-5"]

        result = parse_param_overrides(param_strings)

        assert result == {
            "int_param": 42,
            "float_param": 3.14,
            "exp_param": 1e-5,
        }

    def test_parse_null_values(self):
        """Test parsing null/none values."""
        param_strings = ["param1=null", "param2=none", "param3=~"]

        result = parse_param_overrides(param_strings)

        assert result == {
            "param1": None,
            "param2": None,
            "param3": None,
        }

    def test_parse_list_values(self):
        """Test parsing list parameter values."""
        param_strings = ["list1=[1,2,3]", "list2=[a,b,c]", "empty_list=[]"]

        result = parse_param_overrides(param_strings)

        assert result == {
            "list1": [1, 2, 3],
            "list2": ["a", "b", "c"],
            "empty_list": [],
        }

    def test_parse_empty_value(self):
        """Test parsing empty parameter values."""
        param_strings = ["empty_param="]

        result = parse_param_overrides(param_strings)

        assert result == {"empty_param": ""}

    def test_parse_value_with_equals(self):
        """Test parsing values that contain equals signs."""
        param_strings = ["url=http://example.com?param=value"]

        result = parse_param_overrides(param_strings)

        assert result == {"url": "http://example.com?param=value"}

    def test_invalid_param_format(self):
        """Test invalid parameter format raises ConfigError."""
        with pytest.raises(ConfigError, match="Invalid parameter format"):
            parse_param_overrides(["invalid_param_without_equals"])

    def test_empty_param_key(self):
        """Test empty parameter key raises ConfigError."""
        with pytest.raises(ConfigError, match="Empty parameter key"):
            parse_param_overrides(["=value"])


class TestMergeConfigs:
    """Test merge_configs function."""

    def test_merge_simple_configs(self):
        """Test merging simple configurations."""
        base = {"param1": "base_value", "param2": 123}
        override = {"param2": 456, "param3": "new_value"}

        result = merge_configs(base, override)

        assert result == {
            "param1": "base_value",
            "param2": 456,  # overridden
            "param3": "new_value",
        }

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
        base = {"param1": "value1"}
        override = {"param2": "value2"}

        original_base = base.copy()
        original_override = override.copy()

        result = merge_configs(base, override)

        assert base == original_base
        assert override == original_override
        assert result == {"param1": "value1", "param2": "value2"}

    def test_merge_override_nested_with_non_dict(self):
        """Test overriding nested dict with non-dict value."""
        base = {"section": {"param1": "value1", "param2": "value2"}}
        override = {"section": "simple_value"}

        result = merge_configs(base, override)

        assert result == {"section": "simple_value"}


class TestResolveConfig:
    """Test resolve_config function."""

    def test_resolve_with_config_file(self, temp_dir):
        """Test resolving config with specified file."""
        config_path = temp_dir / "config.yaml"
        config_data = {"param1": "value1", "param2": 123}
        config_path.write_text(yaml.safe_dump(config_data))

        result = resolve_config(config_path=config_path)

        assert result == config_data

    def test_resolve_with_default_config(self, temp_dir):
        """Test resolving config with default config file."""
        config_path = temp_dir / "config.yaml"
        config_data = {"param1": "value1"}
        config_path.write_text(yaml.safe_dump(config_data))

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

    def test_resolve_with_param_overrides(self, temp_dir):
        """Test resolving config with parameter overrides."""
        config_path = temp_dir / "config.yaml"
        config_data = {"param1": "original", "param2": 123}
        config_path.write_text(yaml.safe_dump(config_data))

        param_overrides = ["param1=overridden", "param3=new_value"]

        result = resolve_config(
            config_path=config_path, param_overrides=param_overrides
        )

        assert result == {
            "param1": "overridden",
            "param2": 123,
            "param3": "new_value",
        }

    def test_resolve_only_param_overrides(self):
        """Test resolving config with only parameter overrides."""
        param_overrides = ["param1=value1", "param2=123"]

        result = resolve_config(param_overrides=param_overrides)

        assert result == {"param1": "value1", "param2": 123}

    def test_resolve_custom_default_config_name(self, temp_dir):
        """Test resolving with custom default config name."""
        config_path = temp_dir / "custom.yaml"
        config_data = {"param1": "value1"}
        config_path.write_text(yaml.safe_dump(config_data))

        with patch("yanex.core.config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir
            result = resolve_config(default_config_name="custom.yaml")

        assert result == config_data
