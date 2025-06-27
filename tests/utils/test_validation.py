"""
Tests for yanex.utils.validation module.
"""

import pytest
from pathlib import Path

from yanex.utils.validation import (
    validate_experiment_name,
    validate_experiment_id,
    validate_tags,
    validate_script_path,
    validate_config_data,
)
from yanex.utils.exceptions import ValidationError


class TestValidateExperimentName:
    """Test experiment name validation."""

    def test_valid_names(self):
        """Test valid experiment names."""
        valid_names = [
            "test_experiment",
            "experiment-123",
            "My Experiment",
            "simple",
            "test_exp_with_numbers_123",
            "A-B_C 123",
        ]

        for name in valid_names:
            result = validate_experiment_name(name)
            assert result == name.strip()

    def test_name_stripping(self):
        """Test that names are properly stripped."""
        assert validate_experiment_name("  test  ") == "test"
        assert validate_experiment_name("\ttest\n") == "test"

    def test_empty_name(self):
        """Test empty names raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_experiment_name("")

        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_experiment_name("   ")

        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_experiment_name(None)

    def test_name_too_long(self):
        """Test names that are too long."""
        long_name = "a" * 101
        with pytest.raises(ValidationError, match="cannot exceed 100 characters"):
            validate_experiment_name(long_name)

    def test_invalid_characters(self):
        """Test names with invalid characters."""
        invalid_names = [
            "test@experiment",
            "test#experiment",
            "test$experiment",
            "test/experiment",
            "test\\experiment",
            "test.experiment",
            "test,experiment",
        ]

        for name in invalid_names:
            with pytest.raises(ValidationError, match="can only contain"):
                validate_experiment_name(name)


class TestValidateExperimentId:
    """Test experiment ID validation."""

    def test_valid_ids(self):
        """Test valid experiment IDs."""
        valid_ids = [
            "12345678",
            "abcdef12",
            "a1b2c3d4",
            "00000000",
            "ffffffff",
        ]

        for exp_id in valid_ids:
            result = validate_experiment_id(exp_id)
            assert result == exp_id

    def test_invalid_id_length(self):
        """Test IDs with invalid length."""
        invalid_ids = [
            "1234567",  # too short
            "123456789",  # too long
            "",  # empty
            "abc",  # too short
        ]

        for exp_id in invalid_ids:
            with pytest.raises(ValidationError, match="Invalid experiment ID format"):
                validate_experiment_id(exp_id)

    def test_invalid_id_characters(self):
        """Test IDs with invalid characters."""
        invalid_ids = [
            "1234567g",  # invalid hex character
            "ABCDEF12",  # uppercase not allowed
            "1234-567",  # hyphen not allowed
            "12345678 ",  # space not allowed
        ]

        for exp_id in invalid_ids:
            with pytest.raises(ValidationError, match="Invalid experiment ID format"):
                validate_experiment_id(exp_id)


class TestValidateTags:
    """Test tag validation."""

    def test_valid_tags(self):
        """Test valid tag lists."""
        valid_tag_lists = [
            ["test", "experiment"],
            ["ml", "pytorch", "vision"],
            ["tag_with_underscores"],
            ["tag-with-hyphens"],
            ["tag123"],
            [],  # empty list is valid
        ]

        for tags in valid_tag_lists:
            result = validate_tags(tags)
            assert result == tags

    def test_tag_stripping_and_filtering(self):
        """Test that tags are stripped and empty ones filtered."""
        input_tags = ["  tag1  ", "", "tag2", "   ", "tag3"]
        result = validate_tags(input_tags)
        assert result == ["tag1", "tag2", "tag3"]

    def test_invalid_tag_type(self):
        """Test that non-list input raises ValidationError."""
        with pytest.raises(ValidationError, match="Tags must be a list"):
            validate_tags("not_a_list")

    def test_non_string_tags(self):
        """Test that non-string tags raise ValidationError."""
        with pytest.raises(ValidationError, match="Tag must be a string"):
            validate_tags([123, "valid_tag"])

        with pytest.raises(ValidationError, match="Tag must be a string"):
            validate_tags(["valid_tag", None])

    def test_tag_too_long(self):
        """Test that overly long tags raise ValidationError."""
        long_tag = "a" * 51
        with pytest.raises(ValidationError, match="Tag too long"):
            validate_tags([long_tag])

    def test_invalid_tag_characters(self):
        """Test that tags with invalid characters raise ValidationError."""
        invalid_tags = [
            "tag with spaces",
            "tag@symbol",
            "tag.dot",
            "tag/slash",
        ]

        for tag in invalid_tags:
            with pytest.raises(
                ValidationError, match="Tag contains invalid characters"
            ):
                validate_tags([tag])


class TestValidateScriptPath:
    """Test script path validation."""

    def test_valid_script_path(self, temp_dir):
        """Test valid Python script path."""
        script_path = temp_dir / "test_script.py"
        script_path.write_text("print('hello')")

        result = validate_script_path(script_path)
        assert result == script_path

    def test_nonexistent_script(self, temp_dir):
        """Test nonexistent script raises ValidationError."""
        script_path = temp_dir / "nonexistent.py"

        with pytest.raises(ValidationError, match="Script file does not exist"):
            validate_script_path(script_path)

    def test_script_path_is_directory(self, temp_dir):
        """Test directory path raises ValidationError."""
        dir_path = temp_dir / "script_dir"
        dir_path.mkdir()

        with pytest.raises(ValidationError, match="Script path is not a file"):
            validate_script_path(dir_path)

    def test_non_python_script(self, temp_dir):
        """Test non-Python script raises ValidationError."""
        script_path = temp_dir / "script.txt"
        script_path.write_text("not python")

        with pytest.raises(ValidationError, match="Script must be a Python file"):
            validate_script_path(script_path)


class TestValidateConfigData:
    """Test configuration data validation."""

    def test_valid_config_data(self):
        """Test valid configuration dictionaries."""
        valid_configs = [
            {},
            {"param1": "value1"},
            {"param1": 123, "param2": 45.6},
            {"param1": True, "param2": False},
            {"param1": None},
            {"param1": [1, 2, 3]},
            {"param1": {"nested": "value"}},
            {"complex": {"nested": {"values": [1, 2, {"deep": True}]}}},
        ]

        for config in valid_configs:
            result = validate_config_data(config)
            assert result == config

    def test_non_dict_config(self):
        """Test non-dictionary config raises ValidationError."""
        invalid_configs = [
            "string",
            123,
            [1, 2, 3],
            None,
            True,
        ]

        for config in invalid_configs:
            with pytest.raises(
                ValidationError, match="Configuration must be a dictionary"
            ):
                validate_config_data(config)

    def test_invalid_value_types(self):
        """Test invalid value types raise ValidationError."""

        class CustomClass:
            pass

        invalid_configs = [
            {"param": CustomClass()},
            {"param": object()},
            {"param": lambda x: x},
            {"nested": {"param": set([1, 2, 3])}},
        ]

        for config in invalid_configs:
            with pytest.raises(ValidationError, match="must be JSON serializable"):
                validate_config_data(config)
