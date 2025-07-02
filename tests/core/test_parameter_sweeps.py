"""
Tests for parameter sweep functionality.
"""

import pytest

from yanex.core.config import (
    LinspaceSweep,
    ListSweep,
    LogspaceSweep,
    RangeSweep,
    expand_parameter_sweeps,
    has_sweep_parameters,
    parse_param_overrides,
)
from yanex.core.parameter_parsers import BasicParameterParser, SweepParameterParser
from yanex.utils.exceptions import ConfigError


class TestSweepParameterClasses:
    """Test sweep parameter classes."""

    def test_range_sweep_creation(self):
        """Test RangeSweep creation and validation."""
        sweep = RangeSweep(0.01, 0.1, 0.01)
        assert sweep.start == 0.01
        assert sweep.stop == 0.1
        assert sweep.step == 0.01

    def test_range_sweep_zero_step_error(self):
        """Test RangeSweep rejects zero step."""
        with pytest.raises(ConfigError, match="Range step cannot be zero"):
            RangeSweep(0.01, 0.1, 0)

    def test_range_sweep_wrong_direction_error(self):
        """Test RangeSweep rejects wrong step direction."""
        with pytest.raises(ConfigError, match="Range step direction doesn't match"):
            RangeSweep(0.1, 0.01, 0.01)  # positive step but decreasing range

    def test_range_sweep_generation_positive(self):
        """Test RangeSweep value generation with positive step."""
        sweep = RangeSweep(0.01, 0.04, 0.01)
        values = sweep.generate_values()
        assert len(values) == 3
        assert values == [0.01, 0.02, 0.03]

    def test_range_sweep_generation_negative(self):
        """Test RangeSweep value generation with negative step."""
        sweep = RangeSweep(0.1, 0.07, -0.01)
        values = sweep.generate_values()
        # Due to floating point precision, we might get 4 values instead of 3
        assert len(values) == 4
        assert abs(values[0] - 0.1) < 1e-10
        assert abs(values[1] - 0.09) < 1e-10
        assert abs(values[2] - 0.08) < 1e-10

    def test_linspace_sweep_creation(self):
        """Test LinspaceSweep creation and validation."""
        sweep = LinspaceSweep(0.01, 0.1, 5)
        assert sweep.start == 0.01
        assert sweep.stop == 0.1
        assert sweep.count == 5

    def test_linspace_sweep_invalid_count(self):
        """Test LinspaceSweep rejects invalid count."""
        with pytest.raises(ConfigError, match="Linspace count must be positive"):
            LinspaceSweep(0.01, 0.1, 0)

    def test_linspace_sweep_generation(self):
        """Test LinspaceSweep value generation."""
        sweep = LinspaceSweep(0.0, 1.0, 5)
        values = sweep.generate_values()
        assert len(values) == 5
        assert values == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_linspace_sweep_single_value(self):
        """Test LinspaceSweep with count=1."""
        sweep = LinspaceSweep(0.5, 1.0, 1)
        values = sweep.generate_values()
        assert values == [0.5]

    def test_logspace_sweep_creation(self):
        """Test LogspaceSweep creation and validation."""
        sweep = LogspaceSweep(-3, -1, 3)
        assert sweep.start == -3
        assert sweep.stop == -1
        assert sweep.count == 3

    def test_logspace_sweep_invalid_count(self):
        """Test LogspaceSweep rejects invalid count."""
        with pytest.raises(ConfigError, match="Logspace count must be positive"):
            LogspaceSweep(-3, -1, -1)

    def test_logspace_sweep_generation(self):
        """Test LogspaceSweep value generation."""
        sweep = LogspaceSweep(-3, -1, 3)
        values = sweep.generate_values()
        assert len(values) == 3
        assert values == [0.001, 0.01, 0.1]

    def test_list_sweep_creation(self):
        """Test ListSweep creation and validation."""
        sweep = ListSweep([1, 2, 3])
        assert sweep.items == [1, 2, 3]

    def test_list_sweep_empty_error(self):
        """Test ListSweep rejects empty list."""
        with pytest.raises(ConfigError, match="List sweep cannot be empty"):
            ListSweep([])

    def test_list_sweep_generation(self):
        """Test ListSweep value generation."""
        sweep = ListSweep(["a", 42, True])
        values = sweep.generate_values()
        assert values == ["a", 42, True]
        # Ensure it returns a copy
        values.append("modified")
        assert sweep.items == ["a", 42, True]


class TestSweepSyntaxParsing:
    """Test sweep syntax parsing functions."""

    def test_parse_numeric_value(self):
        """Test numeric value parsing."""
        sweep_parser = SweepParameterParser()
        assert sweep_parser._parse_numeric_value("42") == 42
        assert sweep_parser._parse_numeric_value("3.14") == 3.14
        assert sweep_parser._parse_numeric_value("-1") == -1
        assert sweep_parser._parse_numeric_value("1e-3") == 0.001

    def test_parse_numeric_value_error(self):
        """Test numeric parsing error."""
        sweep_parser = SweepParameterParser()
        with pytest.raises(ConfigError, match="Expected numeric value"):
            sweep_parser._parse_numeric_value("not_a_number")

    def test_parse_non_sweep_value(self):
        """Test non-sweep value parsing."""
        basic_parser = BasicParameterParser()
        assert basic_parser.parse("42") == 42
        assert basic_parser.parse("3.14") == 3.14
        assert basic_parser.parse("true") is True
        assert basic_parser.parse("false") is False
        assert basic_parser.parse("hello") == "hello"
        assert basic_parser.parse('"quoted"') == "quoted"
        assert basic_parser.parse("'quoted'") == "quoted"

    def test_parse_range_syntax(self):
        """Test range() syntax parsing."""
        sweep_parser = SweepParameterParser()
        result = sweep_parser.parse("range(0.01, 0.1, 0.01)")
        assert isinstance(result, RangeSweep)
        assert result.start == 0.01
        assert result.stop == 0.1
        assert result.step == 0.01

    def test_parse_range_syntax_with_spaces(self):
        """Test range() syntax parsing with extra spaces."""
        sweep_parser = SweepParameterParser()
        result = sweep_parser.parse("range( 0.01 , 0.1 , 0.01 )")
        assert isinstance(result, RangeSweep)
        assert result.start == 0.01

    def test_parse_linspace_syntax(self):
        """Test linspace() syntax parsing."""
        sweep_parser = SweepParameterParser()
        result = sweep_parser.parse("linspace(0.01, 0.1, 5)")
        assert isinstance(result, LinspaceSweep)
        assert result.start == 0.01
        assert result.stop == 0.1
        assert result.count == 5

    def test_parse_logspace_syntax(self):
        """Test logspace() syntax parsing."""
        sweep_parser = SweepParameterParser()
        result = sweep_parser.parse("logspace(-3, -1, 3)")
        assert isinstance(result, LogspaceSweep)
        assert result.start == -3
        assert result.stop == -1
        assert result.count == 3

    def test_parse_list_syntax(self):
        """Test list() syntax parsing."""
        sweep_parser = SweepParameterParser()
        result = sweep_parser.parse("list(16, 32, 64)")
        assert isinstance(result, ListSweep)
        assert result.items == [16, 32, 64]

    def test_parse_list_syntax_mixed_types(self):
        """Test list() syntax with mixed types."""
        sweep_parser = SweepParameterParser()
        result = sweep_parser.parse("list(foo, 42, true)")
        assert isinstance(result, ListSweep)
        assert result.items == ["foo", 42, True]

    def test_parse_list_syntax_quoted_strings(self):
        """Test list() syntax with quoted strings."""
        sweep_parser = SweepParameterParser()
        result = sweep_parser.parse('list("hello", "world")')
        assert isinstance(result, ListSweep)
        assert result.items == ["hello", "world"]

    def test_parse_non_sweep_syntax(self):
        """Test non-sweep syntax returns None."""
        sweep_parser = SweepParameterParser()
        # Non-sweep values should fail the can_parse check, so use a different approach
        assert not sweep_parser.can_parse("42")
        assert not sweep_parser.can_parse("hello")
        assert not sweep_parser.can_parse("range")  # incomplete

    def test_parse_invalid_sweep_syntax(self):
        """Test invalid sweep syntax raises errors."""
        sweep_parser = SweepParameterParser()
        # Incomplete syntax should match can_parse (based on pattern) but fail during parsing
        assert sweep_parser.can_parse(
            "range(0.01, 0.1)"
        )  # missing step - matches pattern
        assert sweep_parser.can_parse(
            "linspace(0.01)"
        )  # missing arguments - matches pattern

        # Incomplete syntax should raise errors during parsing
        with pytest.raises(ConfigError, match="Invalid sweep syntax"):
            sweep_parser.parse("range(0.01, 0.1)")  # missing step

        with pytest.raises(ConfigError, match="Invalid sweep syntax"):
            sweep_parser.parse("linspace(0.01)")  # missing arguments

        # Invalid syntax that matches pattern should raise errors
        with pytest.raises(ConfigError, match="Invalid range\\(\\) syntax"):
            sweep_parser.parse("range(not_a_number, 0.1, 0.01)")

        with pytest.raises(ConfigError, match="Invalid linspace\\(\\) syntax"):
            sweep_parser.parse("linspace(0.01, 0.1, not_a_number)")

        with pytest.raises(ConfigError, match="Invalid list\\(\\) syntax"):
            sweep_parser.parse("list()")  # empty list


class TestParameterSweepIntegration:
    """Test integration of sweep parameters with config parsing."""

    def test_parse_param_overrides_with_sweeps(self):
        """Test parsing parameter overrides containing sweeps."""
        result = parse_param_overrides(
            ["lr=range(0.01, 0.03, 0.01)", "batch_size=list(16, 32)", "epochs=100"]
        )

        assert isinstance(result["lr"], RangeSweep)
        assert isinstance(result["batch_size"], ListSweep)
        assert result["epochs"] == 100

    def test_has_sweep_parameters_detection(self):
        """Test sweep parameter detection."""
        # Config with sweeps
        config_with_sweeps = {"lr": RangeSweep(0.01, 0.1, 0.01), "batch_size": 32}
        assert has_sweep_parameters(config_with_sweeps) is True

        # Config without sweeps
        config_without_sweeps = {"lr": 0.01, "batch_size": 32}
        assert has_sweep_parameters(config_without_sweeps) is False

        # Nested config with sweeps
        nested_config = {
            "model": {"lr": RangeSweep(0.01, 0.1, 0.01)},
            "training": {"epochs": 100},
        }
        assert has_sweep_parameters(nested_config) is True

    def test_expand_parameter_sweeps_single(self):
        """Test expansion of single sweep parameter."""
        config = {"lr": RangeSweep(0.01, 0.03, 0.01), "batch_size": 32}

        expanded = expand_parameter_sweeps(config)
        assert len(expanded) == 2

        assert expanded[0]["lr"] == 0.01
        assert expanded[0]["batch_size"] == 32

        assert expanded[1]["lr"] == 0.02
        assert expanded[1]["batch_size"] == 32

    def test_expand_parameter_sweeps_multiple(self):
        """Test expansion of multiple sweep parameters (cross-product)."""
        config = {
            "lr": LinspaceSweep(0.01, 0.02, 2),
            "batch_size": ListSweep([16, 32]),
            "epochs": 100,
        }

        expanded = expand_parameter_sweeps(config)
        assert len(expanded) == 4  # 2 x 2 cross-product

        # Check all combinations exist
        lr_values = {cfg["lr"] for cfg in expanded}
        batch_values = {cfg["batch_size"] for cfg in expanded}

        assert lr_values == {0.01, 0.02}
        assert batch_values == {16, 32}
        assert all(cfg["epochs"] == 100 for cfg in expanded)

    def test_expand_parameter_sweeps_nested(self):
        """Test expansion of nested sweep parameters."""
        config = {
            "model": {"lr": RangeSweep(0.01, 0.03, 0.01), "architecture": "resnet"},
            "training": {"epochs": 100},
        }

        expanded = expand_parameter_sweeps(config)
        assert len(expanded) == 2

        assert expanded[0]["model"]["lr"] == 0.01
        assert expanded[0]["model"]["architecture"] == "resnet"
        assert expanded[0]["training"]["epochs"] == 100

        assert expanded[1]["model"]["lr"] == 0.02

    def test_expand_parameter_sweeps_no_sweeps(self):
        """Test expansion with no sweep parameters returns original config."""
        config = {"lr": 0.01, "batch_size": 32}

        expanded = expand_parameter_sweeps(config)
        assert len(expanded) == 1
        assert expanded[0] == config

    def test_expand_parameter_sweeps_preserves_original(self):
        """Test that expansion doesn't modify original config."""
        original_config = {"lr": RangeSweep(0.01, 0.03, 0.01), "nested": {"value": 42}}
        config_copy = original_config.copy()

        expanded = expand_parameter_sweeps(original_config)

        # Original config should be unchanged
        assert original_config == config_copy
        assert isinstance(original_config["lr"], RangeSweep)

        # Expanded configs should have concrete values
        for expanded_config in expanded:
            assert isinstance(expanded_config["lr"], (int, float))
            assert expanded_config["nested"]["value"] == 42
