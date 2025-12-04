"""Tests for field formatting utilities."""

from rich.text import Text

from yanex.cli.formatters.fields import (
    format_cancelled_message,
    format_description,
    format_duration_styled,
    format_error_message,
    format_experiment_id,
    format_experiment_name,
    format_slot_name,
    format_status,
    format_status_symbol,
    format_success_message,
    format_tags,
    format_target_marker,
    format_timestamp_absolute,
    format_timestamp_relative,
    format_verbose,
    format_warning_message,
)
from yanex.cli.formatters.theme import (
    ID_STYLE,
    NAME_STYLE,
    SLOT_STYLE,
    STATUS_COLORS,
    STATUS_SYMBOLS,
    TAGS_STYLE,
    TARGET_STYLE,
    TIMESTAMP_STYLE,
)


class TestFormatExperimentId:
    """Tests for format_experiment_id function."""

    def test_returns_text_object(self):
        """Test that a Text object is returned."""
        result = format_experiment_id("abc12345")
        assert isinstance(result, Text)

    def test_uses_id_style(self):
        """Test that ID_STYLE is applied."""
        result = format_experiment_id("abc12345")
        assert result.style == ID_STYLE

    def test_preserves_id_value(self):
        """Test that the ID value is preserved."""
        result = format_experiment_id("abc12345")
        assert str(result) == "abc12345"


class TestFormatExperimentName:
    """Tests for format_experiment_name function."""

    def test_valid_name_uses_name_style(self):
        """Test that valid names use NAME_STYLE."""
        result = format_experiment_name("my-experiment")
        assert result.style == NAME_STYLE
        assert str(result) == "my-experiment"

    def test_none_name_shows_fallback(self):
        """Test that None name shows fallback text."""
        result = format_experiment_name(None)
        assert str(result) == "(unnamed)"
        assert result.style == "dim"

    def test_empty_name_shows_fallback(self):
        """Test that empty name shows fallback text."""
        result = format_experiment_name("")
        assert str(result) == "(unnamed)"

    def test_custom_fallback(self):
        """Test that custom fallback can be provided."""
        result = format_experiment_name(None, fallback="[no name]")
        assert str(result) == "[no name]"


class TestFormatStatus:
    """Tests for format_status function."""

    def test_completed_status(self):
        """Test formatting completed status."""
        result = format_status("completed")
        text = str(result)
        assert STATUS_SYMBOLS["completed"] in text
        assert "completed" in text
        assert result.style == STATUS_COLORS["completed"]

    def test_failed_status(self):
        """Test formatting failed status."""
        result = format_status("failed")
        assert STATUS_SYMBOLS["failed"] in str(result)
        assert result.style == STATUS_COLORS["failed"]

    def test_running_status(self):
        """Test formatting running status."""
        result = format_status("running")
        assert STATUS_SYMBOLS["running"] in str(result)
        assert result.style == STATUS_COLORS["running"]

    def test_without_symbol(self):
        """Test formatting status without symbol."""
        result = format_status("completed", include_symbol=False)
        assert str(result) == "completed"
        assert STATUS_SYMBOLS["completed"] not in str(result)

    def test_unknown_status(self):
        """Test formatting unknown status."""
        result = format_status("unknown")
        assert result.style == "white"


class TestFormatStatusSymbol:
    """Tests for format_status_symbol function."""

    def test_returns_only_symbol(self):
        """Test that only the symbol is returned."""
        result = format_status_symbol("completed")
        assert str(result) == STATUS_SYMBOLS["completed"]

    def test_uses_status_color(self):
        """Test that the status color is used."""
        result = format_status_symbol("failed")
        assert result.style == STATUS_COLORS["failed"]


class TestFormatTimestampRelative:
    """Tests for format_timestamp_relative function."""

    def test_none_timestamp(self):
        """Test formatting None timestamp."""
        result = format_timestamp_relative(None)
        assert str(result) == "-"
        assert result.style == "dim"

    def test_valid_timestamp(self):
        """Test formatting a valid timestamp."""
        result = format_timestamp_relative("2023-01-01T12:00:00Z")
        assert isinstance(result, Text)
        assert result.style == TIMESTAMP_STYLE

    def test_invalid_timestamp_preserved(self):
        """Test that invalid timestamps are preserved."""
        result = format_timestamp_relative("invalid")
        assert str(result) == "invalid"


class TestFormatTimestampAbsolute:
    """Tests for format_timestamp_absolute function."""

    def test_none_timestamp(self):
        """Test formatting None timestamp."""
        result = format_timestamp_absolute(None)
        assert str(result) == "-"
        assert result.style == "dim"

    def test_valid_timestamp(self):
        """Test formatting a valid timestamp."""
        result = format_timestamp_absolute("2023-01-01T12:00:00Z")
        assert isinstance(result, Text)
        # Should contain formatted date
        text = str(result)
        assert "2023-01-01" in text
        assert result.style == TIMESTAMP_STYLE


class TestFormatDurationStyled:
    """Tests for format_duration_styled function."""

    def test_none_start_time(self):
        """Test formatting when start_time is None."""
        result = format_duration_styled(None, None, "completed")
        assert str(result) == "-"
        assert result.style == "dim"

    def test_completed_duration_uses_green(self):
        """Test that completed experiments use green."""
        result = format_duration_styled(
            "2023-01-01T12:00:00Z", "2023-01-01T12:01:00Z", "completed"
        )
        assert result.style == STATUS_COLORS["completed"]

    def test_running_duration_uses_yellow(self):
        """Test that running experiments use yellow."""
        result = format_duration_styled("2023-01-01T12:00:00Z", None, "running")
        assert result.style == STATUS_COLORS["running"]

    def test_failed_duration_uses_red(self):
        """Test that failed experiments use red."""
        result = format_duration_styled(
            "2023-01-01T12:00:00Z", "2023-01-01T12:01:00Z", "failed"
        )
        assert result.style == STATUS_COLORS["failed"]


class TestFormatTags:
    """Tests for format_tags function."""

    def test_none_tags(self):
        """Test formatting None tags."""
        result = format_tags(None)
        assert str(result) == "-"
        assert result.style == "dim"

    def test_empty_tags(self):
        """Test formatting empty tags list."""
        result = format_tags([])
        assert str(result) == "-"
        assert result.style == "dim"

    def test_single_tag(self):
        """Test formatting single tag."""
        result = format_tags(["ml"])
        assert str(result) == "ml"
        assert result.style == TAGS_STYLE

    def test_multiple_tags(self):
        """Test formatting multiple tags."""
        result = format_tags(["ml", "training", "v1"])
        assert str(result) == "ml, training, v1"
        assert result.style == TAGS_STYLE


class TestFormatSlotName:
    """Tests for format_slot_name function."""

    def test_formats_with_angle_brackets(self):
        """Test that slot name is wrapped in angle brackets."""
        result = format_slot_name("data")
        assert str(result) == "<data>"

    def test_uses_slot_style(self):
        """Test that SLOT_STYLE is used."""
        result = format_slot_name("model")
        assert result.style == SLOT_STYLE


class TestFormatTargetMarker:
    """Tests for format_target_marker function."""

    def test_returns_star_marker(self):
        """Test that <*> marker is returned."""
        result = format_target_marker()
        assert str(result) == "<*>"

    def test_uses_target_style(self):
        """Test that TARGET_STYLE is used."""
        result = format_target_marker()
        assert result.style == TARGET_STYLE


class TestFormatDescription:
    """Tests for format_description function."""

    def test_none_description(self):
        """Test formatting None description."""
        result = format_description(None)
        assert str(result) == "-"
        assert result.style == "dim"

    def test_empty_description(self):
        """Test formatting empty description."""
        result = format_description("")
        assert str(result) == "-"
        assert result.style == "dim"

    def test_valid_description(self):
        """Test formatting valid description."""
        result = format_description("Training run with new hyperparameters")
        assert str(result) == "Training run with new hyperparameters"


class TestMessageFormatters:
    """Tests for message formatting functions."""

    def test_success_message(self):
        """Test format_success_message."""
        result = format_success_message("Experiment completed")
        assert "[green]" in result
        assert "Experiment completed" in result

    def test_error_message(self):
        """Test format_error_message."""
        result = format_error_message("Experiment failed")
        assert "[red]" in result
        assert "Experiment failed" in result

    def test_warning_message(self):
        """Test format_warning_message."""
        result = format_warning_message("Uncommitted changes")
        assert "[yellow]" in result
        assert "Uncommitted changes" in result

    def test_verbose_message(self):
        """Test format_verbose."""
        result = format_verbose("Debug information")
        assert "[dim]" in result
        assert "Debug information" in result

    def test_cancelled_message(self):
        """Test format_cancelled_message."""
        result = format_cancelled_message("Experiment cancelled")
        assert "Experiment cancelled" in result
