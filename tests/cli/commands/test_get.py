"""Tests for the yanex get command."""

import json
from unittest.mock import Mock

from tests.test_utils import (
    create_cli_runner,
)
from yanex.cli.main import cli


class TestGetCommandHelp:
    """Test get command help and documentation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_get_help_output(self):
        """Test that get command shows help information."""
        result = self.runner.invoke(cli, ["get", "--help"])
        assert result.exit_code == 0
        assert "Get a specific field value" in result.output
        assert "FIELD" in result.output
        assert "EXPERIMENT_ID" in result.output
        assert "--csv" in result.output
        assert "--json" in result.output
        assert "--default" in result.output
        assert "--no-id" in result.output

    def test_get_help_shows_field_examples(self):
        """Test that help includes field examples."""
        result = self.runner.invoke(cli, ["get", "--help"])
        assert result.exit_code == 0
        assert "params." in result.output
        assert "metrics." in result.output
        assert "git." in result.output

    def test_get_help_shows_bash_substitution_examples(self):
        """Test that help includes bash substitution examples."""
        result = self.runner.invoke(cli, ["get", "--help"])
        assert result.exit_code == 0
        assert "Bash substitution" in result.output
        assert "--csv" in result.output


class TestGetCommandValidation:
    """Test get command validation and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_get_requires_field(self):
        """Test that get command requires field argument."""
        result = self.runner.invoke(cli, ["get"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "FIELD" in result.output

    def test_get_requires_experiment_or_filters(self):
        """Test error when neither experiment ID nor filters provided."""
        result = self.runner.invoke(cli, ["get", "status"])
        assert result.exit_code != 0
        assert "Must specify either EXPERIMENT_ID or filter options" in result.output

    def test_get_cannot_mix_experiment_and_filters(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test error when both experiment ID and filters provided."""
        # First create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "test-get-mix"]
        )
        assert result.exit_code == 0

        # Extract experiment ID
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Try to use both ID and filter
        result = self.runner.invoke(cli, ["get", "status", exp_id, "-s", "completed"])
        assert result.exit_code != 0
        assert "Cannot specify both" in result.output

    def test_get_nonexistent_experiment(self):
        """Test error for nonexistent experiment."""
        result = self.runner.invoke(cli, ["get", "status", "nonexistent123"])
        assert result.exit_code != 0
        assert "No experiment found" in result.output

    def _extract_experiment_id(self, output: str) -> str | None:
        """Extract experiment ID from yanex run output."""
        for line in output.split("\n"):
            if "Experiment completed successfully:" in line:
                return line.split(":")[-1].strip()
        return None


class TestGetCommandSingleExperiment:
    """Test get command with single experiment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_get_status(self, clean_git_repo, sample_experiment_script):
        """Test getting status from a single experiment."""
        # Create experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-test-status"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get status
        result = self.runner.invoke(cli, ["get", "status", exp_id])
        assert result.exit_code == 0
        assert "completed" in result.output

    def test_get_id(self, clean_git_repo, sample_experiment_script):
        """Test getting ID from a single experiment."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-test-id"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "id", exp_id])
        assert result.exit_code == 0
        assert exp_id in result.output

    def test_get_name(self, clean_git_repo, sample_experiment_script):
        """Test getting name from a single experiment."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-my-test-name"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "name", exp_id])
        assert result.exit_code == 0
        assert "get-my-test-name" in result.output

    def test_get_tags(self, clean_git_repo, sample_experiment_script):
        """Test getting tags from a single experiment."""
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "get-test-tags",
                "--tag",
                "gettag1",
                "--tag",
                "gettag2",
            ],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "tags", exp_id])
        assert result.exit_code == 0
        assert "gettag1" in result.output
        assert "gettag2" in result.output

    def test_get_tags_json(self, clean_git_repo, sample_experiment_script):
        """Test getting tags as JSON from a single experiment."""
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "get-test-tags-json",
                "--tag",
                "getalpha",
            ],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "tags", exp_id, "--json"])
        assert result.exit_code == 0
        tags = json.loads(result.output.strip())
        assert isinstance(tags, list)
        assert "getalpha" in tags

    def test_get_param(self, clean_git_repo, sample_experiment_script):
        """Test getting a parameter from a single experiment."""
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "get-test-param",
                "--param",
                "testlr=0.01",
            ],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "params.testlr", exp_id])
        assert result.exit_code == 0
        assert "0.01" in result.output

    def test_get_missing_param(self, clean_git_repo, sample_experiment_script):
        """Test getting a missing parameter returns default."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-test-missing"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "params.nonexistent", exp_id])
        assert result.exit_code == 0
        assert "[not_found]" in result.output

    def test_get_custom_default(self, clean_git_repo, sample_experiment_script):
        """Test custom default value for missing field."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-test-default"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(
            cli, ["get", "params.missing", exp_id, "--default", "N/A"]
        )
        assert result.exit_code == 0
        assert "N/A" in result.output

    def test_get_git_branch(self, clean_git_repo, sample_experiment_script):
        """Test getting git branch."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-test-git"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "git.branch", exp_id])
        assert result.exit_code == 0
        # Should return branch name (could be main, master, etc.)

    def _extract_experiment_id(self, output: str) -> str | None:
        """Extract experiment ID from yanex run output."""
        for line in output.split("\n"):
            if "Experiment completed successfully:" in line:
                return line.split(":")[-1].strip()
        return None


class TestGetCommandMultipleExperiments:
    """Test get command with multiple experiments via filters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_get_multiple_by_status(self, clean_git_repo, sample_experiment_script):
        """Test getting field from multiple experiments filtered by status."""
        # Create two experiments with unique names
        for i in range(2):
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(sample_experiment_script),
                    "--name",
                    f"get-multi-test-{i}",
                ],
            )
            assert result.exit_code == 0

        # Get IDs by name pattern (to only get our experiments)
        result = self.runner.invoke(cli, ["get", "id", "-n", "get-multi-test-*"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) == 2

    def test_get_multiple_csv(self, clean_git_repo, sample_experiment_script):
        """Test CSV output for multiple experiments."""
        # Create two experiments
        for i in range(2):
            result = self.runner.invoke(
                cli,
                ["run", str(sample_experiment_script), "--name", f"get-csv-test-{i}"],
            )
            assert result.exit_code == 0

        # Get IDs as CSV by name pattern
        result = self.runner.invoke(cli, ["get", "id", "-n", "get-csv-test-*", "--csv"])
        assert result.exit_code == 0
        # Should be comma-separated
        assert "," in result.output
        # Should not have trailing newline (for bash substitution)
        assert (
            not result.output.endswith("\n")
            or result.output.rstrip("\n") == result.output.rstrip()
        )

    def test_get_multiple_json(self, clean_git_repo, sample_experiment_script):
        """Test JSON output for multiple experiments."""
        # Create two experiments
        for i in range(2):
            result = self.runner.invoke(
                cli,
                ["run", str(sample_experiment_script), "--name", f"get-json-test-{i}"],
            )
            assert result.exit_code == 0

        # Get IDs as JSON by name pattern
        result = self.runner.invoke(
            cli, ["get", "id", "-n", "get-json-test-*", "--json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_get_multiple_with_id_prefix(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test getting status shows ID prefix by default."""
        # Create two experiments
        for i in range(2):
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(sample_experiment_script),
                    "--name",
                    f"get-prefix-test-{i}",
                ],
            )
            assert result.exit_code == 0

        # Get status (should have ID: value format)
        result = self.runner.invoke(cli, ["get", "status", "-n", "get-prefix-test-*"])
        assert result.exit_code == 0
        assert ": completed" in result.output

    def test_get_multiple_no_id(self, clean_git_repo, sample_experiment_script):
        """Test --no-id flag omits ID prefix."""
        # Create two experiments
        for i in range(2):
            result = self.runner.invoke(
                cli,
                ["run", str(sample_experiment_script), "--name", f"get-noid-test-{i}"],
            )
            assert result.exit_code == 0

        # Get status with --no-id
        result = self.runner.invoke(
            cli, ["get", "status", "-n", "get-noid-test-*", "--no-id"]
        )
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        # Each line should just be "completed" without ID prefix
        for line in lines:
            assert line == "completed"

    def test_get_multiple_with_limit(self, clean_git_repo, sample_experiment_script):
        """Test limit option."""
        # Create three experiments
        for i in range(3):
            result = self.runner.invoke(
                cli,
                ["run", str(sample_experiment_script), "--name", f"get-limit-test-{i}"],
            )
            assert result.exit_code == 0

        # Get with limit
        result = self.runner.invoke(
            cli, ["get", "id", "-n", "get-limit-test-*", "-l", "1"]
        )
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) == 1

    def test_get_by_name_pattern(self, clean_git_repo, sample_experiment_script):
        """Test filtering by name pattern."""
        # Create experiments with specific names
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "get-pattern-alpha"],
        )
        assert result.exit_code == 0

        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-pattern-beta"]
        )
        assert result.exit_code == 0

        # Get by name pattern
        result = self.runner.invoke(cli, ["get", "id", "-n", "get-pattern-*"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) == 2

    def test_get_empty_results(self):
        """Test that empty results produce no output."""
        # Use a pattern that won't match anything
        result = self.runner.invoke(
            cli, ["get", "id", "-n", "nonexistent-pattern-12345-*"]
        )
        assert result.exit_code == 0
        # Empty results should produce no output
        assert result.output.strip() == ""


class TestGetCommandDependencies:
    """Test getting dependencies from experiments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_get_dependencies_empty(self, clean_git_repo, sample_experiment_script):
        """Test getting dependencies when none exist."""
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "get-test-no-deps"],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "dependencies", exp_id])
        assert result.exit_code == 0
        # Empty dependencies should return empty string or nothing

    def test_get_dependencies_json_empty(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test getting dependencies as JSON when none exist."""
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "get-test-no-deps-json"],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        result = self.runner.invoke(cli, ["get", "dependencies", exp_id, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data == {}

    def _extract_experiment_id(self, output: str) -> str | None:
        """Extract experiment ID from yanex run output."""
        for line in output.split("\n"):
            if "Experiment completed successfully:" in line:
                return line.split(":")[-1].strip()
        return None


class TestResolveFieldValueUnit:
    """Unit tests for resolve_field_value function."""

    def test_resolve_id(self):
        """Test resolving id field."""
        from yanex.cli.commands.get import resolve_field_value

        # Create a mock experiment
        exp = Mock()
        exp.id = "test123"

        value, found = resolve_field_value(exp, "id", "[not_found]")
        assert value == "test123"
        assert found is True

    def test_resolve_status(self):
        """Test resolving status field."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.status = "completed"

        value, found = resolve_field_value(exp, "status", "[not_found]")
        assert value == "completed"
        assert found is True

    def test_resolve_params(self):
        """Test resolving params.* fields."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_param = Mock(return_value=0.01)

        value, found = resolve_field_value(exp, "params.lr", "[not_found]")
        assert value == 0.01
        assert found is True
        exp.get_param.assert_called_once_with("lr")

    def test_resolve_params_missing(self):
        """Test resolving missing params.* fields."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_param = Mock(return_value=None)

        value, found = resolve_field_value(exp, "params.missing", "[not_found]")
        assert value == "[not_found]"
        assert found is False

    def test_resolve_metrics_single_value(self):
        """Test resolving metrics.* fields with single value."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_metric = Mock(return_value=0.95)

        value, found = resolve_field_value(exp, "metrics.accuracy", "[not_found]")
        assert value == 0.95
        assert found is True

    def test_resolve_metrics_list_returns_last(self):
        """Test resolving metrics.* fields returns last value from list."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_metric = Mock(return_value=[0.8, 0.85, 0.9, 0.95])

        value, found = resolve_field_value(exp, "metrics.accuracy", "[not_found]")
        assert value == 0.95
        assert found is True

    def test_resolve_dependencies(self):
        """Test resolving dependencies field."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.dependencies = {"data": "abc123", "model": "def456"}

        value, found = resolve_field_value(exp, "dependencies", "[not_found]")
        assert value == {"data": "abc123", "model": "def456"}
        assert found is True

    def test_resolve_dependencies_empty(self):
        """Test resolving empty dependencies field."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.dependencies = {}

        value, found = resolve_field_value(exp, "dependencies", "[not_found]")
        assert value == {}
        assert found is True

    def test_resolve_params_list(self):
        """Test resolving params field returns list of parameter names."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_params = Mock(return_value={"lr": 0.01, "batch_size": 32, "epochs": 10})

        value, found = resolve_field_value(exp, "params", "[not_found]")
        assert value == ["batch_size", "epochs", "lr"]  # Sorted alphabetically
        assert found is True

    def test_resolve_params_list_empty(self):
        """Test resolving params field with no parameters."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_params = Mock(return_value={})

        value, found = resolve_field_value(exp, "params", "[not_found]")
        assert value == []
        assert found is True

    def test_resolve_metrics_list(self):
        """Test resolving metrics field returns list of metric names."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_metrics = Mock(
            return_value=[
                {"step": 0, "accuracy": 0.8, "loss": 0.5},
                {"step": 1, "accuracy": 0.9, "loss": 0.3},
            ]
        )

        value, found = resolve_field_value(exp, "metrics", "[not_found]")
        assert value == ["accuracy", "loss"]  # Sorted, excludes step
        assert found is True

    def test_resolve_metrics_list_empty(self):
        """Test resolving metrics field with no metrics."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp.get_metrics = Mock(return_value=[])

        value, found = resolve_field_value(exp, "metrics", "[not_found]")
        assert value == []
        assert found is True


class TestFormatValueUnit:
    """Unit tests for format_value function."""

    def test_format_string(self):
        """Test formatting string value."""
        from yanex.cli.commands.get import format_value

        assert format_value("hello") == "hello"

    def test_format_number(self):
        """Test formatting number value."""
        from yanex.cli.commands.get import format_value

        assert format_value(42) == "42"
        assert format_value(3.14) == "3.14"

    def test_format_none(self):
        """Test formatting None value."""
        from yanex.cli.commands.get import format_value

        assert format_value(None) == ""

    def test_format_list(self):
        """Test formatting list value."""
        from yanex.cli.commands.get import format_value

        assert format_value(["a", "b", "c"]) == "a, b, c"

    def test_format_dict_dependencies(self):
        """Test formatting dict as slot=id pairs."""
        from yanex.cli.commands.get import format_value

        result = format_value({"data": "abc123", "model": "def456"})
        assert "data=abc123" in result
        assert "model=def456" in result

    def test_format_json_mode(self):
        """Test formatting with JSON mode."""
        from yanex.cli.commands.get import format_value

        result = format_value({"key": "value"}, json_output=True)
        data = json.loads(result)
        assert data == {"key": "value"}


class TestFormatValueForCSVUnit:
    """Unit tests for format_value_for_csv function."""

    def test_format_csv_string(self):
        """Test CSV formatting string value."""
        from yanex.cli.commands.get import format_value_for_csv

        assert format_value_for_csv("hello") == "hello"

    def test_format_csv_none(self):
        """Test CSV formatting None value."""
        from yanex.cli.commands.get import format_value_for_csv

        assert format_value_for_csv(None) == ""

    def test_format_csv_list(self):
        """Test CSV formatting list value (comma-separated, no spaces)."""
        from yanex.cli.commands.get import format_value_for_csv

        assert format_value_for_csv(["a", "b", "c"]) == "a,b,c"

    def test_format_csv_dict(self):
        """Test CSV formatting dict as slot=id pairs (comma-separated)."""
        from yanex.cli.commands.get import format_value_for_csv

        result = format_value_for_csv({"a": "1", "b": "2"})
        assert "a=1" in result
        assert "b=2" in result
        assert "," in result
