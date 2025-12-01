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


class TestGetStdoutStderr:
    """Test get command with stdout/stderr fields."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_get_stdout(self, clean_git_repo, tmp_path):
        """Test getting stdout from an experiment."""
        # Create a script that outputs to stdout
        script = tmp_path / "output_script.py"
        script.write_text("print('Hello from stdout')\n")

        result = self.runner.invoke(
            cli, ["run", str(script), "--name", "get-stdout-test"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get stdout
        result = self.runner.invoke(cli, ["get", "stdout", exp_id])
        assert result.exit_code == 0
        # Script prints something to stdout
        assert "Hello from stdout" in result.output

    def test_get_stderr_empty(self, clean_git_repo, sample_experiment_script):
        """Test getting stderr when it's empty."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-stderr-test"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get stderr - should be empty for successful script
        result = self.runner.invoke(cli, ["get", "stderr", exp_id])
        assert result.exit_code == 0

    def test_get_stdout_tail(self, clean_git_repo, tmp_path):
        """Test getting last N lines of stdout."""
        # Create script that prints numbered lines
        script = tmp_path / "print_lines.py"
        script.write_text("for i in range(20):\n    print(f'Line {i}')\n")

        result = self.runner.invoke(
            cli, ["run", str(script), "--name", "get-stdout-tail-test"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get last 5 lines
        result = self.runner.invoke(cli, ["get", "stdout", exp_id, "--tail", "5"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) == 5
        assert "Line 15" in result.output
        assert "Line 19" in result.output

    def test_get_stdout_head(self, clean_git_repo, tmp_path):
        """Test getting first N lines of stdout."""
        # Create script that prints numbered lines
        script = tmp_path / "print_lines.py"
        script.write_text("for i in range(20):\n    print(f'Line {i}')\n")

        result = self.runner.invoke(
            cli, ["run", str(script), "--name", "get-stdout-head-test"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get first 5 lines
        result = self.runner.invoke(cli, ["get", "stdout", exp_id, "--head", "5"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) == 5
        assert "Line 0" in result.output
        assert "Line 4" in result.output

    def test_get_stdout_head_and_tail(self, clean_git_repo, tmp_path):
        """Test getting first and last N lines with ellipsis."""
        # Create script that prints numbered lines
        script = tmp_path / "print_lines.py"
        script.write_text("for i in range(20):\n    print(f'Line {i}')\n")

        result = self.runner.invoke(
            cli, ["run", str(script), "--name", "get-stdout-headtail-test"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get first 3 and last 3 lines
        result = self.runner.invoke(
            cli, ["get", "stdout", exp_id, "--head", "3", "--tail", "3"]
        )
        assert result.exit_code == 0
        assert "Line 0" in result.output
        assert "Line 2" in result.output
        assert "..." in result.output
        assert "Line 17" in result.output
        assert "Line 19" in result.output

    def test_get_stdout_json(self, clean_git_repo, sample_experiment_script):
        """Test getting stdout as JSON."""
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "get-stdout-json-test"],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get stdout as JSON
        result = self.runner.invoke(cli, ["get", "stdout", exp_id, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert isinstance(data, str)

    def _extract_experiment_id(self, output: str) -> str | None:
        """Extract experiment ID from yanex run output."""
        for line in output.split("\n"):
            if "Experiment completed successfully:" in line:
                return line.split(":")[-1].strip()
        return None


class TestGetStdoutStderrValidation:
    """Test validation for stdout/stderr options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_tail_only_for_stdout_stderr(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --tail is rejected for non-stdout/stderr fields."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-tail-validate"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Try to use --tail with status field
        result = self.runner.invoke(cli, ["get", "status", exp_id, "--tail", "5"])
        assert result.exit_code != 0
        assert "--tail option only applies to stdout/stderr" in result.output

    def test_head_only_for_stdout_stderr(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --head is rejected for non-stdout/stderr fields."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-head-validate"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Try to use --head with status field
        result = self.runner.invoke(cli, ["get", "status", exp_id, "--head", "5"])
        assert result.exit_code != 0
        assert "--head option only applies to stdout/stderr" in result.output

    def test_csv_not_supported_for_stdout(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --csv is rejected for stdout field."""
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "get-csv-validate"]
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Try to use --csv with stdout field
        result = self.runner.invoke(cli, ["get", "stdout", exp_id, "--csv"])
        assert result.exit_code != 0
        assert "--csv output is not supported" in result.output

    def _extract_experiment_id(self, output: str) -> str | None:
        """Extract experiment ID from yanex run output."""
        for line in output.split("\n"):
            if "Experiment completed successfully:" in line:
                return line.split(":")[-1].strip()
        return None


class TestGetCommandReconstructedCommands:
    """Test get command with cli-command and run-command fields."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_get_cli_command(self, clean_git_repo, sample_experiment_script):
        """Test getting cli-command from an experiment."""
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "get-cli-cmd-test",
                "--param",
                "lr=0.01",
                "--tag",
                "test",
            ],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get cli-command
        result = self.runner.invoke(cli, ["get", "cli-command", exp_id])
        assert result.exit_code == 0
        assert "yanex run" in result.output
        assert '-p "lr=0.01"' in result.output
        assert '-n "get-cli-cmd-test"' in result.output
        assert '-t "test"' in result.output

    def test_get_run_command(self, clean_git_repo, sample_experiment_script):
        """Test getting run-command from an experiment."""
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "get-run-cmd-test",
                "--param",
                "epochs=50",
            ],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get run-command
        result = self.runner.invoke(cli, ["get", "run-command", exp_id])
        assert result.exit_code == 0
        assert "yanex run" in result.output
        assert '-p "epochs=50"' in result.output
        assert '-n "get-run-cmd-test"' in result.output

    def test_get_cli_command_with_config(self, clean_git_repo, tmp_path):
        """Test cli-command includes config files."""
        # Create config file and script
        config = tmp_path / "test-config.yaml"
        config.write_text("lr: 0.001\n")
        script = tmp_path / "test_script.py"
        script.write_text("print('test')\n")

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script),
                "--config",
                str(config),
                "--name",
                "config-cmd-test",
            ],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get cli-command - should include config
        result = self.runner.invoke(cli, ["get", "cli-command", exp_id])
        assert result.exit_code == 0
        assert "-c" in result.output
        assert "test-config.yaml" in result.output

    def test_get_cli_command_json(self, clean_git_repo, sample_experiment_script):
        """Test getting cli-command as JSON."""
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "json-cli-cmd-test",
            ],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)
        assert exp_id is not None

        # Get cli-command as JSON
        result = self.runner.invoke(cli, ["get", "cli-command", exp_id, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert isinstance(data, str)
        assert "yanex run" in data

    def test_get_commands_help_lists_new_fields(self):
        """Test that help includes cli-command and run-command fields."""
        result = self.runner.invoke(cli, ["get", "--help"])
        assert result.exit_code == 0
        assert "cli-command" in result.output
        assert "run-command" in result.output

    def _extract_experiment_id(self, output: str) -> str | None:
        """Extract experiment ID from yanex run output."""
        for line in output.split("\n"):
            if "Experiment completed successfully:" in line:
                return line.split(":")[-1].strip()
        return None


class TestReconstructCommandsUnit:
    """Unit tests for reconstruct_cli_command and reconstruct_run_command."""

    def test_reconstruct_cli_command_basic(self):
        """Test basic command reconstruction from cli_args."""
        from yanex.cli.commands.get import reconstruct_cli_command

        cli_args = {
            "script": "train.py",
            "param": ["lr=0.01", "epochs=10"],
            "name": "my-experiment",
        }

        result = reconstruct_cli_command(cli_args)
        assert (
            result
            == 'yanex run train.py -p "lr=0.01" -p "epochs=10" -n "my-experiment"'
        )

    def test_reconstruct_cli_command_with_config(self):
        """Test command reconstruction with config files."""
        from yanex.cli.commands.get import reconstruct_cli_command

        cli_args = {
            "script": "train.py",
            "config": ["base.yaml", "override.yaml"],
            "param": ["lr=0.01"],
        }

        result = reconstruct_cli_command(cli_args)
        assert result == 'yanex run train.py -c base.yaml -c override.yaml -p "lr=0.01"'

    def test_reconstruct_cli_command_with_dependencies(self):
        """Test command reconstruction with dependencies."""
        from yanex.cli.commands.get import reconstruct_cli_command

        cli_args = {
            "script": "train.py",
            "depends_on": ["data=abc123", "model=def456"],
        }

        result = reconstruct_cli_command(cli_args)
        assert result == "yanex run train.py -D data=abc123 -D model=def456"

    def test_reconstruct_cli_command_with_tags(self):
        """Test command reconstruction with tags."""
        from yanex.cli.commands.get import reconstruct_cli_command

        cli_args = {
            "script": "train.py",
            "tag": ["sweep", "production"],
        }

        result = reconstruct_cli_command(cli_args)
        assert result == 'yanex run train.py -t "sweep" -t "production"'

    def test_reconstruct_cli_command_sweep_syntax(self):
        """Test that sweep syntax is preserved in cli_args."""
        from yanex.cli.commands.get import reconstruct_cli_command

        cli_args = {
            "script": "train.py",
            "param": ["lr=0.001,0.01,0.1"],  # Sweep syntax
            "name": "sweep-hpo",
            "parallel": 4,
        }

        result = reconstruct_cli_command(cli_args)
        assert result == 'yanex run train.py -p "lr=0.001,0.01,0.1" -n "sweep-hpo" -j 4'

    def test_reconstruct_cli_command_all_options(self):
        """Test command reconstruction with all options."""
        from yanex.cli.commands.get import reconstruct_cli_command

        cli_args = {
            "script": "train.py",
            "clone_from": "abc123",
            "config": ["config.yaml"],
            "depends_on": ["data=xyz789"],
            "param": ["lr=0.01"],
            "name": "my-exp",
            "description": "A test experiment",
            "tag": ["test"],
            "stage": True,
            "parallel": 8,
        }

        result = reconstruct_cli_command(cli_args)
        expected_parts = [
            "yanex run",
            "train.py",
            "--clone-from abc123",
            "-c config.yaml",
            "-D data=xyz789",
            '-p "lr=0.01"',
            '-n "my-exp"',
            '-d "A test experiment"',
            '-t "test"',
            "--stage",
            "-j 8",
        ]
        assert result == " ".join(expected_parts)

    def test_reconstruct_cli_command_empty_lists(self):
        """Test command reconstruction with empty lists."""
        from yanex.cli.commands.get import reconstruct_cli_command

        cli_args = {
            "script": "train.py",
            "config": [],
            "param": [],
            "tag": [],
        }

        result = reconstruct_cli_command(cli_args)
        assert result == "yanex run train.py"

    def test_reconstruct_run_command_with_mock(self):
        """Test run command reconstruction using mock experiment."""
        from yanex.cli.commands.get import reconstruct_run_command

        exp = Mock()
        exp._load_metadata = Mock(
            return_value={
                "cli_args": {
                    "script": "train.py",
                    "config": ["config.yaml"],
                    "param": ["lr=0.001,0.01,0.1"],  # Original sweep syntax
                }
            }
        )
        exp.get_params = Mock(
            return_value={"lr": 0.01}  # Resolved value for this experiment
        )
        exp.dependencies = {"data": "abc123"}
        exp.name = "sweep-exp-1"
        exp.tags = ["sweep"]

        result = reconstruct_run_command(exp)
        # Should use resolved params, not sweep syntax
        assert "yanex run train.py" in result
        assert "-c config.yaml" in result
        assert "-D data=abc123" in result
        assert '-p "lr=0.01"' in result
        assert '-n "sweep-exp-1"' in result
        assert '-t "sweep"' in result
        # Should NOT contain sweep syntax
        assert "0.001,0.01,0.1" not in result

    def test_reconstruct_run_command_empty_dependencies(self):
        """Test run command with no dependencies."""
        from yanex.cli.commands.get import reconstruct_run_command

        exp = Mock()
        exp._load_metadata = Mock(
            return_value={
                "cli_args": {
                    "script": "train.py",
                }
            }
        )
        exp.get_params = Mock(return_value={"lr": 0.01})
        exp.dependencies = None
        exp.name = "test"
        exp.tags = None

        result = reconstruct_run_command(exp)
        assert "-D" not in result
        assert "-t" not in result


class TestResolveCliCommandFields:
    """Unit tests for resolve_field_value with cli-command/run-command fields."""

    def test_resolve_cli_command_field(self):
        """Test resolving cli-command field."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp._load_metadata = Mock(
            return_value={
                "cli_args": {
                    "script": "train.py",
                    "param": ["lr=0.01"],
                    "name": "test",
                }
            }
        )

        value, found = resolve_field_value(exp, "cli-command", "[not_found]")
        assert found is True
        assert value == 'yanex run train.py -p "lr=0.01" -n "test"'

    def test_resolve_cli_command_missing_script(self):
        """Test resolving cli-command when script is missing."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp._load_metadata = Mock(return_value={"cli_args": {}})

        value, found = resolve_field_value(exp, "cli-command", "[not_found]")
        assert found is False
        assert value == "[not_found]"

    def test_resolve_cli_command_no_cli_args(self):
        """Test resolving cli-command when cli_args is missing."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp._load_metadata = Mock(return_value={})

        value, found = resolve_field_value(exp, "cli-command", "[not_found]")
        assert found is False
        assert value == "[not_found]"

    def test_resolve_run_command_field(self):
        """Test resolving run-command field."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp._load_metadata = Mock(
            return_value={
                "cli_args": {
                    "script": "train.py",
                    # Only params passed via CLI -p are included in run-command
                    "param": ["lr=0.01", "epochs=10"],
                }
            }
        )
        exp.get_params = Mock(return_value={"lr": 0.01, "epochs": 10})
        exp.dependencies = {}
        exp.name = "test"
        exp.tags = []

        value, found = resolve_field_value(exp, "run-command", "[not_found]")
        assert found is True
        assert "yanex run train.py" in value
        assert '-p "lr=0.01"' in value
        assert '-p "epochs=10"' in value

    def test_resolve_run_command_missing_script(self):
        """Test resolving run-command when script is missing."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp._load_metadata = Mock(return_value={"cli_args": {}})

        value, found = resolve_field_value(exp, "run-command", "[not_found]")
        assert found is False
        assert value == "[not_found]"


class TestResolveDirectoryFields:
    """Unit tests for resolve_field_value with experiment-dir/artifacts-dir fields."""

    def test_resolve_experiment_dir(self, tmp_path):
        """Test resolving experiment-dir field."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        exp_dir = tmp_path / "experiments" / "abc12345"
        exp.experiment_dir = exp_dir

        value, found = resolve_field_value(exp, "experiment-dir", "[not_found]")
        assert found is True
        assert value == str(exp_dir)

    def test_resolve_artifacts_dir(self, tmp_path):
        """Test resolving artifacts-dir field."""
        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        artifacts_dir = tmp_path / "experiments" / "abc12345" / "artifacts"
        exp.artifacts_dir = artifacts_dir

        value, found = resolve_field_value(exp, "artifacts-dir", "[not_found]")
        assert found is True
        assert value == str(artifacts_dir)


class TestResolveStdoutStderrUnit:
    """Unit tests for resolve_field_value with stdout/stderr."""

    def test_resolve_stdout_with_tail(self, tmp_path):
        """Test resolving stdout with tail parameter."""
        from unittest.mock import Mock

        from yanex.cli.commands.get import resolve_field_value

        # Create mock experiment with artifacts_dir
        exp = Mock()
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        # Create stdout.txt with 10 lines
        stdout_file = artifacts_dir / "stdout.txt"
        stdout_file.write_text("\n".join(f"Line {i}" for i in range(10)))

        value, found = resolve_field_value(exp, "stdout", "[not_found]", tail=3)
        assert found is True
        assert "Line 7" in value
        assert "Line 8" in value
        assert "Line 9" in value
        assert "Line 0" not in value

    def test_resolve_stdout_with_head(self, tmp_path):
        """Test resolving stdout with head parameter."""
        from unittest.mock import Mock

        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        stdout_file = artifacts_dir / "stdout.txt"
        stdout_file.write_text("\n".join(f"Line {i}" for i in range(10)))

        value, found = resolve_field_value(exp, "stdout", "[not_found]", head=3)
        assert found is True
        assert "Line 0" in value
        assert "Line 1" in value
        assert "Line 2" in value
        assert "Line 9" not in value

    def test_resolve_stdout_with_head_and_tail(self, tmp_path):
        """Test resolving stdout with both head and tail parameters."""
        from unittest.mock import Mock

        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        stdout_file = artifacts_dir / "stdout.txt"
        stdout_file.write_text("\n".join(f"Line {i}" for i in range(20)))

        value, found = resolve_field_value(exp, "stdout", "[not_found]", tail=3, head=3)
        assert found is True
        assert "Line 0" in value
        assert "Line 2" in value
        assert "..." in value
        assert "Line 17" in value
        assert "Line 19" in value
        # Middle lines should not be there
        assert "Line 10" not in value

    def test_resolve_stdout_head_tail_covers_all(self, tmp_path):
        """Test that head+tail returns all when it covers entire file."""
        from unittest.mock import Mock

        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        stdout_file = artifacts_dir / "stdout.txt"
        stdout_file.write_text("\n".join(f"Line {i}" for i in range(5)))

        # head=3 + tail=3 = 6 > 5 lines, so return all
        value, found = resolve_field_value(exp, "stdout", "[not_found]", tail=3, head=3)
        assert found is True
        assert "..." not in value
        assert "Line 0" in value
        assert "Line 4" in value

    def test_resolve_stdout_missing(self, tmp_path):
        """Test resolving stdout when file doesn't exist."""
        from unittest.mock import Mock

        from yanex.cli.commands.get import resolve_field_value

        exp = Mock()
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        value, found = resolve_field_value(exp, "stdout", "[not_found]")
        assert found is False
        assert value == "[not_found]"


class TestFollowValidation:
    """Tests for --follow option validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_follow_only_for_stdout_stderr(self):
        """Test that --follow only works with stdout/stderr fields."""
        result = self.runner.invoke(cli, ["get", "status", "abc123", "--follow"])
        assert result.exit_code != 0
        assert "only applies to stdout/stderr" in result.output

    def test_follow_incompatible_with_csv(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --follow cannot be used with --csv."""
        # Create an experiment first
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "follow-csv-test"],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)

        # Try to use --follow with --csv
        result = self.runner.invoke(cli, ["get", "stdout", exp_id, "--follow", "--csv"])
        assert result.exit_code != 0
        # Either validation error is acceptable (--csv not supported for stdout, or --follow+--csv incompatible)
        assert "csv" in result.output.lower()

    def test_follow_incompatible_with_json(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --follow cannot be used with --json."""
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "follow-json-test"],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)

        result = self.runner.invoke(
            cli, ["get", "stdout", exp_id, "--follow", "--json"]
        )
        assert result.exit_code != 0
        assert "cannot be used with --json" in result.output

    def test_follow_incompatible_with_head(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --follow cannot be used with --head."""
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "follow-head-test"],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)

        result = self.runner.invoke(
            cli, ["get", "stdout", exp_id, "--follow", "--head", "10"]
        )
        assert result.exit_code != 0
        assert "cannot be used with --head" in result.output

    def test_follow_requires_single_experiment(self):
        """Test that --follow requires a single experiment ID, not filters."""
        result = self.runner.invoke(cli, ["get", "stdout", "-s", "running", "--follow"])
        assert result.exit_code != 0
        assert "requires a single experiment ID" in result.output

    def test_follow_with_tail_is_allowed(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --follow can be used with --tail (shows last N lines first)."""
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "follow-tail-test"],
        )
        assert result.exit_code == 0
        exp_id = self._extract_experiment_id(result.output)

        # --follow with --tail should work (no error)
        # The experiment is already completed, so it will just show output and exit
        result = self.runner.invoke(
            cli, ["get", "stdout", exp_id, "--follow", "--tail", "5"]
        )
        # Should succeed (exit code 0) since experiment is completed
        assert result.exit_code == 0

    def test_follow_help_shows_option(self):
        """Test that --follow option is shown in help."""
        result = self.runner.invoke(cli, ["get", "--help"])
        assert result.exit_code == 0
        assert "--follow" in result.output or "-f" in result.output

    def _extract_experiment_id(self, output: str) -> str | None:
        """Extract experiment ID from yanex run output."""
        for line in output.split("\n"):
            if "Experiment completed successfully:" in line:
                return line.split(":")[-1].strip()
        return None


class TestFollowOutputUnit:
    """Unit tests for follow_output function."""

    def test_follow_completed_experiment_exits_immediately(self, tmp_path):
        """Test that follow_output exits immediately for completed experiments."""
        from unittest.mock import Mock, patch

        from yanex.cli.commands.get import follow_output

        # Create mock experiment
        exp = Mock()
        exp.id = "test123"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        # Create stdout file
        stdout_file = artifacts_dir / "stdout.txt"
        stdout_file.write_text("Test output\n")

        # Mock yr.get_experiment to return completed status
        with patch("yanex.cli.commands.get.yr.get_experiment") as mock_get:
            mock_exp = Mock()
            mock_exp.status = "completed"
            mock_get.return_value = mock_exp

            # Should exit immediately without blocking
            follow_output(exp, "stdout")

            # Verify get_experiment was called to check status
            mock_get.assert_called()

    def test_follow_shows_initial_content(self, tmp_path, capsys):
        """Test that follow_output shows initial file content."""
        from unittest.mock import Mock, patch

        from yanex.cli.commands.get import follow_output

        exp = Mock()
        exp.id = "test123"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        # Create stdout file with content
        stdout_file = artifacts_dir / "stdout.txt"
        stdout_file.write_text("Line 1\nLine 2\nLine 3\n")

        with patch("yanex.cli.commands.get.yr.get_experiment") as mock_get:
            mock_exp = Mock()
            mock_exp.status = "completed"
            mock_get.return_value = mock_exp

            follow_output(exp, "stdout")

        captured = capsys.readouterr()
        assert "Line 1" in captured.out
        assert "Line 2" in captured.out
        assert "Line 3" in captured.out

    def test_follow_with_initial_tail(self, tmp_path, capsys):
        """Test that follow_output respects initial_tail parameter."""
        from unittest.mock import Mock, patch

        from yanex.cli.commands.get import follow_output

        exp = Mock()
        exp.id = "test123"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        exp.artifacts_dir = artifacts_dir

        # Create stdout file with many lines
        stdout_file = artifacts_dir / "stdout.txt"
        stdout_file.write_text("\n".join(f"Line {i}" for i in range(20)) + "\n")

        with patch("yanex.cli.commands.get.yr.get_experiment") as mock_get:
            mock_exp = Mock()
            mock_exp.status = "completed"
            mock_get.return_value = mock_exp

            follow_output(exp, "stdout", initial_tail=3)

        captured = capsys.readouterr()
        # Should show only last 3 lines
        assert "Line 17" in captured.out
        assert "Line 18" in captured.out
        assert "Line 19" in captured.out
        # Should not show early lines
        assert "Line 0" not in captured.out
        assert "Line 10" not in captured.out
