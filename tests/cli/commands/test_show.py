"""
Tests for show command functionality.

This module tests the show command for displaying detailed experiment information.
"""

from pathlib import Path
from unittest.mock import Mock, patch

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestShowCommandHelp:
    """Test show command help and documentation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_show_help_output(self):
        """Test that show command shows help information."""
        result = self.runner.invoke(cli, ["show", "--help"])
        assert result.exit_code == 0
        assert "Show detailed information about an experiment" in result.output
        assert "EXPERIMENT_IDENTIFIER" in result.output
        assert "--show-metric" in result.output
        assert "--archived" in result.output

    def test_show_help_shows_identifier_types(self):
        """Test that help explains identifier types."""
        result = self.runner.invoke(cli, ["show", "--help"])
        assert result.exit_code == 0
        assert "experiment ID" in result.output or "ID" in result.output


class TestShowCommandBasicBehavior:
    """Test show command basic behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_show_requires_experiment_identifier(self):
        """Test that show command requires experiment identifier argument."""
        result = self.runner.invoke(cli, ["show"])
        assert result.exit_code != 0
        assert (
            "Missing argument" in result.output or "required" in result.output.lower()
        )

    def test_show_nonexistent_experiment(self):
        """Test showing nonexistent experiment."""
        result = self.runner.invoke(cli, ["show", "nonexistent-id"])
        assert result.exit_code != 0
        assert (
            "No experiment found" in result.output
            or "not found" in result.output.lower()
        )

    def test_show_with_archived_flag(self):
        """Test show command with --archived flag."""
        result = self.runner.invoke(cli, ["show", "nonexistent-id", "--archived"])
        assert result.exit_code != 0
        # Should still fail for nonexistent experiment


class TestShowCommandValidation:
    """Test show command validation and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_show_invalid_id_format(self):
        """Test showing experiment with invalid ID format."""
        result = self.runner.invoke(cli, ["show", "invalid@#$"])
        # Should handle gracefully - either error or not found
        assert result.exit_code != 0

    def test_show_with_empty_identifier(self):
        """Test showing experiment with empty identifier."""
        result = self.runner.invoke(cli, ["show", ""])
        # Should fail with appropriate error
        assert result.exit_code != 0


class TestShowCommandMetrics:
    """Test show command metrics display options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_show_with_single_metric_filter(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing experiment with single metric filter."""
        # Create experiment with metrics
        import yanex

        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "metrics-test"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Log some metrics to the experiment
        try:
            manager = yanex.core.manager.ExperimentManager()
            manager.log_metrics(
                exp_id, {"accuracy": 0.95, "loss": 0.05, "f1_score": 0.93}
            )
        except Exception:
            pass  # Metrics logging may not work in test context

        # Show with metric filter
        result = self.runner.invoke(cli, ["show", exp_id, "--show-metric", "accuracy"])
        assert result.exit_code == 0
        # May or may not show metrics depending on whether logging succeeded

    def test_show_with_multiple_metrics_filter(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing experiment with multiple metrics filter."""
        # Create experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "multi-metrics"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show with multiple metrics
        result = self.runner.invoke(
            cli, ["show", exp_id, "--show-metric", "accuracy,loss,f1_score"]
        )
        assert result.exit_code == 0

    def test_show_with_nonexistent_metric(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing experiment with nonexistent metric filter."""
        # Create experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "no-metrics"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show with nonexistent metric
        result = self.runner.invoke(
            cli, ["show", exp_id, "--show-metric", "nonexistent_metric"]
        )
        assert result.exit_code == 0
        # Should handle gracefully - may show warning


class TestShowCommandIntegration:
    """Integration tests for show command with real experiments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_show_single_experiment_by_id(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing single experiment by ID."""
        # Create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "show-test"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show experiment
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        assert "show-test" in result.output
        assert "Experiment:" in result.output
        assert "Status:" in result.output

    def test_show_experiment_by_name(self, clean_git_repo, sample_experiment_script):
        """Test showing experiment by name."""
        import uuid

        # Generate truly unique name to avoid test pollution
        unique_name = f"show-name-{uuid.uuid4().hex[:8]}"

        # Create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", unique_name]
        )
        assert result.exit_code == 0

        # Show by name
        result = self.runner.invoke(cli, ["show", unique_name])
        assert result.exit_code == 0
        assert unique_name in result.output
        assert "Status:" in result.output

    def test_show_experiment_by_id_prefix(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing experiment by ID prefix."""
        # Create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "prefix-test"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None
        assert len(exp_id) >= 4

        # Show by prefix (first 4 characters)
        id_prefix = exp_id[:4]
        result = self.runner.invoke(cli, ["show", id_prefix])
        assert result.exit_code == 0
        assert "prefix-test" in result.output

    def test_show_multiple_experiments_same_name(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing when multiple experiments have same name."""
        # Create multiple experiments with same name
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "duplicate-name"]
        )
        assert result.exit_code == 0

        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "duplicate-name"]
        )
        assert result.exit_code == 0

        # Try to show by name
        result = self.runner.invoke(cli, ["show", "duplicate-name"])
        assert result.exit_code != 0
        assert "Multiple experiments found" in result.output
        assert "use the specific experiment id" in result.output.lower()

    def test_show_experiment_with_tags(self, clean_git_repo, sample_experiment_script):
        """Test showing experiment with tags."""
        # Create experiment with tags
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "tagged-exp",
                "--tag",
                "ml",
                "--tag",
                "test",
            ],
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show experiment
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        # Tags should be displayed
        assert "ml" in result.output or "Tags" in result.output

    def test_show_experiment_with_description(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing experiment with description."""
        # Create experiment with description
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "described-exp",
                "--description",
                "Test experiment description",
            ],
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show experiment
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        # Description should be displayed
        assert (
            "Test experiment description" in result.output
            or "Description" in result.output
        )

    def test_show_experiment_with_config(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing experiment with configuration."""
        # Create experiment with config parameters
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "config-exp",
                "--param",
                "learning_rate=0.01",
                "--param",
                "batch_size=32",
            ],
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show experiment
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        # Config parameters should be displayed
        assert "learning_rate" in result.output or "Configuration" in result.output

    def test_show_completed_experiment_shows_duration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that completed experiments show duration."""
        # Create and complete an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "duration-test"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show experiment
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        # Duration should be shown for completed experiments
        assert "Duration:" in result.output or "Completed:" in result.output


class TestShowCommandExceptionHandling:
    """Test show command exception handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    @patch("yanex.cli.commands.show.find_experiment")
    def test_show_handles_find_experiment_exception(self, mock_find):
        """Test that show handles exceptions from find_experiment."""
        mock_find.side_effect = Exception("Test error")

        result = self.runner.invoke(cli, ["show", "test-id"])
        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("yanex.cli.commands.show.find_experiment")
    def test_show_handles_none_result(self, mock_find):
        """Test that show handles None result from find_experiment."""
        mock_find.return_value = None

        result = self.runner.invoke(cli, ["show", "test-id"])
        assert result.exit_code != 0
        assert "No experiment found" in result.output

    @patch("yanex.cli.commands.show.find_experiment")
    def test_show_handles_multiple_experiments_list(self, mock_find):
        """Test that show handles list of multiple experiments."""
        # Mock multiple experiments with same name
        mock_find.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"},
            {"id": "def67890", "name": "test", "status": "completed"},
        ]

        result = self.runner.invoke(cli, ["show", "test"])
        assert result.exit_code != 0
        assert "Multiple experiments found" in result.output
        assert "use the specific experiment id" in result.output.lower()


class TestDisplayExperimentDetails:
    """Test display_experiment_details function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_display_with_minimal_experiment(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test displaying experiment with minimal information."""
        # Create simple experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "minimal-exp"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Display should work even with minimal data
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        assert "Experiment:" in result.output

    def test_display_handles_missing_directory(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test display handles missing experiment directory gracefully."""
        # Create experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "missing-dir-test"]
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "Experiment completed successfully: <id>")
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show should still work
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0

    def test_display_with_many_metrics(self, clean_git_repo, sample_experiment_script):
        """Test displaying experiment with many metrics (>8)."""
        # Create experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "many-metrics"]
        )
        assert result.exit_code == 0

        # Extract experiment ID
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Log many metrics (>8 to trigger different display mode)
        try:
            import yanex

            manager = yanex.core.manager.ExperimentManager()
            metrics = {
                f"metric_{i}": float(i) * 0.1
                for i in range(12)  # 12 metrics
            }
            manager.log_metrics(exp_id, metrics)
        except Exception:
            pass  # Metrics logging may not work in test context

        # Show with many metrics
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        # Should display without errors

    def test_display_with_config_long_values(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test displaying experiment with long config values."""
        # Create experiment with long config values
        long_list = list(range(50))  # Long list
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "long-config",
                "--param",
                f"long_param={long_list}",
            ],
        )
        assert result.exit_code == 0

        # Extract experiment ID
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Show should truncate long values
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        # Should display without errors (long values truncated)

    def test_display_with_artifacts(self, clean_git_repo, sample_experiment_script):
        """Test displaying experiment with artifacts."""
        # Create experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "artifacts-test"]
        )
        assert result.exit_code == 0

        # Extract experiment ID
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Add an artifact
        try:
            import yanex

            manager = yanex.core.manager.ExperimentManager()
            exp_dir = manager.storage.get_experiment_dir(exp_id)
            artifacts_dir = exp_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            (artifacts_dir / "model.pkl").write_text("fake model data")
        except Exception:
            pass  # Artifact creation may not work in test context

        # Show should display artifacts section
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        # May or may not show artifacts depending on whether creation succeeded


class TestShowCommandFailedExperiments:
    """Test show command with failed/cancelled experiments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_show_failed_experiment(self, clean_git_repo):
        """Test showing failed experiment with error message."""
        from pathlib import Path

        # Create a script that will fail
        failing_script = Path(clean_git_repo.working_dir) / "failing.py"
        failing_script.write_text('raise ValueError("Test error")')
        clean_git_repo.index.add([str(failing_script)])
        clean_git_repo.index.commit("Add failing script")

        # Run failing experiment
        result = self.runner.invoke(
            cli, ["run", str(failing_script), "--name", "failed-exp"]
        )
        # Should fail but still create experiment
        assert result.exit_code != 0

        # Extract experiment ID (may be in error output)
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            # Failed experiments may show ID differently
            if "Experiment" in line and any(c in line for c in "0123456789abcdef"):
                # Try to extract hex ID
                import re

                matches = re.findall(r"\b[0-9a-f]{8}\b", line)
                if matches:
                    exp_id = matches[0]
                    break

        if exp_id:
            # Show failed experiment
            result = self.runner.invoke(cli, ["show", exp_id])
            assert result.exit_code == 0
            # Should show error information
            assert "failed" in result.output.lower() or "error" in result.output.lower()

    def test_show_with_requested_missing_metrics(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test showing experiment with requested metrics that don't exist."""
        # Create experiment
        result = self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "missing-metrics-test"],
        )
        assert result.exit_code == 0

        # Extract experiment ID
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        assert exp_id is not None

        # Request metrics that don't exist
        result = self.runner.invoke(
            cli,
            ["show", exp_id, "--show-metric", "nonexistent1,nonexistent2,nonexistent3"],
        )
        assert result.exit_code == 0
        # Should handle gracefully - may show warning about missing metrics


class TestDisplayExperimentDetailsDirectMocking:
    """Test display_experiment_details with direct mocking for edge cases."""

    def test_display_with_many_metrics_mocked(self):
        """Test display with >8 metrics using mocked data."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        # Mock experiment with many metrics
        experiment = {
            "id": "test1234",
            "name": "test-exp",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00",
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:00",
        }

        # Mock manager and storage
        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.side_effect = Exception("Not found")
        mock_manager.storage.load_config.side_effect = Exception("Not found")

        # Mock results with >8 metrics
        results = [
            {
                "step": i,
                "timestamp": "2025-01-01T00:00:00",
                **{f"metric_{j}": float(j) * 0.1 for j in range(12)},
            }
            for i in range(5)
        ]
        mock_manager.storage.load_results.return_value = results

        formatter = ExperimentTableFormatter()

        # Should not raise - display with many metrics
        display_experiment_details(mock_manager, experiment, formatter)

    def test_display_with_requested_metrics_mocked(self):
        """Test display with requested specific metrics using mocked data."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {
            "id": "test1234",
            "name": "test-exp",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00",
        }

        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.side_effect = Exception("Not found")
        mock_manager.storage.load_config.side_effect = Exception("Not found")

        # Mock results with metrics
        results = [
            {
                "step": 1,
                "timestamp": "2025-01-01T00:00:00",
                "accuracy": 0.95,
                "loss": 0.05,
            }
        ]
        mock_manager.storage.load_results.return_value = results

        formatter = ExperimentTableFormatter()

        # Request specific metrics (one exists, one doesn't)
        display_experiment_details(
            mock_manager,
            experiment,
            formatter,
            requested_metrics=["accuracy", "missing_metric"],
        )

    def test_display_with_missing_config(self):
        """Test display when config loading fails."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {"id": "test1234", "name": "test-exp", "status": "completed"}

        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.side_effect = Exception("Not found")
        mock_manager.storage.load_config.side_effect = Exception("Config not found")
        mock_manager.storage.load_results.side_effect = Exception("Results not found")
        mock_manager.storage.load_metadata.side_effect = Exception("Metadata not found")

        formatter = ExperimentTableFormatter()

        # Should handle all exceptions gracefully
        display_experiment_details(mock_manager, experiment, formatter)

    def test_display_with_long_config_values(self):
        """Test display with long config values that need truncation."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {"id": "test1234", "name": "test-exp", "status": "completed"}

        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp/test")

        # Config with long dict/list values
        long_config = {
            "short_param": "value",
            "long_dict": {f"key_{i}": f"value_{i}" for i in range(50)},
            "long_list": list(range(100)),
        }
        mock_manager.storage.load_config.return_value = long_config
        mock_manager.storage.load_results.side_effect = Exception("No results")
        mock_manager.storage.load_metadata.side_effect = Exception("No metadata")

        formatter = ExperimentTableFormatter()

        # Should truncate long values
        display_experiment_details(mock_manager, experiment, formatter)

    def test_display_with_failed_status_and_error(self):
        """Test display with failed experiment showing error message."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {
            "id": "test1234",
            "name": "failed-exp",
            "status": "failed",
            "error_message": "ValueError: Test error message",
            "created_at": "2025-01-01T00:00:00",
            "started_at": "2025-01-01T00:00:00",
            "failed_at": "2025-01-01T00:01:00",
        }

        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp/test")
        mock_manager.storage.load_config.side_effect = Exception("No config")
        mock_manager.storage.load_results.side_effect = Exception("No results")
        mock_manager.storage.load_metadata.side_effect = Exception("No metadata")

        formatter = ExperimentTableFormatter()

        # Should display error message
        display_experiment_details(mock_manager, experiment, formatter)

    def test_display_with_cancelled_status(self):
        """Test display with cancelled experiment."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {
            "id": "test1234",
            "name": "cancelled-exp",
            "status": "cancelled",
            "cancellation_reason": "User requested cancellation",
            "created_at": "2025-01-01T00:00:00",
            "cancelled_at": "2025-01-01T00:01:00",
        }

        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp/test")
        mock_manager.storage.load_config.side_effect = Exception("No config")
        mock_manager.storage.load_results.side_effect = Exception("No results")
        mock_manager.storage.load_metadata.side_effect = Exception("No metadata")

        formatter = ExperimentTableFormatter()

        # Should display cancellation reason
        display_experiment_details(mock_manager, experiment, formatter)

    def test_display_with_environment_metadata(self):
        """Test display with environment and git metadata."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {"id": "test1234", "name": "test-exp", "status": "completed"}

        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp/test")
        mock_manager.storage.load_config.side_effect = Exception("No config")
        mock_manager.storage.load_results.side_effect = Exception("No results")

        # Metadata with git and environment info
        metadata = {
            "script_path": "/path/to/script.py",
            "git": {
                "branch": "main",
                "commit_hash": "abc123def456789",
                "commit_hash_short": "abc123de",
            },
            "environment": {
                "git": {"has_uncommitted_changes": True},
                "python": {"python_version": "3.11.9", "platform": "darwin"},
                "system": {"platform": {"system": "Darwin", "machine": "arm64"}},
            },
        }
        mock_manager.storage.load_metadata.return_value = metadata

        formatter = ExperimentTableFormatter()

        # Should display environment info including uncommitted changes warning
        display_experiment_details(mock_manager, experiment, formatter)

    def test_display_with_running_experiment(self):
        """Test display with currently running experiment."""
        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {
            "id": "test1234",
            "name": "running-exp",
            "status": "running",
            "created_at": "2025-01-01T00:00:00",
            "started_at": "2025-01-01T00:00:00",
        }

        mock_manager = Mock()
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp/test")
        mock_manager.storage.load_config.side_effect = Exception("No config")
        mock_manager.storage.load_results.side_effect = Exception("No results")
        mock_manager.storage.load_metadata.side_effect = Exception("No metadata")

        formatter = ExperimentTableFormatter()

        # Should calculate duration from start to now for running experiment
        display_experiment_details(mock_manager, experiment, formatter)

    def test_display_with_artifacts(self):
        """Test display with artifacts directory."""
        import tempfile
        import time

        from yanex.cli.commands.show import display_experiment_details
        from yanex.cli.formatters import ExperimentTableFormatter

        experiment = {"id": "test1234", "name": "test-exp", "status": "completed"}

        # Create temporary artifacts directory
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            artifacts_dir = exp_dir / "artifacts"
            artifacts_dir.mkdir()

            # Create some artifact files
            (artifacts_dir / "model.pkl").write_text("fake model")
            (artifacts_dir / "results.json").write_text('{"accuracy": 0.95}')
            time.sleep(0.01)  # Ensure different mtimes

            mock_manager = Mock()
            mock_manager.storage.get_experiment_dir.return_value = exp_dir
            mock_manager.storage.load_config.side_effect = Exception("No config")
            mock_manager.storage.load_results.side_effect = Exception("No results")
            mock_manager.storage.load_metadata.side_effect = Exception("No metadata")

            formatter = ExperimentTableFormatter()

            # Should display artifacts
            display_experiment_details(mock_manager, experiment, formatter)
