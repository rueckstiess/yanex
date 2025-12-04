"""
Tests for yanex CLI compare command functionality.
"""

from unittest.mock import patch

import pytest

from tests.test_utils import (
    TestDataFactory,
    TestFileHelpers,
    create_cli_runner,
)
from yanex.cli.commands.compare import compare_experiments
from yanex.core.comparison import ExperimentComparisonData
from yanex.core.manager import ExperimentManager


class TestCompareCommand:
    """Test compare command CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    @pytest.mark.parametrize(
        "filter_args,expected_message",
        [
            (["--status", "completed"], "No regular experiments found to compare"),
            (
                ["--started-after", "2024-01-01"],
                "No regular experiments found to compare",
            ),
            (["--max-rows", "10"], "No regular experiments found to compare"),
        ],
    )
    def test_compare_no_experiments_found(self, filter_args, expected_message):
        """Test compare command when no experiments are found."""
        with patch(
            "yanex.cli.commands.compare.find_experiments_by_filters"
        ) as mock_find:
            mock_find.return_value = []

            result = self.runner.invoke(compare_experiments, filter_args)

            assert result.exit_code == 0
            assert expected_message in result.output

    def test_compare_specific_experiments_by_id(self):
        """Test compare command with specific experiment IDs."""
        mock_experiments = [
            {"id": "exp001", "name": "experiment-1"},
            {"id": "exp002", "name": "experiment-2"},
        ]

        mock_comparison_data = TestDataFactory.create_comparison_data(
            [
                {
                    "id": "exp001",
                    "name": "experiment-1",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",
                    "param:epochs": "10",
                    "metric:accuracy": "0.95",
                    "metric:loss": "0.05",
                },
                {
                    "id": "exp002",
                    "name": "experiment-2",
                    "started": "2025-01-01 12:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.02",
                    "param:epochs": "5",
                    "metric:accuracy": "0.87",
                    "metric:loss": "0.13",
                },
            ]
        )

        with (
            patch(
                "yanex.cli.commands.compare.find_experiments_by_identifiers"
            ) as mock_find,
            patch.object(ExperimentComparisonData, "get_comparison_data") as mock_data,
        ):
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data

            result = self.runner.invoke(
                compare_experiments, ["exp001", "exp002", "--no-interactive"]
            )

            assert result.exit_code == 0
            # Should show both experiments in the output (might be truncated)
            assert "exp0" in result.output  # exp001 might be truncated to exp0…
            # Should show parameters and metrics (headers might be truncated)
            # Headers are now prefixed: param_learning_rate -> "para…", metric_accuracy -> "metr…"
            assert "para" in result.output  # param_learning_rate truncated
            assert "metr" in result.output  # metric_accuracy truncated

    def test_compare_csv_output(self):
        """Test compare command CSV output functionality."""
        mock_experiments = [{"id": "exp001", "name": "test-exp"}]

        mock_comparison_data = TestDataFactory.create_comparison_data(
            [
                {
                    "id": "exp001",
                    "name": "test-exp",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",
                    "metric:accuracy": "0.95",
                }
            ]
        )

        with (
            patch(
                "yanex.cli.commands.compare.find_experiments_by_identifiers"
            ) as mock_find,
            patch.object(ExperimentComparisonData, "get_comparison_data") as mock_data,
        ):
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data

            # Test CSV output to stdout
            result = self.runner.invoke(compare_experiments, ["exp001", "--csv"])

            assert result.exit_code == 0
            # Check CSV content in stdout
            assert "exp001" in result.output
            assert "param_learning_rate" in result.output
            assert "metric_accuracy" in result.output
            # CSV should have header row
            assert "id" in result.output

    def test_compare_json_output(self):
        """Test compare command JSON output functionality."""
        mock_experiments = [{"id": "exp001", "name": "test-exp"}]

        mock_comparison_data = TestDataFactory.create_comparison_data(
            [
                {
                    "id": "exp001",
                    "name": "test-exp",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",
                    "metric:accuracy": "0.95",
                }
            ]
        )

        with (
            patch(
                "yanex.cli.commands.compare.find_experiments_by_identifiers"
            ) as mock_find,
            patch.object(ExperimentComparisonData, "get_comparison_data") as mock_data,
        ):
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data

            # Test JSON output to stdout
            result = self.runner.invoke(compare_experiments, ["exp001", "--json"])

            assert result.exit_code == 0
            # Check JSON structure
            import json

            output_data = json.loads(result.output)
            assert "experiments" in output_data
            assert len(output_data["experiments"]) == 1
            assert output_data["experiments"][0]["id"] == "exp001"
            assert "param_learning_rate" in output_data["experiments"][0]
            assert "metric_accuracy" in output_data["experiments"][0]

    def test_compare_markdown_output(self):
        """Test compare command markdown output functionality."""
        mock_experiments = [{"id": "exp001", "name": "test-exp"}]

        mock_comparison_data = TestDataFactory.create_comparison_data(
            [
                {
                    "id": "exp001",
                    "name": "test-exp",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",
                    "metric:accuracy": "0.95",
                }
            ]
        )

        with (
            patch(
                "yanex.cli.commands.compare.find_experiments_by_identifiers"
            ) as mock_find,
            patch.object(ExperimentComparisonData, "get_comparison_data") as mock_data,
        ):
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data

            # Test markdown output to stdout
            result = self.runner.invoke(compare_experiments, ["exp001", "--markdown"])

            assert result.exit_code == 0
            # Check markdown table format
            assert "|" in result.output  # Table delimiters
            assert "---" in result.output  # Header separator
            assert "exp001" in result.output
            assert "param_learning_rate" in result.output
            assert "metric_accuracy" in result.output

    def test_compare_mutually_exclusive_options(self):
        """Test that identifiers and filters are mutually exclusive."""
        result = self.runner.invoke(
            compare_experiments, ["exp001", "--status", "completed"]
        )

        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

    @pytest.mark.parametrize(
        "filter_options,expected_in_output",
        [
            (["--status", "completed"], ["exp001"]),
            (["--status", "failed"], []),
            (["--max-rows", "1"], ["exp001"]),
        ],
    )
    def test_compare_filter_by_status(self, filter_options, expected_in_output):
        """Test compare command with various filter options."""
        # Mock experiments matching filters
        mock_experiments = [
            {"id": "exp001", "name": "completed-exp", "status": "completed"}
        ]
        if not expected_in_output:  # For failed status, return empty
            mock_experiments = []

        with patch(
            "yanex.cli.commands.compare.find_experiments_by_filters"
        ) as mock_find:
            mock_find.return_value = mock_experiments

            result = self.runner.invoke(
                compare_experiments, filter_options + ["--no-interactive"]
            )

            assert result.exit_code == 0
            for exp_id in expected_in_output:
                assert exp_id in result.output

    @pytest.mark.parametrize(
        "flag_options",
        [
            (["--only-different"]),
            (["--params", "learning_rate,epochs"]),
            (["--metrics", "accuracy,loss"]),
            (["--params", "learning_rate", "--metrics", "accuracy"]),
        ],
    )
    def test_compare_display_options(self, flag_options):
        """Test various display options for compare command."""
        mock_experiments = [
            {"id": "exp001", "name": "exp-1"},
            {"id": "exp002", "name": "exp-2"},
        ]

        mock_comparison_data = TestDataFactory.create_comparison_data(
            [
                {
                    "id": "exp001",
                    "name": "exp-1",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",
                    "param:epochs": "10",
                    "metric:accuracy": "0.95",
                    "metric:loss": "0.05",
                },
                {
                    "id": "exp002",
                    "name": "exp-2",
                    "started": "2025-01-01 12:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.02",
                    "param:epochs": "5",
                    "metric:accuracy": "0.87",
                    "metric:loss": "0.13",
                },
            ]
        )

        with (
            patch(
                "yanex.cli.commands.compare.find_experiments_by_identifiers"
            ) as mock_find,
            patch.object(ExperimentComparisonData, "get_comparison_data") as mock_data,
        ):
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data

            result = self.runner.invoke(
                compare_experiments,
                ["exp001", "exp002"] + flag_options + ["--no-interactive"],
            )

            assert result.exit_code == 0
            # Should show some expected content (truncated headers)
            assert "exp0" in result.output  # experiment IDs

    def test_compare_max_rows_limit(self):
        """Test --max-rows option."""
        # Create multiple experiments
        experiments = []
        for i in range(5):
            exp_id = f"exp{i:03d}"
            metadata = {"id": exp_id, "name": f"experiment-{i}", "status": "completed"}
            experiments.append(metadata)

        with patch(
            "yanex.cli.commands.compare.find_experiments_by_filters"
        ) as mock_find:
            mock_find.return_value = experiments

            result = self.runner.invoke(
                compare_experiments, ["--max-rows", "3", "--no-interactive"]
            )

            assert result.exit_code == 0
            assert "Limiting display to first 3 experiments" in result.output

    def test_compare_no_comparison_data(self):
        """Test when no comparison data is available."""
        with (
            patch(
                "yanex.cli.commands.compare.find_experiments_by_filters"
            ) as mock_find,
            patch.object(ExperimentComparisonData, "get_comparison_data") as mock_data,
        ):
            mock_find.return_value = [{"id": "exp001", "name": "test"}]
            mock_data.return_value = {"rows": []}  # No data

            result = self.runner.invoke(compare_experiments, ["--no-interactive"])

            assert result.exit_code == 0
            assert "No comparison data available" in result.output


class TestCompareCommandIntegration:
    """Test compare command integration with real data processing."""

    def test_compare_data_extraction_integration(self, tmp_path):
        """Test complete data extraction and comparison functionality."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        # Create realistic test experiments using utilities
        exp1_metadata = TestDataFactory.create_experiment_metadata(
            experiment_id="exp001",
            name="baseline-model",
            status="completed",
            started_at="2025-01-01T10:00:00Z",
            completed_at="2025-01-01T11:30:00Z",
            duration=5400,
            tags=["baseline", "training"],
        )
        exp1_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_type": "transformer",
        }
        exp1_results = {
            "final_accuracy": 0.8934,
            "final_loss": 0.2156,
            "best_epoch": 8,
            "training_time": 5367,
        }

        exp2_metadata = TestDataFactory.create_experiment_metadata(
            experiment_id="exp002",
            name="improved-model",
            status="completed",
            started_at="2025-01-01T14:00:00Z",
            completed_at="2025-01-01T15:45:00Z",
            duration=6300,
            tags=["improved", "training"],
        )
        exp2_config = {
            "learning_rate": 0.0005,
            "batch_size": 64,
            "epochs": 15,
            "model_type": "transformer",
        }
        exp2_results = {
            "final_accuracy": 0.9245,
            "final_loss": 0.1432,
            "best_epoch": 12,
            "training_time": 6234,
        }

        # Create experiment directories with all files
        TestFileHelpers.create_experiment_files(
            experiments_dir / "exp001", exp1_metadata, exp1_config, exp1_results
        )
        TestFileHelpers.create_experiment_files(
            experiments_dir / "exp002", exp2_metadata, exp2_config, exp2_results
        )

        # Test with real ExperimentComparisonData
        comparison = ExperimentComparisonData(ExperimentManager(experiments_dir))
        comparison_data = comparison.get_comparison_data(["exp001", "exp002"])

        # Verify data structure
        assert comparison_data["total_experiments"] == 2
        assert len(comparison_data["rows"]) == 2

        # Check parameter columns
        assert "learning_rate" in comparison_data["param_columns"]
        assert "batch_size" in comparison_data["param_columns"]
        assert "epochs" in comparison_data["param_columns"]

        # Check metric columns
        assert "final_accuracy" in comparison_data["metric_columns"]
        assert "final_loss" in comparison_data["metric_columns"]
        assert "best_epoch" in comparison_data["metric_columns"]

        # Check row data format
        rows = comparison_data["rows"]
        exp1_row = next(row for row in rows if row["id"] == "exp001")
        exp2_row = next(row for row in rows if row["id"] == "exp002")

        # Verify basic columns
        assert exp1_row["name"] == "baseline-model"
        assert exp1_row["status"] == "completed"
        assert exp1_row["tags"] == "baseline, training"
        assert exp1_row["duration"] == "01:30:00"  # Formatted duration

        # Verify parameter columns
        assert exp1_row["param:learning_rate"] == "0.001"
        assert exp1_row["param:batch_size"] == "32"
        assert exp2_row["param:learning_rate"] == "0.0005"
        assert exp2_row["param:batch_size"] == "64"

        # Verify metric columns
        assert exp1_row["metric:final_accuracy"] == "0.8934"
        assert exp1_row["metric:final_loss"] == "0.2156"
        assert exp2_row["metric:final_accuracy"] == "0.9245"
        assert exp2_row["metric:final_loss"] == "0.1432"

    def test_compare_only_different_filtering_integration(self, tmp_path):
        """Test --only-different filtering with real data."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        # Create experiments with some identical parameters
        exp1_metadata = TestDataFactory.create_experiment_metadata(
            experiment_id="exp001", name="model-a", status="completed"
        )
        exp1_config = {
            "learning_rate": 0.001,  # Different
            "batch_size": 32,  # Same
            "model_type": "bert",  # Same
            "optimizer": "adam",  # Different
        }
        exp1_results = {
            "accuracy": 0.91,  # Different
            "epochs_trained": 10,  # Same
        }

        exp2_metadata = TestDataFactory.create_experiment_metadata(
            experiment_id="exp002", name="model-b", status="completed"
        )
        exp2_config = {
            "learning_rate": 0.002,  # Different
            "batch_size": 32,  # Same
            "model_type": "bert",  # Same
            "optimizer": "sgd",  # Different
        }
        exp2_results = {
            "accuracy": 0.89,  # Different
            "epochs_trained": 10,  # Same
        }

        TestFileHelpers.create_experiment_files(
            experiments_dir / "exp001", exp1_metadata, exp1_config, exp1_results
        )
        TestFileHelpers.create_experiment_files(
            experiments_dir / "exp002", exp2_metadata, exp2_config, exp2_results
        )

        # Test with only_different=True
        comparison = ExperimentComparisonData(ExperimentManager(experiments_dir))
        comparison_data = comparison.get_comparison_data(
            ["exp001", "exp002"], only_different=True
        )

        # Should only include parameters/metrics with different values
        expected_params = {"learning_rate", "optimizer"}  # Different values
        expected_metrics = {"accuracy"}  # Different values

        assert set(comparison_data["param_columns"]) == expected_params
        assert set(comparison_data["metric_columns"]) == expected_metrics

        # Should not include batch_size, model_type, epochs_trained (same values)
        assert "batch_size" not in comparison_data["param_columns"]
        assert "model_type" not in comparison_data["param_columns"]
        assert "epochs_trained" not in comparison_data["metric_columns"]

    @pytest.mark.parametrize(
        "config_data,results_data,expected_param_types,expected_metric_types",
        [
            (
                {
                    "learning_rate": 0.001,  # Numeric
                    "model_name": "bert-base",  # String
                    "use_gpu": True,  # Boolean -> String
                    "layers": 12,  # Numeric
                },
                {
                    "accuracy": 0.923,  # Numeric
                    "model_size": "110M",  # String
                    "training_steps": 1000,  # Numeric
                },
                {
                    "param:learning_rate": "numeric",
                    "param:model_name": "string",
                    "param:use_gpu": "string",  # Boolean formatted as string
                    "param:layers": "numeric",
                },
                {
                    "metric:accuracy": "numeric",
                    "metric:model_size": "string",
                    "metric:training_steps": "numeric",
                },
            ),
        ],
    )
    def test_compare_column_type_inference_integration(
        self,
        tmp_path,
        config_data,
        results_data,
        expected_param_types,
        expected_metric_types,
    ):
        """Test column type inference with real data."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        exp_metadata = TestDataFactory.create_experiment_metadata(
            experiment_id="exp001", name="test-exp", status="completed"
        )

        TestFileHelpers.create_experiment_files(
            experiments_dir / "exp001", exp_metadata, config_data, results_data
        )

        comparison = ExperimentComparisonData(ExperimentManager(experiments_dir))
        comparison_data = comparison.get_comparison_data(["exp001"])

        column_types = comparison_data["column_types"]

        # Check fixed column types
        assert column_types["id"] == "string"
        assert column_types["name"] == "string"
        assert column_types["started"] == "datetime"
        assert column_types["duration"] == "string"
        assert column_types["status"] == "string"
        assert column_types["tags"] == "string"

        # Check inferred parameter types
        for param_col, expected_type in expected_param_types.items():
            assert column_types[param_col] == expected_type

        # Check inferred metric types
        for metric_col, expected_type in expected_metric_types.items():
            assert column_types[metric_col] == expected_type
