"""
Tests for experiment comparison data extraction.
"""

import json
import tempfile
from pathlib import Path

from yanex.core.comparison import ExperimentComparisonData
from yanex.core.manager import ExperimentManager


class TestExperimentComparisonData:
    """Test ExperimentComparisonData class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir) / "experiments"
        self.experiments_dir.mkdir(parents=True)

        self.manager = ExperimentManager(self.experiments_dir)
        self.comparison = ExperimentComparisonData(self.manager)

    def _create_test_experiment(
        self, exp_id: str, metadata: dict, config: dict = None, results: dict = None
    ):
        """Create a test experiment with given data."""
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with (exp_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f)

        # Save config if provided
        if config is not None:
            import yaml

            with (exp_dir / "config.yaml").open("w") as f:
                yaml.dump(config, f)

        # Save results if provided
        if results is not None:
            with (exp_dir / "results.json").open("w") as f:
                json.dump(results, f)

    def test_extract_single_experiment_complete(self):
        """Test extracting complete experiment data."""
        exp_id = "test123"
        metadata = {
            "id": exp_id,
            "name": "test-exp",
            "status": "completed",
            "start_time": "2025-01-01T10:00:00Z",
            "end_time": "2025-01-01T11:00:00Z",
            "script_path": "train.py",
            "tags": ["test", "training"],
        }
        config = {"learning_rate": 0.01, "epochs": 10, "model_type": "linear"}
        results = {"accuracy": 0.95, "loss": 0.05, "final_score": 0.92}

        self._create_test_experiment(exp_id, metadata, config, results)

        exp_data = self.comparison._extract_single_experiment(exp_id)

        assert exp_data is not None
        assert exp_data["id"] == exp_id
        assert exp_data["name"] == "test-exp"
        assert exp_data["status"] == "completed"
        assert exp_data["config"] == config
        assert exp_data["results"] == results
        assert exp_data["tags"] == ["test", "training"]

    def test_extract_single_experiment_missing_files(self):
        """Test extracting experiment with missing config/results files."""
        exp_id = "test456"
        metadata = {"id": exp_id, "name": "minimal-exp", "status": "failed"}

        self._create_test_experiment(exp_id, metadata)  # No config or results

        exp_data = self.comparison._extract_single_experiment(exp_id)

        assert exp_data is not None
        assert exp_data["id"] == exp_id
        assert exp_data["name"] == "minimal-exp"
        assert exp_data["status"] == "failed"
        assert exp_data["config"] == {}
        assert exp_data["results"] == {}

    def test_discover_columns_auto_discovery(self):
        """Test automatic column discovery."""
        experiments_data = [
            {
                "config": {"learning_rate": 0.01, "epochs": 10},
                "results": {"accuracy": 0.95, "loss": 0.05},
            },
            {
                "config": {"learning_rate": 0.02, "batch_size": 32},
                "results": {"accuracy": 0.92, "f1_score": 0.88},
            },
        ]

        param_columns, metric_columns = self.comparison.discover_columns(
            experiments_data
        )

        assert set(param_columns) == {"batch_size", "epochs", "learning_rate"}
        assert set(metric_columns) == {"accuracy", "f1_score", "loss"}

    def test_discover_columns_specified(self):
        """Test column discovery with specified parameters/metrics."""
        experiments_data = [
            {
                "config": {"learning_rate": 0.01, "epochs": 10, "batch_size": 32},
                "results": {"accuracy": 0.95, "loss": 0.05, "f1_score": 0.88},
            }
        ]

        param_columns, metric_columns = self.comparison.discover_columns(
            experiments_data, params=["learning_rate", "epochs"], metrics=["accuracy"]
        )

        assert param_columns == ["learning_rate", "epochs"]
        assert metric_columns == ["accuracy"]

    def test_build_comparison_matrix(self):
        """Test building comparison matrix."""
        exp_data = {
            "id": "test123",
            "metadata": {
                "start_time": "2025-01-01T10:00:00Z",
                "end_time": "2025-01-01T11:00:00Z",
            },
            "name": "test-exp",
            "status": "completed",
            "script_path": "train.py",
            "config": {"learning_rate": 0.01},
            "results": {"accuracy": 0.95},
        }

        rows = self.comparison.build_comparison_matrix(
            [exp_data], ["learning_rate"], ["accuracy"]
        )

        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == "test123"
        assert row["name"] == "test-exp"
        assert row["status"] == "completed"
        assert row["param:learning_rate"] == "0.01"
        assert row["metric:accuracy"] == "0.95"

    def test_format_value_various_types(self):
        """Test value formatting for different data types."""
        # Test None
        assert self.comparison._format_value(None) == "-"

        # Test boolean
        assert self.comparison._format_value(True) == "true"
        assert self.comparison._format_value(False) == "false"

        # Test integers
        assert self.comparison._format_value(42) == "42"

        # Test floats
        assert self.comparison._format_value(3.14159) == "3.1416"
        assert self.comparison._format_value(0.001234) == "0.001234"
        assert self.comparison._format_value(1234.5) == "1234.5"

        # Test lists
        assert self.comparison._format_value([1, 2, 3]) == "1, 2, 3"

        # Test dicts
        assert self.comparison._format_value({"a": 1, "b": 2}) == "a=1, b=2"

    def test_filter_different_columns(self):
        """Test filtering columns with identical values."""
        comparison_rows = [
            {
                "param:learning_rate": "0.01",
                "param:epochs": "10",
                "param:model_type": "linear",
                "metric:accuracy": "0.95",
                "metric:loss": "0.05",
            },
            {
                "param:learning_rate": "0.02",
                "param:epochs": "10",  # Same value
                "param:model_type": "linear",  # Same value
                "metric:accuracy": "0.92",
                "metric:loss": "0.08",
            },
        ]

        filtered_params, filtered_metrics = self.comparison.filter_different_columns(
            comparison_rows,
            ["learning_rate", "epochs", "model_type"],
            ["accuracy", "loss"],
        )

        # Only learning_rate should remain (epochs and model_type are identical)
        assert filtered_params == ["learning_rate"]
        # Both metrics should remain (both have different values)
        assert set(filtered_metrics) == {"accuracy", "loss"}

    def test_infer_column_types(self):
        """Test column type inference."""
        comparison_rows = [
            {
                "param:learning_rate": "0.01",
                "param:epochs": "10",
                "param:model_type": "linear",
                "metric:accuracy": "0.95",
            },
            {
                "param:learning_rate": "0.02",
                "param:epochs": "20",
                "param:model_type": "neural",
                "metric:accuracy": "0.92",
            },
        ]

        column_types = self.comparison.infer_column_types(
            comparison_rows, ["learning_rate", "epochs", "model_type"], ["accuracy"]
        )

        assert column_types["param:learning_rate"] == "numeric"
        assert column_types["param:epochs"] == "numeric"
        assert column_types["param:model_type"] == "string"
        assert column_types["metric:accuracy"] == "numeric"

    def test_get_comparison_data_complete(self):
        """Test complete comparison data extraction."""
        # Create test experiments
        exp1_id = "exp001"
        exp1_metadata = {
            "id": exp1_id,
            "name": "experiment-1",
            "status": "completed",
            "start_time": "2025-01-01T10:00:00Z",
            "end_time": "2025-01-01T11:00:00Z",
        }
        exp1_config = {"learning_rate": 0.01, "epochs": 10}
        exp1_results = {"accuracy": 0.95, "loss": 0.05}

        exp2_id = "exp002"
        exp2_metadata = {
            "id": exp2_id,
            "name": "experiment-2",
            "status": "failed",
            "start_time": "2025-01-01T12:00:00Z",
            "end_time": "2025-01-01T12:30:00Z",
        }
        exp2_config = {"learning_rate": 0.02, "epochs": 10}
        exp2_results = {"accuracy": 0.87, "loss": 0.13}

        self._create_test_experiment(exp1_id, exp1_metadata, exp1_config, exp1_results)
        self._create_test_experiment(exp2_id, exp2_metadata, exp2_config, exp2_results)

        # Get comparison data
        comparison_data = self.comparison.get_comparison_data(
            [exp1_id, exp2_id], only_different=True
        )

        assert comparison_data["total_experiments"] == 2
        assert len(comparison_data["rows"]) == 2

        # Check that epochs is filtered out (same value) but learning_rate remains
        assert "learning_rate" in comparison_data["param_columns"]
        assert "epochs" not in comparison_data["param_columns"]  # Filtered out

        # Check that both metrics remain (different values)
        assert set(comparison_data["metric_columns"]) == {"accuracy", "loss"}

    def test_calculate_duration(self):
        """Test duration calculation."""
        start_time = "2025-01-01T10:00:00Z"
        end_time = "2025-01-01T11:30:45Z"

        duration = self.comparison._calculate_duration(start_time, end_time)
        assert duration == "01:30:45"

        # Test running experiment
        duration = self.comparison._calculate_duration(start_time, None)
        assert duration == "[running]"

        # Test no start time
        duration = self.comparison._calculate_duration(None, end_time)
        assert duration == "-"

    def test_format_datetime(self):
        """Test datetime formatting."""
        dt_str = "2025-01-01T10:30:45Z"
        formatted = self.comparison._format_datetime(dt_str)
        assert formatted == "2025-01-01 10:30:45"

        # Test None
        formatted = self.comparison._format_datetime(None)
        assert formatted == "-"
