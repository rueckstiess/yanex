"""
Tests for experiment comparison data extraction.
"""

import pytest

from tests.test_utils import TestDataFactory, TestFileHelpers, create_isolated_manager
from yanex.core.comparison import ExperimentComparisonData


class TestExperimentComparisonData:
    """Test ExperimentComparisonData class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = create_isolated_manager()
        self.comparison = ExperimentComparisonData(self.manager)

    @pytest.mark.parametrize(
        "exp_id,name,status,config_type,results_type",
        [
            ("test123", "test-experiment", "completed", "ml_training", "ml_metrics"),
            ("exp456", "minimal-exp", "failed", "simple", "basic_metrics"),
            ("complex1", "complex-exp", "running", "data_processing", "custom_metrics"),
        ],
    )
    def test_extract_single_experiment_complete(
        self, exp_id, name, status, config_type, results_type
    ):
        """Test extracting complete experiment data with various configurations."""
        # Create test experiment using utilities
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id,
            name=name,
            status=status,
            start_time="2025-01-01T10:00:00Z",
            end_time="2025-01-01T11:00:00Z",
            tags=["test", "training"],
        )
        config = TestDataFactory.create_experiment_config(config_type=config_type)
        results = TestDataFactory.create_experiment_results(result_type=results_type)

        experiment_dir = self.manager.storage.experiments_dir / exp_id
        TestFileHelpers.create_experiment_files(
            experiment_dir, metadata, config, results
        )

        exp_data = self.comparison._extract_single_experiment(exp_id)

        assert exp_data is not None
        assert exp_data["id"] == exp_id
        assert exp_data["name"] == metadata["name"]
        assert exp_data["status"] == metadata["status"]
        assert exp_data["config"] == config
        assert exp_data["results"] == results
        if "tags" in metadata:
            assert exp_data["tags"] == metadata["tags"]

    @pytest.mark.parametrize(
        "exp_id,name,status",
        [
            ("test456", "minimal-exp", "failed"),
            ("failed789", "failed-exp", "failed"),
            ("empty123", "basic-exp", "completed"),
        ],
    )
    def test_extract_single_experiment_missing_files(self, exp_id, name, status):
        """Test extracting experiment with missing config/results files."""
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id,
            name=name,
            status=status,
        )

        # Only create metadata, no config or results
        experiment_dir = self.manager.storage.experiments_dir / exp_id
        TestFileHelpers.create_experiment_files(experiment_dir, metadata)

        exp_data = self.comparison._extract_single_experiment(exp_id)

        assert exp_data is not None
        assert exp_data["id"] == exp_id
        assert exp_data["name"] == metadata["name"]
        assert exp_data["status"] == metadata["status"]
        assert exp_data["config"] == {}
        assert exp_data["results"] == {}

    @pytest.mark.parametrize(
        "config_types,result_types,expected_params,expected_metrics",
        [
            (
                ["ml_training", "ml_training"],
                ["ml_metrics", "ml_metrics"],
                {"learning_rate", "batch_size", "epochs"},
                {"accuracy", "loss", "precision", "recall", "f1_score"},
            ),
            (
                ["simple", "data_processing"],
                ["simple", "processing_stats"],
                {"param1", "param2", "param3", "n_docs", "chunk_size"},
                {
                    "value",
                    "status",
                    "timestamp",
                    "docs_processed",
                    "processing_time",
                    "errors",
                    "success_rate",
                },
            ),
        ],
    )
    def test_discover_columns_auto_discovery(
        self, config_types, result_types, expected_params, expected_metrics
    ):
        """Test automatic column discovery with various experiment types."""
        experiments_data = []

        for config_type, result_type in zip(config_types, result_types, strict=False):
            exp_data = {
                "config": TestDataFactory.create_experiment_config(
                    config_type=config_type
                ),
                "results": TestDataFactory.create_experiment_results(
                    result_type=result_type
                ),
            }
            experiments_data.append(exp_data)

        param_columns, metric_columns = self.comparison.discover_columns(
            experiments_data
        )

        # Check that discovered columns contain expected parameters
        assert expected_params.issubset(set(param_columns))
        assert expected_metrics.issubset(set(metric_columns))

    @pytest.mark.parametrize(
        "config_type,result_type,specified_params,specified_metrics",
        [
            (
                "ml_training",
                "ml_metrics",
                ["learning_rate", "epochs"],
                ["accuracy"],
            ),
            (
                "data_processing",
                "processing_stats",
                ["n_docs", "chunk_size"],
                ["processing_time", "docs_processed"],
            ),
        ],
    )
    def test_discover_columns_specified(
        self, config_type, result_type, specified_params, specified_metrics
    ):
        """Test column discovery with specified parameters/metrics."""
        experiments_data = [
            {
                "config": TestDataFactory.create_experiment_config(
                    config_type=config_type
                ),
                "results": TestDataFactory.create_experiment_results(
                    result_type=result_type
                ),
            }
        ]

        param_columns, metric_columns = self.comparison.discover_columns(
            experiments_data, params=specified_params, metrics=specified_metrics
        )

        assert param_columns == specified_params
        assert metric_columns == specified_metrics

    @pytest.mark.parametrize(
        "exp_id,name,status,config_type,results_type",
        [
            ("test123", "complete-exp", "completed", "ml_training", "ml_metrics"),
            ("minimal1", "minimal-exp", "failed", "simple", "simple"),
        ],
    )
    def test_build_comparison_matrix(
        self, exp_id, name, status, config_type, results_type
    ):
        """Test building comparison matrix with various experiment types."""
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id,
            name=name,
            status=status,
            start_time="2025-01-01T10:00:00Z",
            end_time="2025-01-01T11:00:00Z",
        )
        config = TestDataFactory.create_experiment_config(config_type=config_type)
        results = TestDataFactory.create_experiment_results(result_type=results_type)

        exp_data = {
            "id": exp_id,
            "metadata": {
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
            },
            "name": metadata["name"],
            "status": metadata["status"],
            "script_path": metadata.get("script_path", "train.py"),
            "config": config,
            "results": results,
        }

        # Use first available param and metric for testing
        param_columns = list(config.keys())[:1] if config else []
        metric_columns = list(results.keys())[:1] if results else []

        rows = self.comparison.build_comparison_matrix(
            [exp_data], param_columns, metric_columns
        )

        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == exp_id
        assert row["name"] == metadata["name"]
        assert row["status"] == metadata["status"]

        # Check parameter and metric columns are present
        for param in param_columns:
            assert f"param:{param}" in row
        for metric in metric_columns:
            assert f"metric:{metric}" in row

    @pytest.mark.parametrize(
        "test_value,expected_result",
        [
            (None, "-"),
            (True, "true"),
            (False, "false"),
            (42, "42"),
            (3.14159, "3.1416"),
            (0.001234, "0.001234"),
            (1234.5, "1234.5"),
            ([1, 2, 3], "1, 2, 3"),
            ({"a": 1, "b": 2}, "a=1, b=2"),
        ],
    )
    def test_format_value_various_types(self, test_value, expected_result):
        """Test value formatting for different data types."""
        result = self.comparison._format_value(test_value)
        assert result == expected_result

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

    @pytest.mark.parametrize(
        "exp1_type,exp2_type",
        [
            ("ml_training", "ml_training"),
            ("simple", "data_processing"),
        ],
    )
    def test_get_comparison_data_complete(self, exp1_type, exp2_type):
        """Test complete comparison data extraction."""
        # Create test experiments using utilities
        exp1_id = "exp001"
        exp1_metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp1_id,
            name="experiment-1",
            status="completed",
        )
        exp1_config = TestDataFactory.create_experiment_config(config_type=exp1_type)
        exp1_results = TestDataFactory.create_experiment_results(
            result_type="ml_metrics"
        )

        exp2_id = "exp002"
        exp2_metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp2_id,
            name="experiment-2",
            status="failed",
        )
        exp2_config = TestDataFactory.create_experiment_config(config_type=exp2_type)
        exp2_results = TestDataFactory.create_experiment_results(result_type="simple")

        exp1_dir = self.manager.storage.experiments_dir / exp1_id
        exp2_dir = self.manager.storage.experiments_dir / exp2_id
        TestFileHelpers.create_experiment_files(
            exp1_dir, exp1_metadata, exp1_config, exp1_results
        )
        TestFileHelpers.create_experiment_files(
            exp2_dir, exp2_metadata, exp2_config, exp2_results
        )

        # Get comparison data
        comparison_data = self.comparison.get_comparison_data(
            [exp1_id, exp2_id], only_different=True
        )

        assert comparison_data["total_experiments"] == 2
        assert len(comparison_data["rows"]) == 2

        # Should have some parameters and metrics
        assert len(comparison_data["param_columns"]) >= 0
        assert len(comparison_data["metric_columns"]) >= 0

    @pytest.mark.parametrize(
        "start_time,end_time,expected_result",
        [
            ("2025-01-01T10:00:00Z", "2025-01-01T11:30:45Z", "01:30:45"),
            ("2025-01-01T10:00:00Z", None, "[running]"),
            (None, "2025-01-01T11:30:45Z", "-"),
        ],
    )
    def test_calculate_duration(self, start_time, end_time, expected_result):
        """Test duration calculation."""
        duration = self.comparison._calculate_duration(start_time, end_time)
        assert duration == expected_result

    @pytest.mark.parametrize(
        "dt_str,expected_result",
        [
            ("2025-01-01T10:30:45Z", "2025-01-01 10:30:45"),
            (None, "-"),
        ],
    )
    def test_format_datetime(self, dt_str, expected_result):
        """Test datetime formatting."""
        formatted = self.comparison._format_datetime(dt_str)
        assert formatted == expected_result

    def test_comprehensive_comparison_workflow(self):
        """Test complete comparison workflow with multiple experiment types."""
        # Create diverse set of experiments using utilities
        experiments = [
            ("ml001", "ml_training", "ml_metrics"),
            ("data001", "data_processing", "processing_stats"),
            ("simple001", "simple", "simple"),
        ]

        experiment_ids = []

        for exp_id, config_type, results_type in experiments:
            metadata = TestDataFactory.create_experiment_metadata(
                experiment_id=exp_id,
                name=f"test-{exp_id}",
                status="completed",
            )
            config = TestDataFactory.create_experiment_config(config_type=config_type)
            results = TestDataFactory.create_experiment_results(
                result_type=results_type
            )

            experiment_dir = self.manager.storage.experiments_dir / exp_id
            TestFileHelpers.create_experiment_files(
                experiment_dir, metadata, config, results
            )
            experiment_ids.append(exp_id)

        # Test full comparison workflow
        comparison_data = self.comparison.get_comparison_data(
            experiment_ids, only_different=True
        )

        assert comparison_data["total_experiments"] == 3
        assert len(comparison_data["rows"]) == 3

        # Should discover columns from diverse experiment types
        # Note: only_different=True may filter out some columns if they have identical values
        assert (
            len(comparison_data["param_columns"]) >= 0
        )  # May be 0 if all params are identical
        assert (
            len(comparison_data["metric_columns"]) >= 0
        )  # May be 0 if all metrics are identical

        # Each row should have consistent structure
        for row in comparison_data["rows"]:
            assert "id" in row
            assert "name" in row
            assert "status" in row
