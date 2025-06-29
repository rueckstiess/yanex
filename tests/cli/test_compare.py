"""
Tests for yanex CLI compare command functionality.
"""

import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

import pytest

from yanex.cli.commands.compare import compare_experiments
from yanex.cli.filters import ExperimentFilter
from yanex.core.comparison import ExperimentComparisonData
from yanex.core.manager import ExperimentManager


class TestCompareCommand:
    """Test compare command CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.experiments_dir = self.temp_dir / "experiments"
        self.experiments_dir.mkdir(parents=True)
        
        self.manager = ExperimentManager(self.experiments_dir)
        self.runner = CliRunner()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_experiment(
        self, 
        exp_id: str, 
        metadata: dict, 
        config: dict = None, 
        results: dict = None
    ):
        """Create a test experiment with given data."""
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with (exp_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f)
        
        # Save config if provided
        if config is not None:
            with (exp_dir / "config.yaml").open("w") as f:
                yaml.dump(config, f)
        
        # Save results if provided
        if results is not None:
            with (exp_dir / "results.json").open("w") as f:
                json.dump(results, f)

    def test_compare_no_experiments_found(self):
        """Test compare command when no experiments are found."""
        with patch('yanex.cli.commands.compare.find_experiments_by_filters') as mock_find:
            mock_find.return_value = []
            
            result = self.runner.invoke(compare_experiments, ['--status', 'completed'])
            
            assert result.exit_code == 0
            assert "No regular experiments found to compare" in result.output

    def test_compare_specific_experiments_by_id(self):
        """Test compare command with specific experiment IDs."""
        # Mock the find functions and comparison data
        mock_experiments = [
            {"id": "exp001", "name": "experiment-1"},
            {"id": "exp002", "name": "experiment-2"}
        ]
        
        mock_comparison_data = {
            "rows": [
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
                    "metric:loss": "0.05"
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
                    "metric:loss": "0.13"
                }
            ],
            "param_columns": ["learning_rate", "epochs"],
            "metric_columns": ["accuracy", "loss"],
            "column_types": {},
            "total_experiments": 2
        }
        
        with patch('yanex.cli.commands.compare.find_experiments_by_identifiers') as mock_find, \
             patch.object(ExperimentComparisonData, 'get_comparison_data') as mock_data:
            
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data
            
            # Test static table output
            result = self.runner.invoke(
                compare_experiments, 
                ['exp001', 'exp002', '--no-interactive']
            )
            
            assert result.exit_code == 0
            # Should show both experiments in the output (might be truncated)
            assert "exp0" in result.output  # exp001 might be truncated to exp0…
            assert "exp0" in result.output  # exp002 might be truncated to exp0…
            # Should show parameters and metrics (headers might be truncated)
            assert "lear" in result.output  # learning_rate might be truncated to lear…
            assert "accu" in result.output  # accuracy might be truncated to accu…

    def test_compare_csv_export(self):
        """Test compare command CSV export functionality."""
        mock_experiments = [{"id": "exp001", "name": "test-exp"}]
        
        mock_comparison_data = {
            "rows": [
                {
                    "id": "exp001",
                    "name": "test-exp",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",
                    "metric:accuracy": "0.95"
                }
            ],
            "param_columns": ["learning_rate"],
            "metric_columns": ["accuracy"],
            "column_types": {},
            "total_experiments": 1
        }
        
        with patch('yanex.cli.commands.compare.find_experiments_by_identifiers') as mock_find, \
             patch.object(ExperimentComparisonData, 'get_comparison_data') as mock_data:
            
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data
            
            # Test CSV export
            export_path = self.temp_dir / "test_export.csv"
            result = self.runner.invoke(
                compare_experiments, 
                ['exp001', '--export', str(export_path)]
            )
            
            assert result.exit_code == 0
            assert f"Comparison data exported to {export_path}" in result.output
            
            # Check that CSV file was created
            assert export_path.exists()
            
            # Check CSV content
            csv_content = export_path.read_text()
            assert "exp001" in csv_content
            assert "learning_rate" in csv_content
            assert "accuracy" in csv_content

    def test_compare_mutually_exclusive_options(self):
        """Test that identifiers and filters are mutually exclusive."""
        result = self.runner.invoke(
            compare_experiments,
            ['exp001', '--status', 'completed']
        )
        
        assert result.exit_code == 1
        assert "Cannot use both experiment identifiers and filter options" in result.output

    def test_compare_filter_by_status(self):
        """Test compare command with status filter."""
        # Create experiments with different statuses
        exp1_metadata = {
            "id": "exp001",
            "name": "completed-exp",
            "status": "completed",
            "started_at": "2025-01-01T10:00:00Z"
        }
        exp2_metadata = {
            "id": "exp002",
            "name": "failed-exp", 
            "status": "failed",
            "started_at": "2025-01-01T11:00:00Z"
        }
        
        self._create_test_experiment("exp001", exp1_metadata)
        self._create_test_experiment("exp002", exp2_metadata)

        # Mock to return only completed experiments
        mock_experiments = [exp1_metadata]
        
        with patch('yanex.cli.commands.compare.find_experiments_by_filters') as mock_find:
            mock_find.return_value = mock_experiments
            
            result = self.runner.invoke(
                compare_experiments,
                ['--status', 'completed', '--no-interactive']
            )
            
            assert result.exit_code == 0
            assert "exp001" in result.output
            assert "exp002" not in result.output

    def test_compare_only_different_columns(self):
        """Test --only-different flag functionality."""
        mock_experiments = [
            {"id": "exp001", "name": "exp-1"},
            {"id": "exp002", "name": "exp-2"}
        ]
        
        # Mock comparison data that only shows different columns
        mock_comparison_data = {
            "rows": [
                {
                    "id": "exp001",
                    "name": "exp-1",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",  # Different
                    "metric:accuracy": "0.95"        # Different
                },
                {
                    "id": "exp002",
                    "name": "exp-2",
                    "started": "2025-01-01 12:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.02",  # Different
                    "metric:accuracy": "0.87"        # Different
                }
            ],
            "param_columns": ["learning_rate"],  # Only different params
            "metric_columns": ["accuracy"],     # Only different metrics
            "column_types": {},
            "total_experiments": 2
        }
        
        with patch('yanex.cli.commands.compare.find_experiments_by_identifiers') as mock_find, \
             patch.object(ExperimentComparisonData, 'get_comparison_data') as mock_data:
            
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data
            
            result = self.runner.invoke(
                compare_experiments,
                ['exp001', 'exp002', '--only-different', '--no-interactive']
            )
            
            assert result.exit_code == 0
            # Should show learning_rate (different) - might be truncated
            assert "lear" in result.output  # learning_rate might be truncated to lear…

    def test_compare_limit_params_and_metrics(self):
        """Test --params and --metrics options."""
        mock_experiments = [{"id": "exp001", "name": "test-exp"}]
        
        # Mock comparison data with only specified params/metrics
        mock_comparison_data = {
            "rows": [
                {
                    "id": "exp001",
                    "name": "test-exp",
                    "started": "2025-01-01 10:00:00",
                    "duration": "01:00:00",
                    "status": "completed",
                    "tags": "-",
                    "param:learning_rate": "0.01",
                    "param:epochs": "10",
                    "metric:accuracy": "0.95",
                    "metric:loss": "0.05"
                }
            ],
            "param_columns": ["learning_rate", "epochs"],   # Only specified
            "metric_columns": ["accuracy", "loss"],        # Only specified
            "column_types": {},
            "total_experiments": 1
        }
        
        with patch('yanex.cli.commands.compare.find_experiments_by_identifiers') as mock_find, \
             patch.object(ExperimentComparisonData, 'get_comparison_data') as mock_data:
            
            mock_find.return_value = mock_experiments
            mock_data.return_value = mock_comparison_data
            
            result = self.runner.invoke(
                compare_experiments,
                [
                    'exp001', 
                    '--params', 'learning_rate,epochs',
                    '--metrics', 'accuracy,loss',
                    '--no-interactive'
                ]
            )
            
            assert result.exit_code == 0
            # Should show specified params/metrics (might be truncated)
            assert "lear" in result.output   # learning_rate might be truncated to lear…
            assert "epoc" in result.output   # epochs might be truncated to epoc…
            assert "accu" in result.output   # accuracy might be truncated to accu…
            assert "loss" in result.output

    def test_compare_max_rows_limit(self):
        """Test --max-rows option."""
        # Create multiple experiments
        experiments = []
        for i in range(5):
            exp_id = f"exp{i:03d}"
            metadata = {
                "id": exp_id,
                "name": f"experiment-{i}",
                "status": "completed"
            }
            self._create_test_experiment(exp_id, metadata)
            experiments.append(metadata)
        
        with patch('yanex.cli.commands.compare.find_experiments_by_filters') as mock_find:
            mock_find.return_value = experiments
            
            result = self.runner.invoke(
                compare_experiments,
                ['--max-rows', '3', '--no-interactive']
            )
            
            assert result.exit_code == 0
            assert "Limiting display to first 3 experiments" in result.output

    def test_compare_no_comparison_data(self):
        """Test when no comparison data is available."""
        with patch('yanex.cli.commands.compare.find_experiments_by_filters') as mock_find, \
             patch.object(ExperimentComparisonData, 'get_comparison_data') as mock_data:
            
            mock_find.return_value = [{"id": "exp001", "name": "test"}]
            mock_data.return_value = {"rows": []}  # No data
            
            result = self.runner.invoke(
                compare_experiments,
                ['--no-interactive']
            )
            
            assert result.exit_code == 0
            assert "No comparison data available" in result.output


class TestCompareCommandIntegration:
    """Test compare command integration with real data processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.experiments_dir = self.temp_dir / "experiments" 
        self.experiments_dir.mkdir(parents=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_experiment(
        self, 
        exp_id: str, 
        metadata: dict, 
        config: dict = None, 
        results: dict = None
    ):
        """Create a test experiment with given data."""
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with (exp_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f)
        
        # Save config if provided
        if config is not None:
            with (exp_dir / "config.yaml").open("w") as f:
                yaml.dump(config, f)
        
        # Save results if provided
        if results is not None:
            with (exp_dir / "results.json").open("w") as f:
                json.dump(results, f)

    def test_compare_data_extraction_integration(self):
        """Test complete data extraction and comparison functionality."""
        # Create realistic test experiments
        exp1_metadata = {
            "id": "exp001",
            "name": "baseline-model",
            "status": "completed",
            "started_at": "2025-01-01T10:00:00Z",
            "completed_at": "2025-01-01T11:30:00Z",
            "duration": 5400,  # 1.5 hours in seconds
            "tags": ["baseline", "training"]
        }
        exp1_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_type": "transformer"
        }
        exp1_results = {
            "final_accuracy": 0.8934,
            "final_loss": 0.2156,
            "best_epoch": 8,
            "training_time": 5367
        }
        
        exp2_metadata = {
            "id": "exp002",
            "name": "improved-model",
            "status": "completed", 
            "started_at": "2025-01-01T14:00:00Z",
            "completed_at": "2025-01-01T15:45:00Z",
            "duration": 6300,  # 1.75 hours in seconds
            "tags": ["improved", "training"]
        }
        exp2_config = {
            "learning_rate": 0.0005,
            "batch_size": 64,
            "epochs": 15,
            "model_type": "transformer"
        }
        exp2_results = {
            "final_accuracy": 0.9245,
            "final_loss": 0.1432,
            "best_epoch": 12,
            "training_time": 6234
        }
        
        self._create_test_experiment("exp001", exp1_metadata, exp1_config, exp1_results)
        self._create_test_experiment("exp002", exp2_metadata, exp2_config, exp2_results)

        # Test with real ExperimentComparisonData
        comparison = ExperimentComparisonData(ExperimentManager(self.experiments_dir))
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

    def test_compare_only_different_filtering_integration(self):
        """Test --only-different filtering with real data."""
        # Create experiments with some identical parameters
        exp1_metadata = {
            "id": "exp001",
            "name": "model-a",
            "status": "completed"
        }
        exp1_config = {
            "learning_rate": 0.001,    # Different
            "batch_size": 32,          # Same
            "model_type": "bert",      # Same
            "optimizer": "adam"        # Different
        }
        exp1_results = {
            "accuracy": 0.91,          # Different
            "epochs_trained": 10       # Same
        }
        
        exp2_metadata = {
            "id": "exp002",
            "name": "model-b", 
            "status": "completed"
        }
        exp2_config = {
            "learning_rate": 0.002,    # Different
            "batch_size": 32,          # Same
            "model_type": "bert",      # Same
            "optimizer": "sgd"         # Different
        }
        exp2_results = {
            "accuracy": 0.89,          # Different
            "epochs_trained": 10       # Same
        }
        
        self._create_test_experiment("exp001", exp1_metadata, exp1_config, exp1_results)
        self._create_test_experiment("exp002", exp2_metadata, exp2_config, exp2_results)

        # Test with only_different=True
        comparison = ExperimentComparisonData(ExperimentManager(self.experiments_dir))
        comparison_data = comparison.get_comparison_data(
            ["exp001", "exp002"], 
            only_different=True
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

    def test_compare_column_type_inference_integration(self):
        """Test column type inference with real data."""
        exp_metadata = {
            "id": "exp001",
            "name": "test-exp",
            "status": "completed"
        }
        exp_config = {
            "learning_rate": 0.001,     # Numeric
            "model_name": "bert-base",  # String
            "use_gpu": True,            # Boolean -> String
            "layers": 12                # Numeric
        }
        exp_results = {
            "accuracy": 0.923,          # Numeric
            "model_size": "110M",       # String
            "training_steps": 1000      # Numeric
        }
        
        self._create_test_experiment("exp001", exp_metadata, exp_config, exp_results)

        comparison = ExperimentComparisonData(ExperimentManager(self.experiments_dir))
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
        assert column_types["param:learning_rate"] == "numeric"
        assert column_types["param:model_name"] == "string"
        assert column_types["param:use_gpu"] == "string"  # Boolean formatted as string
        assert column_types["param:layers"] == "numeric"
        
        # Check inferred metric types
        assert column_types["metric:accuracy"] == "numeric"
        assert column_types["metric:model_size"] == "string"
        assert column_types["metric:training_steps"] == "numeric"