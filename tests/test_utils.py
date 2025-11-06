"""
Test utilities for reducing duplication in test setup and data creation.

This module provides helper functions and utilities to reduce code duplication
across test files while maintaining test isolation and safety.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from click.testing import CliRunner

from yanex.core.manager import ExperimentManager
from yanex.core.storage import ExperimentStorage


class TestDataFactory:
    """Factory for creating standardized test data."""

    @staticmethod
    def create_experiment_metadata(
        experiment_id: str,
        status: str = "completed",
        name: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """
        Create standard experiment metadata for testing.

        Args:
            experiment_id: Unique experiment identifier
            status: Experiment status (completed, failed, running, etc.)
            name: Optional experiment name
            **overrides: Additional fields to override defaults

        Returns:
            Dictionary containing experiment metadata
        """
        base_metadata = {
            "id": experiment_id,
            "status": status,
            "created_at": "2023-01-01T12:00:00Z",
            "started_at": "2023-01-01T12:00:01Z",
            "script_path": f"/path/to/{experiment_id}/script.py",
            "tags": [],
            "archived": False,
        }

        if name:
            base_metadata["name"] = name

        if status == "completed":
            base_metadata["completed_at"] = "2023-01-01T12:05:00Z"
            base_metadata["duration"] = 299.0
        elif status == "failed":
            base_metadata["failed_at"] = "2023-01-01T12:03:00Z"
            base_metadata["error"] = "Test error message"
        elif status == "running":
            # Remove end time fields for running experiments
            pass

        # Apply any overrides
        base_metadata.update(overrides)
        return base_metadata

    @staticmethod
    def create_experiment_config(
        config_type: str = "ml_training", **overrides: Any
    ) -> dict[str, Any]:
        """
        Create standard experiment configuration for testing.

        Args:
            config_type: Type of config template to use
            **overrides: Additional fields to override defaults

        Returns:
            Dictionary containing experiment configuration
        """
        config_templates = {
            "ml_training": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "model_type": "transformer",
                "optimizer": "adam",
            },
            "data_processing": {
                "n_docs": 1000,
                "chunk_size": 100,
                "format": "json",
                "parallel": True,
            },
            "simple": {
                "param1": "value1",
                "param2": 42,
                "param3": True,
            },
        }

        base_config = config_templates.get(config_type, config_templates["simple"])
        base_config.update(overrides)
        return base_config

    @staticmethod
    def create_experiment_results(
        result_type: str = "ml_metrics", **overrides: Any
    ) -> dict[str, Any]:
        """
        Create standard experiment results for testing.

        Args:
            result_type: Type of results template to use
            **overrides: Additional fields to override defaults

        Returns:
            Dictionary containing experiment results
        """
        result_templates = {
            "ml_metrics": {
                "accuracy": 0.95,
                "loss": 0.05,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "training_time": 299.5,
            },
            "processing_stats": {
                "docs_processed": 1000,
                "processing_time": 45.2,
                "errors": 0,
                "success_rate": 1.0,
            },
            "simple": {
                "value": 123.45,
                "status": "success",
                "timestamp": "2023-01-01T12:05:00Z",
            },
        }

        base_results = result_templates.get(result_type, result_templates["simple"])
        base_results.update(overrides)
        return base_results

    @staticmethod
    def create_result_entry(step: int, **metrics: Any) -> dict[str, Any]:
        """
        Create a standardized result entry for testing.

        Args:
            step: Training/processing step number
            **metrics: Additional metrics to include in the result

        Returns:
            Dictionary containing result entry with step and metrics
        """
        result = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        result.update(metrics)
        return result

    @staticmethod
    def create_comparison_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Create comparison data structure for testing.

        Args:
            rows: List of row dictionaries containing experiment data

        Returns:
            Dictionary containing comparison data structure
        """
        # Extract parameter and metric columns from rows
        param_columns = set()
        metric_columns = set()

        for row in rows:
            for key in row.keys():
                if key.startswith("param:"):
                    param_columns.add(key.replace("param:", ""))
                elif key.startswith("metric:"):
                    metric_columns.add(key.replace("metric:", ""))

        return {
            "rows": rows,
            "param_columns": sorted(param_columns),
            "metric_columns": sorted(metric_columns),
            "column_types": {},
            "total_experiments": len(rows),
        }


class TestAssertions:
    """Custom assertion helpers for domain-specific validation."""

    @staticmethod
    def assert_valid_experiment_metadata(metadata: dict[str, Any]) -> None:
        """
        Assert that experiment metadata has required structure.

        Args:
            metadata: Experiment metadata dictionary to validate
        """
        required_fields = ["id", "status", "created_at"]
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        # Validate status is one of known values
        valid_statuses = [
            "created",
            "running",
            "completed",
            "failed",
            "cancelled",
            "staged",
        ]
        assert metadata["status"] in valid_statuses, (
            f"Invalid status: {metadata['status']}"
        )

        # Validate ID format (should be non-empty string)
        assert isinstance(metadata["id"], str) and len(metadata["id"]) > 0, (
            "ID must be non-empty string"
        )

    @staticmethod
    def assert_valid_experiment_config(config: dict[str, Any]) -> None:
        """
        Assert that experiment configuration is valid.

        Args:
            config: Experiment configuration dictionary to validate
        """
        assert isinstance(config, dict), "Config must be a dictionary"
        assert len(config) > 0, "Config cannot be empty"

    @staticmethod
    def assert_valid_experiment_results(results: dict[str, Any]) -> None:
        """
        Assert that experiment results are valid.

        Args:
            results: Experiment results dictionary to validate
        """
        assert isinstance(results, dict), "Results must be a dictionary"
        # Results can be empty, so no length check

    @staticmethod
    def assert_experiment_directory_structure(experiment_dir: Path) -> None:
        """
        Assert that experiment directory has the expected structure.

        Args:
            experiment_dir: Path to experiment directory
        """
        assert experiment_dir.exists(), (
            f"Experiment directory does not exist: {experiment_dir}"
        )
        assert experiment_dir.is_dir(), (
            f"Experiment path is not a directory: {experiment_dir}"
        )

        # Check for artifacts subdirectory (created by default)
        artifacts_dir = experiment_dir / "artifacts"
        assert artifacts_dir.exists(), f"Artifacts directory missing: {artifacts_dir}"
        assert artifacts_dir.is_dir(), (
            f"Artifacts path is not a directory: {artifacts_dir}"
        )

    @staticmethod
    def assert_experiment_files_exist(
        experiment_dir: Path,
        check_metadata: bool = True,
        check_config: bool = False,
        check_results: bool = False,
    ) -> None:
        """
        Assert that specific experiment files exist in directory.

        Args:
            experiment_dir: Path to experiment directory
            check_metadata: Whether to check for metadata.json
            check_config: Whether to check for config.json
            check_results: Whether to check for metrics.json
        """
        TestAssertions.assert_experiment_directory_structure(experiment_dir)

        if check_metadata:
            metadata_file = experiment_dir / "metadata.json"
            assert metadata_file.exists(), f"Metadata file missing: {metadata_file}"

        if check_config:
            config_file = experiment_dir / "config.json"
            assert config_file.exists(), f"Config file missing: {config_file}"

        if check_results:
            results_file = experiment_dir / "metrics.json"
            assert results_file.exists(), f"Metrics file missing: {results_file}"

    @staticmethod
    def assert_metadata_fields(
        metadata: dict[str, Any], required_fields: list[str]
    ) -> None:
        """
        Assert that metadata contains all required fields.

        Args:
            metadata: Metadata dictionary to check
            required_fields: List of field names that must be present
        """
        for field in required_fields:
            assert field in metadata, f"Required metadata field missing: {field}"


class TestFileHelpers:
    """Helpers for creating test files and scripts."""

    @staticmethod
    def create_test_script(
        temp_dir: Path,
        script_name: str = "test_script.py",
        script_type: str = "simple",
        **template_vars: Any,
    ) -> Path:
        """
        Create a test Python script with specified content.

        Args:
            temp_dir: Directory to create script in
            script_name: Name of the script file
            script_type: Type of script template to use
            **template_vars: Variables to substitute in template

        Returns:
            Path to created script file
        """
        script_templates = {
            "simple": """
import sys
print("Hello from test script")
sys.exit(0)
""",
            "yanex_basic": """
import yanex

# Get parameters
params = yanex.get_params()
print(f"Running with params: {{params}}")

# Log some results
results = {{"test_metric": 42, "status": "success"}}
yanex.log_metrics(results)
""",
            "yanex_ml": """
import yanex

# Get parameters
params = yanex.get_params()
learning_rate = params.get("learning_rate", 0.001)
epochs = params.get("epochs", 10)

# Simulate training
for epoch in range(epochs):
    accuracy = 0.7 + (epoch * 0.03)  # Fake improvement
    loss = 1.0 - accuracy
    yanex.log_metrics({{
        "epoch": epoch,
        "accuracy": accuracy,
        "loss": loss,
        "learning_rate": learning_rate
    }})

print("Training completed")
""",
            "failing": """
import sys
print("This script will fail")
raise ValueError("Test failure")
""",
        }

        template = script_templates.get(script_type, script_templates["simple"])

        # Simple template variable substitution
        for var, value in template_vars.items():
            template = template.replace(f"{{{var}}}", str(value))

        script_path = temp_dir / script_name
        script_path.write_text(template)
        return script_path

    @staticmethod
    def create_config_file(
        temp_dir: Path, config_data: dict[str, Any], filename: str = "config.yaml"
    ) -> Path:
        """
        Create a configuration file with specified data.

        Args:
            temp_dir: Directory to create config in
            config_data: Configuration data to write
            filename: Name of config file

        Returns:
            Path to created config file
        """
        config_path = temp_dir / filename

        if filename.endswith(".json"):
            config_path.write_text(json.dumps(config_data, indent=2))
        elif filename.endswith(".yaml") or filename.endswith(".yml"):
            import yaml

            config_path.write_text(yaml.dump(config_data, default_flow_style=False))
        else:
            # Fallback to JSON
            config_path.write_text(json.dumps(config_data, indent=2))

        return config_path

    @staticmethod
    def create_test_file(file_path: Path, content: str) -> Path:
        """
        Create a test file with specified content.

        Args:
            file_path: Path where the file should be created
            content: Content to write to the file

        Returns:
            Path to the created file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    @staticmethod
    def create_multiple_experiment_directories(
        storage: ExperimentStorage, experiment_ids: list[str]
    ) -> None:
        """
        Create multiple experiment directories with metadata using storage.

        Args:
            storage: ExperimentStorage instance to use
            experiment_ids: List of experiment IDs to create directories for
        """
        for exp_id in experiment_ids:
            storage.create_experiment_directory(exp_id)
            # Create minimal metadata so the experiment is detected by list_experiments()
            metadata = TestDataFactory.create_experiment_metadata(
                experiment_id=exp_id, status="completed"
            )
            storage.save_metadata(exp_id, metadata)

    @staticmethod
    def create_experiment_files(
        experiment_dir: Path,
        metadata: dict[str, Any],
        config: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
    ) -> None:
        """
        Create complete experiment files in a directory.

        Args:
            experiment_dir: Directory to create experiment files in
            metadata: Experiment metadata to save
            config: Optional configuration data to save
            results: Optional results data to save
        """
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata.json
        with (experiment_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        # Save config.yaml if provided
        if config is not None:
            try:
                import yaml

                with (experiment_dir / "config.yaml").open("w") as f:
                    yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                # Fallback to JSON
                with (experiment_dir / "config.json").open("w") as f:
                    json.dump(config, f, indent=2)

        # Save metrics.json if provided
        if results is not None:
            with (experiment_dir / "metrics.json").open("w") as f:
                json.dump(results, f, indent=2)

    @staticmethod
    def create_temp_storage():
        """
        Create a temporary ExperimentStorage instance for testing.

        Returns:
            ExperimentStorage instance using a temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp())
        return ExperimentStorage(temp_dir)


class MockHelpers:
    """Helpers for creating standardized mocks."""

    @staticmethod
    def create_git_validation_mocks():
        """
        Create standard mocks for git validation.

        Returns:
            Dictionary of common git-related mocks
        """
        return {
            "get_git_commit_hash": Mock(return_value="abc123def456"),
            "get_git_branch": Mock(return_value="main"),
            "is_git_repo": Mock(return_value=True),
        }

    @staticmethod
    def create_environment_mocks():
        """
        Create standard mocks for environment capture.

        Returns:
            Dictionary of common environment mocks
        """
        return {
            "capture_environment": Mock(
                return_value={
                    "python_version": "3.11.0",
                    "platform": "linux",
                    "hostname": "test-host",
                    "user": "test-user",
                }
            ),
            "get_system_info": Mock(
                return_value={
                    "cpu_count": 4,
                    "memory_gb": 16,
                    "disk_free_gb": 100,
                }
            ),
        }


# Additional fixtures that supplement the existing conftest.py
# These are kept separate to avoid conflicts


def create_isolated_storage(temp_dir: Path) -> ExperimentStorage:
    """
    Create an isolated ExperimentStorage instance for testing.

    Args:
        temp_dir: Temporary directory for storage

    Returns:
        ExperimentStorage instance
    """
    return ExperimentStorage(temp_dir)


def create_isolated_manager(temp_dir: Path | None = None) -> ExperimentManager:
    """
    Create an isolated ExperimentManager instance for testing.

    Args:
        temp_dir: Temporary directory for experiments (creates one if None)

    Returns:
        ExperimentManager instance
    """
    if temp_dir is None:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
    return ExperimentManager(experiments_dir=temp_dir)


def create_cli_runner() -> CliRunner:
    """
    Create a Click CLI runner for testing.

    Returns:
        CliRunner instance
    """
    return CliRunner()


# Constants for common test data
TEST_EXPERIMENT_IDS = [
    "test001",
    "test002",
    "test003",
    "exp001",
    "exp002",
    "sample01",
    "sample02",
]

TEST_EXPERIMENT_NAMES = [
    "Test Experiment 1",
    "Training Run Alpha",
    "Data Processing Job",
    "Model Evaluation",
    "Hyperparameter Sweep",
]

TEST_TAGS = [
    "test",
    "production",
    "experimental",
    "ml-training",
    "data-processing",
    "baseline",
    "comparison",
]
