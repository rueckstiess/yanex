"""
Demonstration of test utilities - showing how they can reduce duplication.

This file demonstrates the test utilities without modifying existing tests,
showing the before/after patterns for common test scenarios.
"""

import json

from tests.test_utils import (
    TestAssertions,
    TestDataFactory,
    TestFileHelpers,
)
from yanex.core.storage import ExperimentStorage


class TestUtilityDemonstration:
    """Demonstrate test utility usage patterns."""

    def test_old_style_metadata_creation(self, temp_dir):
        """Example of old-style manual metadata creation (before utilities)."""
        # Old way - manual metadata creation (common pattern in existing tests)
        experiment_id = "test123"
        metadata = {
            "id": experiment_id,
            "status": "completed",
            "created_at": "2023-01-01T12:00:00Z",
            "started_at": "2023-01-01T12:00:01Z",
            "completed_at": "2023-01-01T12:05:00Z",
            "script_path": f"/path/to/{experiment_id}/script.py",
            "tags": [],
            "archived": False,
            "duration": 299.0,
        }

        # Manual validation
        assert "id" in metadata
        assert "status" in metadata
        assert "created_at" in metadata
        assert metadata["status"] in ["created", "running", "completed", "failed"]

        # This pattern is repeated in many test files
        assert metadata["id"] == experiment_id
        assert metadata["status"] == "completed"

    def test_new_style_metadata_creation(self, temp_dir):
        """Example of new-style utility-based metadata creation."""
        # New way - using TestDataFactory
        experiment_id = "test123"
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=experiment_id, status="completed", name="Demo Experiment"
        )

        # Validation using utility
        TestAssertions.assert_valid_experiment_metadata(metadata)

        # More readable assertions
        assert metadata["id"] == experiment_id
        assert metadata["status"] == "completed"
        assert metadata["name"] == "Demo Experiment"

    def test_old_style_storage_setup(self, temp_dir):
        """Example of old-style manual storage setup (before utilities)."""
        # Old way - manual storage setup (common in existing tests)
        experiments_dir = temp_dir / "experiments"
        experiments_dir.mkdir()
        storage = ExperimentStorage(experiments_dir)

        # Manual experiment creation and validation
        experiment_id = "exp001"
        exp_dir = storage.create_experiment_directory(experiment_id)

        # Manual assertions
        assert exp_dir.exists()
        assert exp_dir.is_dir()
        assert (exp_dir / "artifacts").exists()

    def test_new_style_storage_setup(self, isolated_storage):
        """Example of new-style fixture-based storage setup."""
        # New way - using isolated_storage fixture
        experiment_id = "exp001"
        exp_dir = isolated_storage.create_experiment_directory(experiment_id)

        # Using utility assertions
        TestAssertions.assert_experiment_directory_structure(exp_dir)

    def test_old_style_config_and_script_creation(self, temp_dir):
        """Example of old-style manual file creation (before utilities)."""
        # Old way - manual config creation
        config_data = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_type": "transformer",
        }
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps(config_data, indent=2))

        # Old way - manual script creation
        script_path = temp_dir / "test_script.py"
        script_content = """
import yanex
params = yanex.get_params()
results = {"accuracy": 0.95, "loss": 0.05}
yanex.log_metrics(results)
"""
        script_path.write_text(script_content)

        # Manual validation
        assert config_path.exists()
        assert script_path.exists()

    def test_new_style_config_and_script_creation(self, temp_dir):
        """Example of new-style utility-based file creation."""
        # New way - using TestDataFactory and TestFileHelpers
        config_data = TestDataFactory.create_experiment_config("ml_training")
        config_path = TestFileHelpers.create_config_file(temp_dir, config_data)

        script_path = TestFileHelpers.create_test_script(
            temp_dir, script_type="yanex_basic"
        )

        # Utility validation
        TestAssertions.assert_valid_experiment_config(config_data)
        assert config_path.exists()
        assert script_path.exists()

    def test_bulk_experiment_creation_old_style(self, temp_dir):
        """Example of creating multiple test experiments the old way."""
        # Old way - lots of repetitive code

        experiments = []
        for i in range(3):
            exp_id = f"exp{i:03d}"
            metadata = {
                "id": exp_id,
                "status": "completed" if i < 2 else "failed",
                "created_at": f"2023-01-{i + 1:02d}T12:00:00Z",
                "started_at": f"2023-01-{i + 1:02d}T12:00:01Z",
                "script_path": f"/path/to/{exp_id}/script.py",
                "tags": ["test"],
                "archived": False,
            }
            if metadata["status"] == "completed":
                metadata["completed_at"] = f"2023-01-{i + 1:02d}T12:05:00Z"
                metadata["duration"] = 299.0
            else:
                metadata["failed_at"] = f"2023-01-{i + 1:02d}T12:03:00Z"
                metadata["error"] = "Test error"

            experiments.append(metadata)

            # Manual validation for each
            assert "id" in metadata
            assert "status" in metadata

        assert len(experiments) == 3

    def test_bulk_experiment_creation_new_style(self, temp_dir):
        """Example of creating multiple test experiments with utilities."""
        # New way - using utilities for consistent data creation

        experiments = []
        for i in range(3):
            exp_id = f"exp{i:03d}"
            status = "completed" if i < 2 else "failed"

            metadata = TestDataFactory.create_experiment_metadata(
                experiment_id=exp_id,
                status=status,
                created_at=f"2023-01-{i + 1:02d}T12:00:00Z",
                tags=["test"],
            )

            experiments.append(metadata)

            # Utility validation
            TestAssertions.assert_valid_experiment_metadata(metadata)

        assert len(experiments) == 3

    def test_complex_test_scenario_old_style(self, temp_dir):
        """Example of complex test scenario the old way."""
        # Old way - lots of setup code
        experiments_dir = temp_dir / "experiments"
        experiments_dir.mkdir()
        storage = ExperimentStorage(experiments_dir)

        # Create experiment
        exp_id = "complex001"
        exp_dir = storage.create_experiment_directory(exp_id)

        # Create metadata
        metadata = {
            "id": exp_id,
            "status": "completed",
            "name": "Complex Test",
            "created_at": "2023-01-01T12:00:00Z",
            "started_at": "2023-01-01T12:00:01Z",
            "completed_at": "2023-01-01T12:05:00Z",
            "script_path": str(exp_dir / "script.py"),
            "tags": ["ml", "test"],
            "archived": False,
            "duration": 299.0,
        }

        # Create config
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }

        # Create results
        results = {
            "accuracy": 0.95,
            "loss": 0.05,
            "training_time": 299.5,
        }

        # Save files manually
        metadata_file = exp_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        config_file = exp_dir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        results_file = exp_dir / "metrics.json"
        results_file.write_text(json.dumps(results, indent=2))

        # Manual validations
        assert exp_dir.exists()
        assert metadata_file.exists()
        assert config_file.exists()
        assert results_file.exists()
        assert metadata["status"] == "completed"

    def test_complex_test_scenario_new_style(self, isolated_storage):
        """Example of complex test scenario with utilities."""
        # New way - using utilities for cleaner setup
        exp_id = "complex001"
        exp_dir = isolated_storage.create_experiment_directory(exp_id)

        # Create data using factories
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id,
            status="completed",
            name="Complex Test",
            script_path=str(exp_dir / "script.py"),
            tags=["ml", "test"],
        )

        config = TestDataFactory.create_experiment_config("ml_training")
        results = TestDataFactory.create_experiment_results("ml_metrics")

        # Save files (this could be further abstracted if needed)
        metadata_file = exp_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        config_file = exp_dir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        results_file = exp_dir / "metrics.json"
        results_file.write_text(json.dumps(results, indent=2))

        # Utility validations
        TestAssertions.assert_experiment_files_exist(
            exp_dir, check_metadata=True, check_config=True, check_results=True
        )
        TestAssertions.assert_valid_experiment_metadata(metadata)
        TestAssertions.assert_valid_experiment_config(config)
        TestAssertions.assert_valid_experiment_results(results)


class TestFixtureComparison:
    """Compare old vs new fixture usage patterns."""

    def test_old_storage_pattern(self, temp_dir):
        """Old pattern: manual storage creation in each test."""
        # This pattern appears in many existing tests
        experiments_dir = temp_dir / "experiments"
        experiments_dir.mkdir()
        storage = ExperimentStorage(experiments_dir)

        # Test logic using storage
        exp_id = "test001"
        exp_dir = storage.create_experiment_directory(exp_id)
        assert exp_dir.exists()

    def test_new_storage_pattern(self, isolated_storage):
        """New pattern: using isolated_storage fixture."""
        # Much cleaner - no setup boilerplate
        exp_id = "test001"
        exp_dir = isolated_storage.create_experiment_directory(exp_id)
        assert exp_dir.exists()

    def test_old_cli_pattern(self, temp_dir):
        """Old pattern: manual CLI runner setup."""
        from click.testing import CliRunner

        # This setup is repeated across CLI tests
        runner = CliRunner()

        # Test CLI command
        from yanex.cli.main import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_new_cli_pattern(self, cli_runner):
        """New pattern: using cli_runner fixture."""
        # Cleaner - no setup needed
        from yanex.cli.main import cli

        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0


# This demonstrates that the utilities can coexist with existing patterns
# and provide benefits without requiring massive changes to existing tests
