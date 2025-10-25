"""
Tests for --clone-from functionality in yanex run command.
"""

import os

from tests.test_utils import TestFileHelpers, create_cli_runner
from yanex.cli.main import cli
from yanex.core.manager import ExperimentManager


class TestCloneFrom:
    """Test --clone-from functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_clone_from_basic(self, tmp_path):
        """Test basic cloning of experiment parameters."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment with parameters
            config_data = {"learning_rate": 0.01, "epochs": 100, "batch_size": 32}
            config_path = TestFileHelpers.create_config_file(
                tmp_path, config_data, "config.yaml"
            )

            # Run original experiment
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID from the output or by listing experiments
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            assert len(experiments) == 1
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Clone the experiment (dry run to check parameters)
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "learning_rate" in result.output
            assert "0.01" in result.output
            assert "epochs" in result.output
            assert "100" in result.output
            assert "batch_size" in result.output
            assert "32" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_with_param_override(self, tmp_path):
        """Test cloning with CLI parameter override."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment with parameters
            config_data = {"learning_rate": 0.01, "epochs": 100, "batch_size": 32}
            config_path = TestFileHelpers.create_config_file(
                tmp_path, config_data, "config.yaml"
            )

            # Run original experiment
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Clone with parameter override (dry run to check)
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--param",
                    "learning_rate=0.05",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

            # Should have the overridden value
            assert "learning_rate" in result.output
            assert "0.05" in result.output  # Override value

            # Should still have other cloned values
            assert "epochs" in result.output
            assert "100" in result.output
            assert "batch_size" in result.output
            assert "32" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_with_config_override(self, tmp_path):
        """Test cloning with config file override."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment with parameters
            original_config = {"learning_rate": 0.01, "epochs": 100, "batch_size": 32}
            original_config_path = TestFileHelpers.create_config_file(
                tmp_path, original_config, "original.yaml"
            )

            # Run original experiment
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(original_config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Create new config file with different values
            new_config = {"learning_rate": 0.02, "epochs": 200}
            new_config_path = TestFileHelpers.create_config_file(
                tmp_path, new_config, "new.yaml"
            )

            # Clone with config file override (dry run to check)
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--config",
                    str(new_config_path),
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

            # Should have config file values (override cloned values)
            assert "learning_rate" in result.output
            assert "0.02" in result.output  # Config file value

            assert "epochs" in result.output
            assert "200" in result.output  # Config file value

            # Should still have batch_size from cloned experiment (not in new config)
            assert "batch_size" in result.output
            assert "32" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_with_both_overrides(self, tmp_path):
        """Test cloning with both config file and CLI parameter overrides."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment with parameters
            original_config = {"learning_rate": 0.01, "epochs": 100, "batch_size": 32}
            original_config_path = TestFileHelpers.create_config_file(
                tmp_path, original_config, "original.yaml"
            )

            # Run original experiment
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(original_config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Create new config file
            new_config = {"learning_rate": 0.02, "epochs": 200}
            new_config_path = TestFileHelpers.create_config_file(
                tmp_path, new_config, "new.yaml"
            )

            # Clone with both config file and CLI param override (dry run)
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--config",
                    str(new_config_path),
                    "--param",
                    "learning_rate=0.05",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

            # CLI param should override everything
            assert "learning_rate" in result.output
            assert "0.05" in result.output  # CLI param value (highest precedence)

            # Config file value
            assert "epochs" in result.output
            assert "200" in result.output

            # Cloned value (not in config or CLI)
            assert "batch_size" in result.output
            assert "32" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_shortened_id(self, tmp_path):
        """Test cloning with shortened experiment ID."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment with parameters
            config_data = {"learning_rate": 0.01, "epochs": 100}
            config_path = TestFileHelpers.create_config_file(
                tmp_path, config_data, "config.yaml"
            )

            # Run original experiment
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Use shortened ID (first 4 characters)
            short_id = original_id[:4]

            # Clone using shortened ID
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    short_id,
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "learning_rate" in result.output
            assert "0.01" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_nonexistent_experiment(self, tmp_path):
        """Test cloning from non-existent experiment ID."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Try to clone from non-existent ID
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    "nonexist",
                    "--dry-run",
                ],
            )
            assert result.exit_code != 0
            assert "No experiment found" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_saves_correct_config(self, tmp_path):
        """Test that cloned experiment saves the correct merged config.yaml."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment with parameters
            original_config = {"learning_rate": 0.01, "epochs": 100, "batch_size": 32}
            original_config_path = TestFileHelpers.create_config_file(
                tmp_path, original_config, "original.yaml"
            )

            # Run original experiment
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(original_config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Clone with parameter override
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--param",
                    "learning_rate=0.05",
                    "--name",
                    "cloned-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Find the cloned experiment
            experiments = manager.list_experiments()
            cloned_id = None
            for exp_id in experiments:
                metadata = manager.get_experiment_metadata(exp_id)
                if metadata.get("name") == "cloned-experiment":
                    cloned_id = exp_id
                    break

            assert cloned_id is not None

            # Load the config from the cloned experiment
            cloned_config = manager.storage.load_config(cloned_id)

            # Verify the config has correct merged values
            assert cloned_config["learning_rate"] == 0.05  # Overridden
            assert cloned_config["epochs"] == 100  # Cloned
            assert cloned_config["batch_size"] == 32  # Cloned

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_nested_parameters(self, tmp_path):
        """Test cloning with nested parameters."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment with nested parameters
            original_config = {
                "model": {"layers": 3, "units": 128, "activation": "relu"},
                "optimizer": {"type": "adam", "learning_rate": 0.001},
            }
            original_config_path = TestFileHelpers.create_config_file(
                tmp_path, original_config, "original.yaml"
            )

            # Run original experiment
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(original_config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Clone with nested parameter override
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--param",
                    "optimizer.learning_rate=0.01",
                    "--name",
                    "cloned-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Find the cloned experiment
            experiments = manager.list_experiments()
            cloned_id = None
            for exp_id in experiments:
                metadata = manager.get_experiment_metadata(exp_id)
                if metadata.get("name") == "cloned-experiment":
                    cloned_id = exp_id
                    break

            assert cloned_id is not None

            # Load the config from the cloned experiment
            cloned_config = manager.storage.load_config(cloned_id)

            # Verify nested structure is preserved and override worked
            assert cloned_config["model"]["layers"] == 3  # Cloned
            assert cloned_config["model"]["units"] == 128  # Cloned
            assert cloned_config["model"]["activation"] == "relu"  # Cloned
            assert cloned_config["optimizer"]["type"] == "adam"  # Cloned
            assert cloned_config["optimizer"]["learning_rate"] == 0.01  # Overridden

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_empty_config(self, tmp_path):
        """Test cloning from experiment with empty config."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Run original experiment with no config
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Clone and add new parameters
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--param",
                    "learning_rate=0.01",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert "learning_rate" in result.output
            assert "0.01" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

    def test_clone_from_verbose_output(self, tmp_path):
        """Test verbose output shows cloning information."""
        # Set up isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create a test script
            script_path = TestFileHelpers.create_test_script(
                tmp_path, "test_script.py", "simple"
            )

            # Create original experiment
            config_data = {"learning_rate": 0.01}
            config_path = TestFileHelpers.create_config_file(
                tmp_path, config_data, "config.yaml"
            )

            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--name",
                    "original-experiment",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Get the experiment ID
            manager = ExperimentManager()
            experiments = manager.list_experiments()
            original_id = experiments[0]  # list_experiments returns list of IDs

            # Clone with verbose flag
            result = self.runner.invoke(
                cli,
                [
                    "--verbose",
                    "run",
                    str(script_path),
                    "--clone-from",
                    original_id,
                    "--dry-run",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert (
                "Cloning from experiment" in result.output
                or "Cloned config from experiment" in result.output
            )
            assert original_id in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            else:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
