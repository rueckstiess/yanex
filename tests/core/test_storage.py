"""
Tests for yanex.core.storage module.
"""

import shutil
from unittest.mock import patch

import pytest

from yanex.core.storage import ExperimentStorage
from yanex.utils.exceptions import StorageError


class TestExperimentStorageInit:
    """Test ExperimentStorage initialization."""

    def test_init_with_default_path(self, temp_dir):
        """Test initialization with default experiments directory."""
        with patch("yanex.core.storage.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir
            storage = ExperimentStorage()

        expected_path = temp_dir / "experiments"
        assert storage.experiments_dir == expected_path
        assert expected_path.exists()

    def test_init_with_custom_path(self, temp_dir):
        """Test initialization with custom experiments directory."""
        custom_path = temp_dir / "custom_experiments"
        storage = ExperimentStorage(custom_path)

        assert storage.experiments_dir == custom_path
        assert custom_path.exists()


class TestCreateExperimentDirectory:
    """Test experiment directory creation."""

    def test_create_new_experiment_directory(self, temp_dir):
        """Test creating new experiment directory."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        exp_dir = storage.create_experiment_directory(experiment_id)

        expected_path = temp_dir / experiment_id
        assert exp_dir == expected_path
        assert exp_dir.exists()
        assert exp_dir.is_dir()
        assert (exp_dir / "artifacts").exists()
        assert (exp_dir / "artifacts").is_dir()

    def test_create_duplicate_experiment_directory(self, temp_dir):
        """Test creating directory that already exists raises StorageError."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        # Create directory first time
        storage.create_experiment_directory(experiment_id)

        # Try to create again
        with pytest.raises(StorageError, match="already exists"):
            storage.create_experiment_directory(experiment_id)


class TestGetExperimentDirectory:
    """Test getting experiment directory."""

    def test_get_existing_experiment_directory(self, temp_dir):
        """Test getting existing experiment directory."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        # Create directory
        created_dir = storage.create_experiment_directory(experiment_id)

        # Get directory
        retrieved_dir = storage.get_experiment_directory(experiment_id)

        assert retrieved_dir == created_dir

    def test_get_nonexistent_experiment_directory(self, temp_dir):
        """Test getting nonexistent directory raises StorageError."""
        storage = ExperimentStorage(temp_dir)

        with pytest.raises(StorageError, match="not found"):
            storage.get_experiment_directory("nonexistent")


class TestSaveLoadMetadata:
    """Test metadata save/load operations."""

    def test_save_and_load_metadata(self, temp_dir):
        """Test saving and loading metadata."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        # Create experiment directory
        storage.create_experiment_directory(experiment_id)

        # Test metadata
        metadata = {
            "id": experiment_id,
            "name": "test_experiment",
            "status": "running",
            "git_commit": "def45678",
        }

        # Save metadata
        storage.save_metadata(experiment_id, metadata)

        # Load metadata
        loaded_metadata = storage.load_metadata(experiment_id)

        # Check that all original data is present
        for key, value in metadata.items():
            assert loaded_metadata[key] == value

        # Check that timestamp was added
        assert "saved_at" in loaded_metadata
        assert isinstance(loaded_metadata["saved_at"], str)

    def test_load_nonexistent_metadata(self, temp_dir):
        """Test loading nonexistent metadata raises StorageError."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        with pytest.raises(StorageError, match="Metadata file not found"):
            storage.load_metadata(experiment_id)


class TestSaveLoadConfig:
    """Test configuration save/load operations."""

    def test_save_and_load_config(self, temp_dir):
        """Test saving and loading configuration."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        config = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "model": {"type": "transformer", "layers": 6},
        }

        # Save config
        storage.save_config(experiment_id, config)

        # Load config
        loaded_config = storage.load_config(experiment_id)

        assert loaded_config == config

    def test_load_nonexistent_config(self, temp_dir):
        """Test loading nonexistent config returns empty dict."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        config = storage.load_config(experiment_id)
        assert config == {}


class TestSaveLoadResults:
    """Test results save/load operations."""

    def test_save_and_load_results(self, temp_dir):
        """Test saving and loading results."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        results = [
            {"step": 0, "accuracy": 0.85, "loss": 0.3},
            {"step": 1, "accuracy": 0.87, "loss": 0.25},
        ]

        # Save results
        storage.save_results(experiment_id, results)

        # Load results
        loaded_results = storage.load_results(experiment_id)

        assert loaded_results == results

    def test_load_nonexistent_results(self, temp_dir):
        """Test loading nonexistent results returns empty list."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        results = storage.load_results(experiment_id)
        assert results == []


class TestAddResultStep:
    """Test adding result steps."""

    def test_add_result_step_auto_increment(self, temp_dir):
        """Test adding result step with auto-increment."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        # Add first result
        step1 = storage.add_result_step(experiment_id, {"accuracy": 0.85})
        assert step1 == 0

        # Add second result
        step2 = storage.add_result_step(experiment_id, {"accuracy": 0.87})
        assert step2 == 1

        # Check results
        results = storage.load_results(experiment_id)
        assert len(results) == 2
        assert results[0]["step"] == 0
        assert results[1]["step"] == 1
        assert "timestamp" in results[0]
        assert "timestamp" in results[1]

    def test_add_result_step_explicit_step(self, temp_dir):
        """Test adding result step with explicit step number."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        # Add result with explicit step
        step = storage.add_result_step(experiment_id, {"accuracy": 0.90}, step=5)
        assert step == 5

        # Add another with auto-increment (should be 6)
        next_step = storage.add_result_step(experiment_id, {"accuracy": 0.91})
        assert next_step == 6

    def test_add_result_step_replace_existing(self, temp_dir):
        """Test replacing existing result step."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        # Add initial result
        storage.add_result_step(experiment_id, {"accuracy": 0.85}, step=0)

        # Replace with new result
        storage.add_result_step(experiment_id, {"accuracy": 0.90}, step=0)

        # Check that result was replaced
        results = storage.load_results(experiment_id)
        assert len(results) == 1
        assert results[0]["step"] == 0
        assert results[0]["accuracy"] == 0.90

    def test_add_result_step_sorting(self, temp_dir):
        """Test that results are sorted by step."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        # Add results out of order
        storage.add_result_step(experiment_id, {"accuracy": 0.90}, step=2)
        storage.add_result_step(experiment_id, {"accuracy": 0.85}, step=0)
        storage.add_result_step(experiment_id, {"accuracy": 0.87}, step=1)

        # Check that results are sorted
        results = storage.load_results(experiment_id)
        assert [r["step"] for r in results] == [0, 1, 2]
        assert [r["accuracy"] for r in results] == [0.85, 0.87, 0.90]


class TestSaveArtifact:
    """Test artifact saving."""

    def test_save_file_artifact(self, temp_dir):
        """Test saving file artifact."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        # Create source file
        source_file = temp_dir / "source.txt"
        source_file.write_text("test content")

        # Save artifact
        artifact_path = storage.save_artifact(experiment_id, "test.txt", source_file)

        # Check artifact was saved
        expected_path = (
            storage.get_experiment_directory(experiment_id) / "artifacts" / "test.txt"
        )
        assert artifact_path == expected_path
        assert artifact_path.exists()
        assert artifact_path.read_text() == "test content"

    def test_save_artifact_nonexistent_source(self, temp_dir):
        """Test saving nonexistent source file raises StorageError."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        source_file = temp_dir / "nonexistent.txt"

        with pytest.raises(StorageError, match="Source path is not a file"):
            storage.save_artifact(experiment_id, "test.txt", source_file)


class TestSaveTextArtifact:
    """Test text artifact saving."""

    def test_save_text_artifact(self, temp_dir):
        """Test saving text artifact."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        content = "This is test content\nwith multiple lines"

        # Save text artifact
        artifact_path = storage.save_text_artifact(experiment_id, "notes.txt", content)

        # Check artifact was saved
        expected_path = (
            storage.get_experiment_directory(experiment_id) / "artifacts" / "notes.txt"
        )
        assert artifact_path == expected_path
        assert artifact_path.exists()
        assert artifact_path.read_text() == content


class TestGetLogPaths:
    """Test getting log file paths."""

    def test_get_log_paths(self, temp_dir):
        """Test getting log file paths."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        log_paths = storage.get_log_paths(experiment_id)

        exp_dir = storage.get_experiment_directory(experiment_id)
        assert log_paths["stdout"] == exp_dir / "stdout.log"
        assert log_paths["stderr"] == exp_dir / "stderr.log"


class TestListExperiments:
    """Test listing experiments."""

    def test_list_experiments(self, temp_dir):
        """Test listing experiments."""
        storage = ExperimentStorage(temp_dir)

        # Create several experiments
        experiment_ids = ["abc12345", "def67890", "ghi13579"]
        for exp_id in experiment_ids:
            storage.create_experiment_directory(exp_id)

        # Create non-experiment directory (should be ignored)
        (temp_dir / "not_experiment").mkdir()
        (temp_dir / "too_long_name").mkdir()

        listed_experiments = storage.list_experiments()

        assert sorted(listed_experiments) == sorted(experiment_ids)

    def test_list_experiments_empty(self, temp_dir):
        """Test listing experiments when directory is empty."""
        storage = ExperimentStorage(temp_dir)

        experiments = storage.list_experiments()
        assert experiments == []

    def test_list_experiments_no_directory(self, temp_dir):
        """Test listing experiments when experiments directory doesn't exist."""
        experiments_dir = temp_dir / "nonexistent"
        storage = ExperimentStorage(experiments_dir)

        # Remove the directory that was auto-created
        shutil.rmtree(experiments_dir)

        experiments = storage.list_experiments()
        assert experiments == []


class TestExperimentExists:
    """Test checking experiment existence."""

    def test_experiment_exists_true(self, temp_dir):
        """Test experiment exists returns True for existing experiment."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        storage.create_experiment_directory(experiment_id)

        assert storage.experiment_exists(experiment_id) is True

    def test_experiment_exists_false(self, temp_dir):
        """Test experiment exists returns False for nonexistent experiment."""
        storage = ExperimentStorage(temp_dir)

        assert storage.experiment_exists("nonexistent") is False


class TestArchiveExperiment:
    """Test experiment archiving."""

    def test_archive_experiment_default_location(self, temp_dir):
        """Test archiving experiment to default location."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"

        # Create experiment with some data
        storage.create_experiment_directory(experiment_id)
        storage.save_metadata(experiment_id, {"test": "data"})

        # Archive experiment
        archive_path = storage.archive_experiment(experiment_id)

        # Check experiment was moved
        expected_archive_path = temp_dir / "archived" / experiment_id
        assert archive_path == expected_archive_path
        assert archive_path.exists()
        assert not storage.experiment_exists(experiment_id)

        # Check archived data is intact
        metadata_path = archive_path / "metadata.json"
        assert metadata_path.exists()

    def test_archive_experiment_custom_location(self, temp_dir):
        """Test archiving experiment to custom location."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"
        archive_dir = temp_dir / "custom_archive"

        storage.create_experiment_directory(experiment_id)

        archive_path = storage.archive_experiment(experiment_id, archive_dir)

        expected_path = archive_dir / experiment_id
        assert archive_path == expected_path
        assert archive_path.exists()

    def test_archive_experiment_already_archived(self, temp_dir):
        """Test archiving experiment when archive already exists."""
        storage = ExperimentStorage(temp_dir)
        experiment_id = "abc12345"
        archive_dir = temp_dir / "archive"

        storage.create_experiment_directory(experiment_id)

        # Create conflicting archive directory
        archive_dir.mkdir()
        (archive_dir / experiment_id).mkdir()

        with pytest.raises(StorageError, match="Archive path already exists"):
            storage.archive_experiment(experiment_id, archive_dir)
