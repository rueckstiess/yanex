"""
Tests for experiment manager core functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import (
    DirtyWorkingDirectoryError,
    ExperimentAlreadyRunningError,
    ExperimentNotFoundError,
    ValidationError,
)


class TestExperimentManager:
    """Test ExperimentManager class."""

    def test_init_default_directory(self):
        """Test manager initialization with default directory."""
        manager = ExperimentManager()
        
        expected_dir = Path.home() / ".yanex" / "experiments"
        assert manager.experiments_dir == expected_dir
        assert manager.storage.experiments_dir == expected_dir

    def test_init_custom_directory(self):
        """Test manager initialization with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom_experiments"
            manager = ExperimentManager(custom_dir)
            
            assert manager.experiments_dir == custom_dir
            assert manager.storage.experiments_dir == custom_dir

    def test_generate_experiment_id_format(self):
        """Test experiment ID generation format."""
        manager = ExperimentManager()
        
        experiment_id = manager.generate_experiment_id()
        
        # Should be 8 characters, all hex
        assert len(experiment_id) == 8
        assert all(c in "0123456789abcdef" for c in experiment_id)

    def test_generate_experiment_id_uniqueness(self):
        """Test experiment ID uniqueness."""
        manager = ExperimentManager()
        
        # Generate multiple IDs
        ids = [manager.generate_experiment_id() for _ in range(100)]
        
        # All should be unique
        assert len(set(ids)) == len(ids)

    @patch('yanex.core.manager.secrets.token_hex')
    def test_generate_experiment_id_collision_detection(self, mock_token_hex):
        """Test ID generation handles collisions properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create an existing experiment directory
            existing_id = "deadbeef"
            (experiments_dir / existing_id).mkdir(parents=True)
            
            # Mock token_hex to return collision first, then unique ID
            mock_token_hex.side_effect = [existing_id, "cafebabe"]
            
            # Generate ID should avoid collision
            new_id = manager.generate_experiment_id()
            assert new_id == "cafebabe"
            assert mock_token_hex.call_count == 2

    @patch('yanex.core.manager.secrets.token_hex')
    def test_generate_experiment_id_max_retries(self, mock_token_hex):
        """Test ID generation fails after max retries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment directory for collision
            existing_id = "deadbeef"
            (experiments_dir / existing_id).mkdir(parents=True)
            
            # Mock to always return the same colliding ID
            mock_token_hex.return_value = existing_id
            
            # Should raise RuntimeError after 10 attempts
            with pytest.raises(RuntimeError, match="Failed to generate unique experiment ID"):
                manager.generate_experiment_id()
            
            assert mock_token_hex.call_count == 10

    def test_get_running_experiment_none_found(self):
        """Test get_running_experiment when no experiments exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            result = manager.get_running_experiment()
            assert result is None

    def test_get_running_experiment_no_directory(self):
        """Test get_running_experiment when experiments directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a path that doesn't exist but is in a writable location
            nonexistent_path = Path(temp_dir) / "nonexistent"
            # Don't create the directory - let storage create it, then remove it
            manager = ExperimentManager(nonexistent_path)
            # Remove the directory that storage created
            manager.experiments_dir.rmdir()
            
            result = manager.get_running_experiment()
            assert result is None

    def test_get_running_experiment_found(self):
        """Test get_running_experiment finds running experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment with running status
            experiment_id = "abc12345"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "running",
                "created_at": "2023-01-01T00:00:00"
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Should find the running experiment
            result = manager.get_running_experiment()
            assert result == experiment_id

    def test_get_running_experiment_ignores_completed(self):
        """Test get_running_experiment ignores non-running experiments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment with completed status
            experiment_id = "abc12345"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "completed",
                "created_at": "2023-01-01T00:00:00"
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Should not find any running experiment
            result = manager.get_running_experiment()
            assert result is None

    def test_get_running_experiment_ignores_corrupted_metadata(self):
        """Test get_running_experiment ignores experiments with corrupted metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment directory with invalid metadata
            experiment_id = "abc12345"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            # Write invalid JSON
            metadata_path = exp_dir / "metadata.json"
            metadata_path.write_text("invalid json content")
            
            # Should not crash and return None
            result = manager.get_running_experiment()
            assert result is None

    def test_get_running_experiment_multiple_experiments(self):
        """Test get_running_experiment with multiple experiments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create multiple experiments
            for i, status in enumerate(["completed", "running", "failed"]):
                experiment_id = f"exp{i:05d}"
                exp_dir = experiments_dir / experiment_id
                exp_dir.mkdir(parents=True)
                
                metadata = {
                    "id": experiment_id,
                    "status": status,
                    "created_at": "2023-01-01T00:00:00"
                }
                manager.storage.save_metadata(experiment_id, metadata)
            
            # Should find the running experiment
            result = manager.get_running_experiment()
            assert result == "exp00001"

    def test_prevent_concurrent_execution_no_running(self):
        """Test prevent_concurrent_execution when no experiment is running."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            # Should not raise any exception
            manager.prevent_concurrent_execution()

    def test_prevent_concurrent_execution_with_running(self):
        """Test prevent_concurrent_execution raises error when experiment is running."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create running experiment
            experiment_id = "running123"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "running",
                "created_at": "2023-01-01T00:00:00"
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Should raise ExperimentAlreadyRunningError
            with pytest.raises(ExperimentAlreadyRunningError) as exc_info:
                manager.prevent_concurrent_execution()
            
            assert experiment_id in str(exc_info.value)
            assert "already running" in str(exc_info.value).lower()

    def test_prevent_concurrent_execution_error_type(self):
        """Test prevent_concurrent_execution raises correct exception type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create running experiment
            experiment_id = "running456"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {"id": experiment_id, "status": "running"}
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Should raise specific exception type
            with pytest.raises(ExperimentAlreadyRunningError):
                manager.prevent_concurrent_execution()


class TestExperimentCreation:
    """Test experiment creation functionality."""

    @patch('yanex.core.manager.validate_clean_working_directory')
    @patch('yanex.core.manager.get_current_commit_info')
    @patch('yanex.core.manager.capture_full_environment')
    def test_create_experiment_basic(self, mock_capture_env, mock_git_info, mock_validate_git):
        """Test basic experiment creation."""
        # Setup mocks
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            script_path = Path(__file__)  # Use this test file as script
            experiment_id = manager.create_experiment(script_path)
            
            # Verify experiment was created
            assert len(experiment_id) == 8
            assert manager.storage.experiment_exists(experiment_id)
            
            # Verify metadata
            metadata = manager.storage.load_metadata(experiment_id)
            assert metadata["id"] == experiment_id
            assert metadata["script_path"] == str(script_path.resolve())
            assert metadata["status"] == "created"
            assert metadata["name"] is None
            assert metadata["tags"] == []
            assert metadata["description"] is None
            assert "created_at" in metadata
            assert "git" in metadata
            assert "environment" in metadata

    @patch('yanex.core.manager.validate_clean_working_directory')
    @patch('yanex.core.manager.get_current_commit_info')
    @patch('yanex.core.manager.capture_full_environment')
    def test_create_experiment_with_options(self, mock_capture_env, mock_git_info, mock_validate_git):
        """Test experiment creation with all options."""
        # Setup mocks
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            script_path = Path(__file__)
            name = "test-experiment"
            config = {"learning_rate": 0.01, "epochs": 10}
            tags = ["ml", "test"]
            description = "Test experiment description"
            
            experiment_id = manager.create_experiment(
                script_path, name=name, config=config, tags=tags, description=description
            )
            
            # Verify metadata
            metadata = manager.storage.load_metadata(experiment_id)
            assert metadata["name"] == name
            assert metadata["tags"] == tags
            assert metadata["description"] == description
            
            # Verify config was saved
            saved_config = manager.storage.load_config(experiment_id)
            assert saved_config == config

    @patch('yanex.core.manager.validate_clean_working_directory')
    def test_create_experiment_dirty_git(self, mock_validate_git):
        """Test experiment creation fails with dirty git working directory."""
        mock_validate_git.side_effect = DirtyWorkingDirectoryError(["modified.py"])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            with pytest.raises(DirtyWorkingDirectoryError):
                manager.create_experiment(Path(__file__))

    @patch('yanex.core.manager.validate_clean_working_directory')
    def test_create_experiment_concurrent_execution(self, mock_validate_git):
        """Test experiment creation fails when another experiment is running."""
        mock_validate_git.return_value = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create running experiment
            running_id = "running123"
            exp_dir = experiments_dir / running_id
            exp_dir.mkdir(parents=True)
            metadata = {"id": running_id, "status": "running"}
            manager.storage.save_metadata(running_id, metadata)
            
            # Should fail to create new experiment
            with pytest.raises(ExperimentAlreadyRunningError):
                manager.create_experiment(Path(__file__))

    @patch('yanex.core.manager.validate_clean_working_directory')
    @patch('yanex.core.manager.get_current_commit_info')
    @patch('yanex.core.manager.capture_full_environment')
    def test_create_experiment_invalid_name(self, mock_capture_env, mock_git_info, mock_validate_git):
        """Test experiment creation fails with invalid name."""
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            # Invalid name should raise ValidationError
            with pytest.raises(ValidationError):
                manager.create_experiment(Path(__file__), name="invalid@name#with$symbols")

    @patch('yanex.core.manager.validate_clean_working_directory')
    @patch('yanex.core.manager.get_current_commit_info')
    @patch('yanex.core.manager.capture_full_environment')
    def test_create_experiment_duplicate_name(self, mock_capture_env, mock_git_info, mock_validate_git):
        """Test experiment creation fails with duplicate name."""
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create first experiment
            name = "duplicate-name"
            manager.create_experiment(Path(__file__), name=name)
            
            # Second experiment with same name should fail
            with pytest.raises(ValueError, match="already in use"):
                manager.create_experiment(Path(__file__), name=name)

    @patch('yanex.core.manager.validate_clean_working_directory')
    @patch('yanex.core.manager.get_current_commit_info')
    @patch('yanex.core.manager.capture_full_environment')
    def test_create_experiment_invalid_tags(self, mock_capture_env, mock_git_info, mock_validate_git):
        """Test experiment creation fails with invalid tags."""
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            # Invalid tags should raise ValidationError
            with pytest.raises(ValidationError):
                manager.create_experiment(Path(__file__), tags=["invalid tag with spaces"])

    def test_find_experiment_by_name_not_found(self):
        """Test find_experiment_by_name when experiment doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            result = manager.find_experiment_by_name("nonexistent")
            assert result is None

    def test_find_experiment_by_name_no_directory(self):
        """Test find_experiment_by_name when experiments directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "nonexistent"
            manager = ExperimentManager(nonexistent_path)
            manager.experiments_dir.rmdir()  # Remove directory created by storage
            
            result = manager.find_experiment_by_name("test")
            assert result is None

    def test_find_experiment_by_name_found(self):
        """Test find_experiment_by_name finds experiment by name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment with name
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "name": "test-experiment",
                "status": "completed"
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Should find the experiment
            result = manager.find_experiment_by_name("test-experiment")
            assert result == experiment_id

    def test_find_experiment_by_name_ignores_corrupted(self):
        """Test find_experiment_by_name ignores corrupted metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment with corrupted metadata
            experiment_id = "corrupt12"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            # Write invalid JSON
            metadata_path = exp_dir / "metadata.json"
            metadata_path.write_text("invalid json")
            
            # Should not crash and return None
            result = manager.find_experiment_by_name("test")
            assert result is None

    @patch('yanex.core.manager.get_current_commit_info')
    @patch('yanex.core.manager.capture_full_environment')
    def test_build_metadata(self, mock_capture_env, mock_git_info):
        """Test build_metadata creates complete metadata."""
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            experiment_id = "test1234"
            script_path = Path(__file__)
            name = "test-experiment"
            tags = ["ml", "test"]
            description = "Test description"
            
            metadata = manager.build_metadata(
                experiment_id, script_path, name, tags, description
            )
            
            # Verify all fields are present
            assert metadata["id"] == experiment_id
            assert metadata["name"] == name
            assert metadata["script_path"] == str(script_path.resolve())
            assert metadata["tags"] == tags
            assert metadata["description"] == description
            assert metadata["status"] == "created"
            assert "created_at" in metadata
            assert metadata["started_at"] is None
            assert metadata["completed_at"] is None
            assert metadata["duration"] is None
            assert metadata["git"] == {"commit": "abc123", "branch": "main"}
            assert metadata["environment"] == {"python_version": "3.11.0"}


class TestExperimentLifecycle:
    """Test experiment lifecycle management functionality."""

    def test_start_experiment_success(self):
        """Test successful experiment start."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment in 'created' state
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "created",
                "created_at": "2023-01-01T00:00:00",
                "started_at": None
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Start the experiment
            manager.start_experiment(experiment_id)
            
            # Verify status changed
            updated_metadata = manager.storage.load_metadata(experiment_id)
            assert updated_metadata["status"] == "running"
            assert updated_metadata["started_at"] is not None
            assert "T" in updated_metadata["started_at"]  # Should be ISO format timestamp

    def test_start_experiment_not_found(self):
        """Test starting non-existent experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            with pytest.raises(ExperimentNotFoundError):
                manager.start_experiment("nonexistent")

    def test_start_experiment_wrong_status(self):
        """Test starting experiment in wrong state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment in 'completed' state
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "completed",
                "created_at": "2023-01-01T00:00:00"
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Should fail to start
            with pytest.raises(ValueError, match="Expected status 'created'"):
                manager.start_experiment(experiment_id)

    def test_complete_experiment_success(self):
        """Test successful experiment completion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "running",
                "started_at": "2023-01-01T12:00:00",
                "completed_at": None,
                "duration": None
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Complete the experiment
            manager.complete_experiment(experiment_id)
            
            # Verify status and timestamps
            updated_metadata = manager.storage.load_metadata(experiment_id)
            assert updated_metadata["status"] == "completed"
            assert updated_metadata["completed_at"] is not None
            assert updated_metadata["duration"] is not None

    def test_complete_experiment_no_start_time(self):
        """Test completing experiment without start time."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment without start time
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "running",
                "started_at": None
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Complete the experiment
            manager.complete_experiment(experiment_id)
            
            # Should complete but duration should be None
            updated_metadata = manager.storage.load_metadata(experiment_id)
            assert updated_metadata["status"] == "completed"
            assert updated_metadata["duration"] is None

    def test_fail_experiment_success(self):
        """Test successful experiment failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "running",
                "started_at": "2023-01-01T12:00:00"
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Fail the experiment
            error_message = "Test error message"
            manager.fail_experiment(experiment_id, error_message)
            
            # Verify status and error info
            updated_metadata = manager.storage.load_metadata(experiment_id)
            assert updated_metadata["status"] == "failed"
            assert updated_metadata["error_message"] == error_message
            assert updated_metadata["completed_at"] is not None
            assert updated_metadata["duration"] is not None

    def test_cancel_experiment_success(self):
        """Test successful experiment cancellation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {
                "id": experiment_id,
                "status": "running",
                "started_at": "2023-01-01T12:00:00"
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Cancel the experiment
            reason = "User requested cancellation"
            manager.cancel_experiment(experiment_id, reason)
            
            # Verify status and cancellation info
            updated_metadata = manager.storage.load_metadata(experiment_id)
            assert updated_metadata["status"] == "cancelled"
            assert updated_metadata["cancellation_reason"] == reason
            assert updated_metadata["completed_at"] is not None
            assert updated_metadata["duration"] is not None

    def test_get_experiment_status_success(self):
        """Test getting experiment status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            metadata = {"id": experiment_id, "status": "running"}
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Get status
            status = manager.get_experiment_status(experiment_id)
            assert status == "running"

    def test_get_experiment_status_not_found(self):
        """Test getting status of non-existent experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            with pytest.raises(ExperimentNotFoundError):
                manager.get_experiment_status("nonexistent")

    def test_get_experiment_metadata_success(self):
        """Test getting complete experiment metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            
            original_metadata = {
                "id": experiment_id,
                "status": "completed",
                "name": "test-experiment"
            }
            manager.storage.save_metadata(experiment_id, original_metadata)
            
            # Get metadata
            metadata = manager.get_experiment_metadata(experiment_id)
            assert metadata["id"] == experiment_id
            assert metadata["status"] == "completed"
            assert metadata["name"] == "test-experiment"

    def test_get_experiment_metadata_not_found(self):
        """Test getting metadata of non-existent experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            with pytest.raises(ExperimentNotFoundError):
                manager.get_experiment_metadata("nonexistent")

    def test_list_experiments_no_filter(self):
        """Test listing all experiments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create multiple experiments
            experiment_ids = ["exp00001", "exp00002", "exp00003"]
            for exp_id in experiment_ids:
                exp_dir = experiments_dir / exp_id
                exp_dir.mkdir(parents=True)
                metadata = {"id": exp_id, "status": "completed"}
                manager.storage.save_metadata(exp_id, metadata)
            
            # List all experiments
            listed_ids = manager.list_experiments()
            assert set(listed_ids) == set(experiment_ids)

    def test_list_experiments_with_filter(self):
        """Test listing experiments with status filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiments with different statuses
            experiments = [
                ("exp00001", "completed"),
                ("exp00002", "failed"),
                ("exp00003", "completed"),
                ("exp00004", "running")
            ]
            
            for exp_id, status in experiments:
                exp_dir = experiments_dir / exp_id
                exp_dir.mkdir(parents=True)
                metadata = {"id": exp_id, "status": status}
                manager.storage.save_metadata(exp_id, metadata)
            
            # Filter by completed status
            completed_ids = manager.list_experiments(status_filter="completed")
            assert set(completed_ids) == {"exp00001", "exp00003"}
            
            # Filter by failed status
            failed_ids = manager.list_experiments(status_filter="failed")
            assert failed_ids == ["exp00002"]

    def test_list_experiments_ignores_corrupted(self):
        """Test listing experiments ignores corrupted metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create valid experiment
            valid_id = "valid123"
            exp_dir = experiments_dir / valid_id
            exp_dir.mkdir(parents=True)
            metadata = {"id": valid_id, "status": "completed"}
            manager.storage.save_metadata(valid_id, metadata)
            
            # Create corrupted experiment
            corrupt_id = "corrupt12"
            corrupt_dir = experiments_dir / corrupt_id
            corrupt_dir.mkdir(parents=True)
            metadata_path = corrupt_dir / "metadata.json"
            metadata_path.write_text("invalid json")
            
            # Should only return valid experiment when filtering
            completed_ids = manager.list_experiments(status_filter="completed")
            assert completed_ids == [valid_id]

    def test_archive_experiment_success(self):
        """Test successful experiment archiving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment
            experiment_id = "test1234"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            metadata = {"id": experiment_id, "status": "completed"}
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Archive the experiment
            archive_path = manager.archive_experiment(experiment_id)
            
            # Verify experiment was moved
            assert not exp_dir.exists()
            assert archive_path.exists()
            assert archive_path.name == experiment_id

    def test_archive_experiment_not_found(self):
        """Test archiving non-existent experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(Path(temp_dir))
            
            with pytest.raises(ExperimentNotFoundError):
                manager.archive_experiment("nonexistent")

    def test_lifecycle_state_transitions(self):
        """Test complete lifecycle state transitions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir)
            manager = ExperimentManager(experiments_dir)
            
            # Create experiment (starts in 'created' state)
            experiment_id = "lifecycle1"
            exp_dir = experiments_dir / experiment_id
            exp_dir.mkdir(parents=True)
            metadata = {
                "id": experiment_id,
                "status": "created",
                "started_at": None,
                "completed_at": None,
                "duration": None
            }
            manager.storage.save_metadata(experiment_id, metadata)
            
            # Verify initial state
            assert manager.get_experiment_status(experiment_id) == "created"
            
            # Start experiment
            manager.start_experiment(experiment_id)
            assert manager.get_experiment_status(experiment_id) == "running"
            
            # Complete experiment
            manager.complete_experiment(experiment_id)
            assert manager.get_experiment_status(experiment_id) == "completed"
            
            # Verify final metadata
            final_metadata = manager.get_experiment_metadata(experiment_id)
            assert final_metadata["started_at"] is not None
            assert final_metadata["completed_at"] is not None
            assert final_metadata["duration"] is not None