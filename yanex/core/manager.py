"""
Experiment Manager - Core orchestration component for yanex.

Handles experiment lifecycle management, ID generation, and coordinates
between all core components (git, config, storage, environment).
"""

import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ExperimentAlreadyRunningError
from ..utils.validation import validate_experiment_name, validate_tags
from .environment import capture_full_environment
from .git_utils import get_current_commit_info, validate_clean_working_directory
from .storage import ExperimentStorage


class ExperimentManager:
    """Central manager for experiment lifecycle and orchestration."""

    def __init__(self, experiments_dir: Optional[Path] = None):
        """Initialize experiment manager.

        Args:
            experiments_dir: Directory for experiment storage.
                          Defaults to ~/.yanex/experiments
        """
        if experiments_dir is None:
            experiments_dir = Path.home() / ".yanex" / "experiments"

        self.experiments_dir = experiments_dir
        self.storage = ExperimentStorage(experiments_dir)

    def generate_experiment_id(self) -> str:
        """Generate unique 8-character hex experiment ID.

        Uses cryptographically secure random generation and checks for
        collisions with existing experiments. Retries up to 10 times
        before failing.

        Returns:
            Unique 8-character hex string

        Raises:
            RuntimeError: If unable to generate unique ID after 10 attempts
        """
        max_attempts = 10

        for _ in range(max_attempts):
            # Generate 8-character hex ID using secure random
            experiment_id = secrets.token_hex(4)

            # Check for collision with existing experiments
            if not self.storage.experiment_exists(experiment_id):
                return experiment_id

        # If we reach here, we had collisions on all attempts
        raise RuntimeError(
            f"Failed to generate unique experiment ID after {max_attempts} attempts"
        )

    def get_running_experiment(self) -> Optional[str]:
        """Check if there's currently a running experiment.

        Scans the experiments directory for experiments with status='running'.

        Returns:
            Experiment ID of running experiment, or None if no running experiment
        """
        if not self.experiments_dir.exists():
            return None

        # Scan all experiment directories
        for experiment_dir in self.experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue

            experiment_id = experiment_dir.name

            # Check if this experiment exists and is running
            if self.storage.experiment_exists(experiment_id):
                try:
                    metadata = self.storage.load_metadata(experiment_id)
                    if metadata.get("status") == "running":
                        return experiment_id
                except Exception:
                    # Skip experiments with corrupted metadata
                    continue

        return None

    def start_experiment(self, experiment_id: str) -> None:
        """Transition experiment to running state.

        Args:
            experiment_id: Experiment identifier

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ValueError: If experiment is not in 'created' state
            StorageError: If metadata update fails
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        # Load current metadata
        metadata = self.storage.load_metadata(experiment_id)

        # Verify experiment is in correct state
        if metadata.get("status") != "created":
            current_status = metadata.get("status", "unknown")
            raise ValueError(
                f"Cannot start experiment {experiment_id}. "
                f"Expected status 'created', got '{current_status}'"
            )

        # Update status and timestamps
        now = datetime.utcnow().isoformat()
        metadata["status"] = "running"
        metadata["started_at"] = now

        # Save updated metadata
        self.storage.save_metadata(experiment_id, metadata)

    def complete_experiment(self, experiment_id: str) -> None:
        """Mark experiment as completed.

        Args:
            experiment_id: Experiment identifier

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            StorageError: If metadata update fails
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        # Load current metadata
        metadata = self.storage.load_metadata(experiment_id)

        # Update status and timestamps
        now = datetime.utcnow().isoformat()
        metadata["status"] = "completed"
        metadata["completed_at"] = now

        # Calculate duration if we have start time
        if metadata.get("started_at"):
            try:
                start_time = datetime.fromisoformat(metadata["started_at"])
                end_time = datetime.fromisoformat(now)
                duration = (end_time - start_time).total_seconds()
                metadata["duration"] = duration
            except (ValueError, TypeError):
                # If we can't parse timestamps, skip duration calculation
                metadata["duration"] = None
        else:
            metadata["duration"] = None

        # Save updated metadata
        self.storage.save_metadata(experiment_id, metadata)

    def fail_experiment(self, experiment_id: str, error_message: str) -> None:
        """Mark experiment as failed with error details.

        Args:
            experiment_id: Experiment identifier
            error_message: Error message describing the failure

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            StorageError: If metadata update fails
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        # Load current metadata
        metadata = self.storage.load_metadata(experiment_id)

        # Update status and error information
        now = datetime.utcnow().isoformat()
        metadata["status"] = "failed"
        metadata["completed_at"] = now
        metadata["error_message"] = error_message

        # Calculate duration if we have start time
        if metadata.get("started_at"):
            try:
                start_time = datetime.fromisoformat(metadata["started_at"])
                end_time = datetime.fromisoformat(now)
                duration = (end_time - start_time).total_seconds()
                metadata["duration"] = duration
            except (ValueError, TypeError):
                metadata["duration"] = None
        else:
            metadata["duration"] = None

        # Save updated metadata
        self.storage.save_metadata(experiment_id, metadata)

    def cancel_experiment(self, experiment_id: str, reason: str) -> None:
        """Mark experiment as cancelled with reason.

        Args:
            experiment_id: Experiment identifier
            reason: Reason for cancellation

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            StorageError: If metadata update fails
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        # Load current metadata
        metadata = self.storage.load_metadata(experiment_id)

        # Update status and cancellation information
        now = datetime.utcnow().isoformat()
        metadata["status"] = "cancelled"
        metadata["completed_at"] = now
        metadata["cancellation_reason"] = reason

        # Calculate duration if we have start time
        if metadata.get("started_at"):
            try:
                start_time = datetime.fromisoformat(metadata["started_at"])
                end_time = datetime.fromisoformat(now)
                duration = (end_time - start_time).total_seconds()
                metadata["duration"] = duration
            except (ValueError, TypeError):
                metadata["duration"] = None
        else:
            metadata["duration"] = None

        # Save updated metadata
        self.storage.save_metadata(experiment_id, metadata)

    def get_experiment_status(self, experiment_id: str) -> str:
        """Get current status of an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Current experiment status

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            StorageError: If metadata cannot be loaded
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        # Load metadata and return status
        metadata = self.storage.load_metadata(experiment_id)
        return metadata.get("status", "unknown")

    def get_experiment_metadata(self, experiment_id: str) -> Dict[str, Any]:
        """Get complete metadata for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Complete experiment metadata

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            StorageError: If metadata cannot be loaded
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        return self.storage.load_metadata(experiment_id)

    def list_experiments(self, status_filter: Optional[str] = None) -> List[str]:
        """List experiment IDs, optionally filtered by status.

        Args:
            status_filter: Optional status to filter by (e.g., 'completed', 'failed')

        Returns:
            List of experiment IDs matching the criteria
        """
        experiment_ids = self.storage.list_experiments()

        if status_filter is None:
            return experiment_ids

        # Filter by status
        filtered_ids = []
        for experiment_id in experiment_ids:
            try:
                if self.get_experiment_status(experiment_id) == status_filter:
                    filtered_ids.append(experiment_id)
            except Exception:
                # Skip experiments with corrupted metadata
                continue

        return filtered_ids

    def archive_experiment(self, experiment_id: str) -> Path:
        """Archive an experiment by moving it to archive directory.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Path where experiment was archived

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            StorageError: If archiving fails
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        return self.storage.archive_experiment(experiment_id)

    def prevent_concurrent_execution(self) -> None:
        """Ensure no other experiment is currently running.

        Raises:
            ExperimentAlreadyRunningError: If another experiment is running
        """
        running_experiment = self.get_running_experiment()
        if running_experiment is not None:
            raise ExperimentAlreadyRunningError(
                f"Experiment {running_experiment} is already running. "
                "Only one experiment can run at a time."
            )

    def create_experiment(
        self,
        script_path: Path,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        allow_dirty: bool = False,
        stage_only: bool = False,
    ) -> str:
        """Create new experiment with metadata.

        Args:
            script_path: Path to the Python script to run
            name: Optional experiment name
            config: Configuration dictionary
            tags: List of tags for the experiment
            description: Optional experiment description
            allow_dirty: Allow running with uncommitted changes
            stage_only: If True, create experiment with "staged" status for later execution

        Returns:
            Experiment ID

        Raises:
            DirtyWorkingDirectoryError: If git working directory is not clean and allow_dirty=False
            ValidationError: If input parameters are invalid
            ExperimentAlreadyRunningError: If another experiment is running (unless stage_only=True)
            StorageError: If experiment creation fails
        """
        # Validate git working directory is clean (unless explicitly allowed)
        if not allow_dirty:
            validate_clean_working_directory()

        # Prevent concurrent execution (unless staging only)
        if not stage_only:
            self.prevent_concurrent_execution()

        # Set defaults
        if config is None:
            config = {}
        if tags is None:
            tags = []

        # Validate inputs
        if name is not None:
            validate_experiment_name(name)
            # Note: We allow duplicate names to support experiment grouping

        validate_tags(tags)

        # Generate unique experiment ID
        experiment_id = self.generate_experiment_id()

        # Create experiment directory structure
        self.storage.create_experiment_directory(experiment_id)

        # Build and save metadata
        metadata = self.build_metadata(
            experiment_id, script_path, name, tags, description, stage_only
        )
        self.storage.save_metadata(experiment_id, metadata)

        # Save resolved configuration
        self.storage.save_config(experiment_id, config)

        return experiment_id

    def build_metadata(
        self,
        experiment_id: str,
        script_path: Path,
        name: Optional[str],
        tags: List[str],
        description: Optional[str],
        stage_only: bool = False,
    ) -> Dict[str, Any]:
        """Build complete experiment metadata.

        Args:
            experiment_id: Unique experiment identifier
            script_path: Path to the Python script
            name: Optional experiment name
            tags: List of experiment tags
            description: Optional experiment description
            stage_only: If True, create with "staged" status

        Returns:
            Complete metadata dictionary
        """
        # Get current timestamp
        timestamp = datetime.utcnow().isoformat()

        # Capture git information
        git_info = get_current_commit_info()

        # Capture environment information
        environment_info = capture_full_environment()

        # Build metadata
        status = "staged" if stage_only else "created"
        metadata = {
            "id": experiment_id,
            "name": name,
            "script_path": str(script_path.resolve()),
            "tags": tags,
            "description": description,
            "status": status,
            "created_at": timestamp,
            "started_at": None,
            "completed_at": None,
            "duration": None,
            "git": git_info,
            "environment": environment_info,
        }

        return metadata

    def execute_staged_experiment(self, experiment_id: str) -> None:
        """Execute a staged experiment.

        Args:
            experiment_id: Experiment identifier

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ValueError: If experiment is not in 'staged' state
            StorageError: If metadata update fails
        """
        # Verify experiment exists
        if not self.storage.experiment_exists(experiment_id):
            from ..utils.exceptions import ExperimentNotFoundError

            raise ExperimentNotFoundError(experiment_id)

        # Load current metadata
        metadata = self.storage.load_metadata(experiment_id)

        # Verify experiment is in staged state
        if metadata.get("status") != "staged":
            current_status = metadata.get("status", "unknown")
            raise ValueError(
                f"Cannot execute experiment {experiment_id}. "
                f"Expected status 'staged', got '{current_status}'"
            )

        # Transition to running state
        now = datetime.utcnow().isoformat()
        metadata["status"] = "running"
        metadata["started_at"] = now
        self.storage.save_metadata(experiment_id, metadata)

    def get_staged_experiments(self) -> List[str]:
        """Get list of staged experiment IDs.

        Returns:
            List of experiment IDs with status 'staged'
        """
        all_experiments = self.storage.list_experiments(include_archived=False)
        staged_experiments = []

        for exp_id in all_experiments:
            try:
                metadata = self.storage.load_metadata(exp_id)
                if metadata.get("status") == "staged":
                    staged_experiments.append(exp_id)
            except Exception:
                # Skip experiments with loading errors
                continue

        return staged_experiments

    def find_experiment_by_name(self, name: str) -> Optional[str]:
        """Find experiment ID by name.

        Args:
            name: Experiment name to search for

        Returns:
            Experiment ID if found, None otherwise
        """
        if not self.experiments_dir.exists():
            return None

        # Search through all experiment directories
        for experiment_dir in self.experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue

            experiment_id = experiment_dir.name
            if self.storage.experiment_exists(experiment_id):
                try:
                    metadata = self.storage.load_metadata(experiment_id)
                    if metadata.get("name") == name:
                        return experiment_id
                except Exception:
                    # Skip experiments with corrupted metadata
                    continue

        return None
