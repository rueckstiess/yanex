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
    ) -> str:
        """Create new experiment with metadata.

        Args:
            script_path: Path to the Python script to run
            name: Optional experiment name
            config: Configuration dictionary
            tags: List of tags for the experiment
            description: Optional experiment description

        Returns:
            Experiment ID

        Raises:
            DirtyWorkingDirectoryError: If git working directory is not clean
            ValidationError: If input parameters are invalid
            ExperimentAlreadyRunningError: If another experiment is running
            StorageError: If experiment creation fails
        """
        # Validate git working directory is clean
        validate_clean_working_directory()

        # Prevent concurrent execution
        self.prevent_concurrent_execution()

        # Set defaults
        if config is None:
            config = {}
        if tags is None:
            tags = []

        # Validate inputs
        if name is not None:
            validate_experiment_name(name)
            # Check if name is already in use
            if self.find_experiment_by_name(name) is not None:
                raise ValueError(f"Experiment name '{name}' is already in use")

        validate_tags(tags)

        # Generate unique experiment ID
        experiment_id = self.generate_experiment_id()

        # Create experiment directory structure
        self.storage.create_experiment_directory(experiment_id)

        # Build and save metadata
        metadata = self.build_metadata(
            experiment_id, script_path, name, tags, description
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
    ) -> Dict[str, Any]:
        """Build complete experiment metadata.

        Args:
            experiment_id: Unique experiment identifier
            script_path: Path to the Python script
            name: Optional experiment name
            tags: List of experiment tags
            description: Optional experiment description

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
        metadata = {
            "id": experiment_id,
            "name": name,
            "script_path": str(script_path.resolve()),
            "tags": tags,
            "description": description,
            "status": "created",
            "created_at": timestamp,
            "started_at": None,
            "completed_at": None,
            "duration": None,
            "git": git_info,
            "environment": environment_info,
        }

        return metadata

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

