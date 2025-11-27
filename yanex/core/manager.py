"""
Experiment Manager - Core orchestration component for yanex.

Handles experiment lifecycle management, ID generation, and coordinates
between all core components (git, config, storage, environment).
"""

import logging
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import git

from ..utils.validation import validate_experiment_name, validate_tags
from .environment import capture_full_environment
from .git_utils import (
    check_patch_size,
    generate_git_patch,
    get_current_commit_info,
    scan_patch_for_secrets,
)
from .storage import ExperimentStorage


class ExperimentManager:
    """Central manager for experiment lifecycle and orchestration."""

    def __init__(self, experiments_dir: Path | None = None):
        """Initialize experiment manager.

        Args:
            experiments_dir: Directory for experiment storage.
                          Defaults to ~/.yanex/experiments unless YANEX_EXPERIMENTS_DIR
                          environment variable is set
        """
        if experiments_dir is None:
            import os

            env_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
            if env_dir:
                experiments_dir = Path(env_dir)
            else:
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

    def get_running_experiment(self) -> str | None:
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

    def get_running_experiments(self) -> list[str]:
        """Get all currently running experiments.

        Returns:
            List of experiment IDs with status='running'
        """
        if not self.experiments_dir.exists():
            return []

        running_experiments = []
        for experiment_dir in self.experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue

            experiment_id = experiment_dir.name
            if self.storage.experiment_exists(experiment_id):
                try:
                    metadata = self.storage.load_metadata(experiment_id)
                    if metadata.get("status") == "running":
                        running_experiments.append(experiment_id)
                except Exception:
                    # Skip experiments with corrupted metadata
                    continue

        return running_experiments

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
        import os

        now = datetime.utcnow().isoformat()
        metadata["status"] = "running"
        metadata["started_at"] = now
        metadata["process_id"] = os.getpid()

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

    def get_experiment_metadata(self, experiment_id: str) -> dict[str, Any]:
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

    def list_experiments(self, status_filter: str | None = None) -> list[str]:
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

    def create_experiment(
        self,
        script_path: Path,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        description: str | None = None,
        stage_only: bool = False,
        script_args: list[str] | None = None,
        cli_args: dict[str, Any] | None = None,
        dependencies: dict[str, str] | None = None,
    ) -> str:
        """Create new experiment with metadata.

        Args:
            script_path: Path to the Python script to run
            name: Optional experiment name
            config: Configuration dictionary
            tags: List of tags for the experiment
            description: Optional experiment description
            stage_only: If True, create experiment with "staged" status for later execution
            script_args: Arguments to pass through to the script via sys.argv
            cli_args: Parsed CLI arguments dictionary (yanex flags only, not script_args)
            dependencies: Dict mapping slot names to experiment IDs this depends on

        Returns:
            Experiment ID

        Raises:
            ValidationError: If input parameters are invalid
            StorageError: If experiment creation fails
            CircularDependencyError: If circular dependency detected
            InvalidDependencyError: If dependency validation fails
        """

        # Set defaults
        if config is None:
            config = {}
        if tags is None:
            tags = []
        if dependencies is None:
            dependencies = {}

        # Validate inputs
        if name is not None:
            validate_experiment_name(name)
            # Note: We allow duplicate names to support experiment grouping

        validate_tags(tags)

        # Generate unique experiment ID
        experiment_id = self.generate_experiment_id()

        # Resolve and validate dependencies if provided
        resolved_dependencies: dict[str, str] = {}
        dependency_metadata = {}
        if dependencies:
            from .dependencies import DependencyResolver

            resolver = DependencyResolver(self)

            # Resolve short IDs and validate
            resolved_dependencies = resolver.resolve_and_validate_dependencies(
                dependencies, for_staging=stage_only, include_archived=True
            )

            # Check for circular dependencies
            for dep_id in resolved_dependencies.values():
                if resolver.detect_circular_dependency(
                    experiment_id, dep_id, include_archived=True
                ):
                    from ..utils.exceptions import CircularDependencyError

                    raise CircularDependencyError(
                        f"Cannot add dependency on '{dep_id}'. "
                        f"This would create a circular dependency chain."
                    )

            # Build metadata about dependencies for debugging
            for dep_id in resolved_dependencies.values():
                try:
                    dep_metadata = self.storage.load_metadata(
                        dep_id, include_archived=True
                    )
                    dependency_metadata[dep_id] = {
                        "resolved_at": datetime.utcnow().isoformat(),
                        "status_at_resolution": dep_metadata.get("status"),
                        "script_path": dep_metadata.get("script_path"),
                        "name": dep_metadata.get("name"),
                    }
                except Exception:
                    # If we can't load metadata, just skip it
                    pass

        # Create experiment directory structure
        self.storage.create_experiment_directory(experiment_id)

        # Build and save metadata
        metadata = self.build_metadata(
            experiment_id,
            script_path,
            name,
            tags,
            description,
            stage_only,
            script_args,
            cli_args,
        )
        self.storage.save_metadata(experiment_id, metadata)

        # Save resolved configuration
        self.storage.save_config(experiment_id, config)

        # Save dependencies if any
        if resolved_dependencies:
            self.storage.dependency_storage.save_dependencies(
                experiment_id, resolved_dependencies, dependency_metadata
            )

        return experiment_id

    def build_metadata(
        self,
        experiment_id: str,
        script_path: Path,
        name: str | None,
        tags: list[str],
        description: str | None,
        stage_only: bool = False,
        script_args: list[str] | None = None,
        cli_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build complete experiment metadata.

        Args:
            experiment_id: Unique experiment identifier
            script_path: Path to the Python script
            name: Optional experiment name
            tags: List of experiment tags
            description: Optional experiment description
            stage_only: If True, create with "staged" status
            script_args: Arguments to pass through to the script via sys.argv
            cli_args: Parsed CLI arguments dictionary (yanex flags only, not script_args)

        Returns:
            Complete metadata dictionary
        """
        logger = logging.getLogger(__name__)

        # Get current timestamp
        timestamp = datetime.utcnow().isoformat()

        # Capture git information
        git_info = get_current_commit_info()

        # Capture git patch if uncommitted changes exist
        git_patch = None
        patch_filename = None
        patch_size_info = None
        secret_scan_results = None

        try:
            git_patch = generate_git_patch()
            if git_patch:
                patch_filename = "git_diff.patch"

                # Check patch size
                patch_size_info = check_patch_size(git_patch, max_size_mb=1.0)
                if patch_size_info["exceeds_limit"]:
                    logger.warning(
                        f"Git patch size ({patch_size_info['size_mb']} MB) exceeds "
                        f"recommended limit of 1.0 MB. Large patches may impact "
                        f"performance and storage."
                    )

                # Scan for potential secrets
                secret_scan_results = scan_patch_for_secrets(git_patch)
                if secret_scan_results["has_secrets"]:
                    findings = secret_scan_results["findings"]
                    logger.warning(
                        f"Potential secrets detected in git patch! "
                        f"Found {len(findings)} potential secret(s). "
                        f"Review patch content before sharing or committing."
                    )
                    # Log details about findings
                    for finding in findings:
                        logger.warning(
                            f"  - {finding['type']} in {finding['filename']} "
                            f"at line {finding['line']}"
                        )

        except git.GitError as e:
            logger.warning(f"Git operation failed while generating patch: {e}")
            # Continue without patch
        except OSError as e:
            logger.warning(f"File system error while generating patch: {e}")
            # Continue without patch
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to process patch data: {type(e).__name__}: {e}")
            # Continue without patch
        except Exception as e:
            logger.warning(
                f"Unexpected error while generating patch: {type(e).__name__}: {e}"
            )
            # Continue without patch

        # Add patch info to git metadata
        git_info["has_uncommitted_changes"] = git_patch is not None
        git_info["patch_file"] = (
            f"artifacts/{patch_filename}" if patch_filename else None
        )

        # Add security and size metadata if patch exists
        if git_patch:
            git_info["patch_size_bytes"] = (
                patch_size_info["size_bytes"] if patch_size_info else None
            )
            git_info["patch_size_mb"] = (
                patch_size_info["size_mb"] if patch_size_info else None
            )
            git_info["patch_has_secrets"] = (
                secret_scan_results["has_secrets"] if secret_scan_results else False
            )
            git_info["patch_secret_count"] = (
                len(secret_scan_results["findings"]) if secret_scan_results else 0
            )

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
            "script_args": script_args if script_args else [],
            "cli_args": cli_args if cli_args else {},
        }

        # Save patch as artifact if it exists
        if git_patch and patch_filename:
            try:
                self.storage.save_text_artifact(
                    experiment_id, patch_filename, git_patch
                )
            except Exception as e:
                logger.warning(f"Failed to save git patch artifact: {e}")
                # Update metadata to reflect failure
                metadata["git"]["patch_file"] = None

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
        import os

        now = datetime.utcnow().isoformat()
        metadata["status"] = "running"
        metadata["started_at"] = now
        metadata["process_id"] = os.getpid()
        self.storage.save_metadata(experiment_id, metadata)

    def get_staged_experiments(self) -> list[str]:
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

    def find_experiment_by_name(self, name: str) -> str | None:
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
