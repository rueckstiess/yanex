"""
Storage management for experiments.
"""

from pathlib import Path

from .storage_composition import CompositeExperimentStorage


class ExperimentStorage(CompositeExperimentStorage):
    """Manages file storage for experiments.

    This class provides a backwards-compatible interface while using
    the new modular storage architecture internally.
    """

    def __init__(self, experiments_dir: Path = None):
        """Initialize experiment storage.

        Args:
            experiments_dir: Base directory for experiments, defaults to ./experiments
        """
        super().__init__(experiments_dir)

    @property
    def experiments_dir(self) -> Path:
        """Access to experiments directory for backwards compatibility."""
        return self.directory_manager.experiments_dir
