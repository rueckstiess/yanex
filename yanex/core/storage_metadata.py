"""Metadata storage for experiments."""

import json
from datetime import datetime
from typing import Any

from ..utils.exceptions import StorageError
from .storage_interfaces import ExperimentDirectoryManager, MetadataStorage


class FileSystemMetadataStorage(MetadataStorage):
    """File system-based experiment metadata storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize metadata storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def save_metadata(
        self,
        experiment_id: str,
        metadata: dict[str, Any],
        include_archived: bool = False,
    ) -> None:
        """Save experiment metadata.

        Args:
            experiment_id: Experiment identifier
            metadata: Metadata dictionary to save
            include_archived: Whether to search archived experiments too

        Raises:
            StorageError: If metadata cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        metadata_path = exp_dir / "metadata.json"

        # Add timestamp
        metadata_with_timestamp = metadata.copy()
        metadata_with_timestamp["saved_at"] = datetime.utcnow().isoformat()

        try:
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata_with_timestamp, f, indent=2, sort_keys=True)
        except Exception as e:
            raise StorageError(f"Failed to save metadata: {e}") from e

    def load_metadata(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any]:
        """Load experiment metadata.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            Metadata dictionary

        Raises:
            StorageError: If metadata cannot be loaded
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        metadata_path = exp_dir / "metadata.json"

        if not metadata_path.exists():
            raise StorageError(f"Metadata file not found: {metadata_path}")

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(f"Failed to load metadata: {e}") from e

    def update_experiment_metadata(
        self,
        experiment_id: str,
        updates: dict[str, Any],
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """Update experiment metadata with new values.

        Args:
            experiment_id: Experiment identifier
            updates: Dictionary of metadata updates to apply
            include_archived: Whether to search archived experiments too

        Returns:
            Updated metadata dictionary

        Raises:
            StorageError: If metadata cannot be updated
        """
        # Load current metadata
        current_metadata = self.load_metadata(experiment_id, include_archived)

        # Apply updates
        updated_metadata = current_metadata.copy()

        # Handle tag operations first (before the main loop)
        if "add_tags" in updates or "remove_tags" in updates:
            current_tags = set(updated_metadata.get("tags", []))

            if "add_tags" in updates:
                current_tags.update(updates["add_tags"])

            if "remove_tags" in updates:
                current_tags.difference_update(updates["remove_tags"])

            updated_metadata["tags"] = sorted(current_tags)

        # Handle other field updates
        for key, value in updates.items():
            if key in ["add_tags", "remove_tags"]:
                # Skip these as they're handled above
                continue
            elif key in ["name", "description", "status"]:
                # Direct field updates
                if value == "":
                    # Empty string means clear the field
                    updated_metadata[key] = None
                else:
                    updated_metadata[key] = value
            else:
                # Other fields - direct assignment
                updated_metadata[key] = value

        # Save updated metadata
        self.save_metadata(experiment_id, updated_metadata, include_archived)

        return updated_metadata
