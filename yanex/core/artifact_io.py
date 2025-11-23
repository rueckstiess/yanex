"""Unified artifact save/load logic with automatic format detection."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifact_formats import get_handler_for_load, get_handler_for_save


def save_artifact_to_path(
    obj: Any,
    target_path: Path,
    saver: Callable[[Any, Path], None] | None = None,
) -> None:
    """Save a Python object to a specific path with automatic format detection.

    Args:
        obj: Python object to save
        target_path: Path where artifact should be saved
        saver: Optional custom saver function (obj, path) -> None

    Raises:
        ValueError: If format can't be auto-detected and no custom saver provided
        ImportError: If required library not installed
        TypeError: If object type doesn't match expected type for extension
    """
    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if saver is not None:
        # Use custom saver
        saver(obj, target_path)
    else:
        # Auto-detect and use format handler
        handler = get_handler_for_save(obj, target_path.name)
        handler.saver(obj, target_path)


def load_artifact_from_path(
    source_path: Path,
    loader: Callable[[Path], Any] | None = None,
) -> Any:
    """Load an artifact from a specific path with automatic format detection.

    Args:
        source_path: Path to artifact to load
        loader: Optional custom loader function (path) -> object

    Returns:
        Loaded object

    Raises:
        ValueError: If format can't be auto-detected and no custom loader provided
        ImportError: If required library not installed
        FileNotFoundError: If source_path doesn't exist
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Artifact not found: {source_path}")

    if loader is not None:
        # Use custom loader
        return loader(source_path)
    else:
        # Auto-detect and use format handler
        handler = get_handler_for_load(source_path.name)
        return handler.loader(source_path)


def copy_artifact_to_path(
    source_path: Path | str,
    target_path: Path,
    filename: str | None = None,
) -> Path:
    """Copy an existing file to a target directory.

    Args:
        source_path: Path to source file
        target_path: Target directory path
        filename: Optional filename to use (defaults to source filename)

    Returns:
        Path where artifact was saved

    Raises:
        FileNotFoundError: If source doesn't exist
        ValueError: If source is not a file
    """
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Source path is not a file: {source_path}")

    # Determine target filename
    if filename is None:
        filename = source_path.name

    # Ensure target directory exists
    target_path.mkdir(parents=True, exist_ok=True)

    # Copy file
    final_path = target_path / filename
    shutil.copy2(source_path, final_path)

    return final_path


def artifact_exists_at_path(artifacts_dir: Path, filename: str) -> bool:
    """Check if an artifact exists.

    Args:
        artifacts_dir: Artifacts directory path
        filename: Name of artifact to check

    Returns:
        True if artifact exists, False otherwise
    """
    artifact_path = artifacts_dir / filename
    return artifact_path.exists() and artifact_path.is_file()


def list_artifacts_at_path(artifacts_dir: Path) -> list[str]:
    """List all artifacts in a directory.

    Args:
        artifacts_dir: Artifacts directory path

    Returns:
        List of artifact filenames (sorted)
    """
    if not artifacts_dir.exists():
        return []

    artifacts = []
    for item in artifacts_dir.iterdir():
        if item.is_file():
            artifacts.append(item.name)

    return sorted(artifacts)
