"""Unified artifact save/load logic with automatic format detection."""

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifact_formats import get_handler_for_load, get_handler_for_save

# Default maximum file size for artifacts (1000MB)
DEFAULT_MAX_FILE_SIZE = 1000 * 1024 * 1024


def _validate_filename(filename: str) -> str:
    """Validate and sanitize filename to prevent path traversal attacks.

    Args:
        filename: Filename to validate

    Returns:
        Validated filename (basename only)

    Raises:
        ValueError: If filename contains path traversal attempts
    """
    # Check for path traversal attempts
    if ".." in filename:
        raise ValueError(
            f"Invalid filename '{filename}': contains '..' (path traversal not allowed)"
        )

    # Check for absolute paths
    if filename.startswith("/") or (len(filename) > 1 and filename[1] == ":"):
        raise ValueError(f"Invalid filename '{filename}': absolute paths not allowed")

    # Extract basename to ensure we only get the filename
    basename = os.path.basename(filename)

    # Ensure we got a valid filename
    if not basename or basename in (".", ".."):
        raise ValueError(f"Invalid filename '{filename}': must be a valid filename")

    return basename


def _validate_file_size(path: Path, max_size: int | None = None) -> None:
    """Validate that file size is within acceptable limits.

    Args:
        path: Path to file to check
        max_size: Maximum file size in bytes (None for no limit)

    Raises:
        ValueError: If file exceeds size limit
    """
    if max_size is None:
        max_size = DEFAULT_MAX_FILE_SIZE

    if max_size <= 0:
        return  # No limit

    file_size = path.stat().st_size
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        limit_mb = max_size / (1024 * 1024)
        raise ValueError(
            f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({limit_mb:.1f}MB)"
        )


def save_artifact_to_path(
    obj: Any,
    target_path: Path,
    saver: Callable[..., None] | None = None,
    max_size: int | None = None,
    **kwargs: Any,
) -> None:
    """Save a Python object to a specific path with automatic format detection.

    Args:
        obj: Python object to save
        target_path: Path where artifact should be saved
        saver: Optional custom saver function (obj, path, **kwargs) -> None
        max_size: Maximum file size in bytes (None uses default 100MB, 0 for no limit)
        **kwargs: Additional arguments passed to the underlying save function.
            Common examples:
            - dpi, bbox_inches, transparent (matplotlib figures)
            - indent, ensure_ascii, sort_keys (JSON)
            - index, sep (pandas CSV)
            - protocol (pickle)
            - compressed (numpy .npz)

    Raises:
        ValueError: If format can't be auto-detected and no custom saver provided,
                   or if filename is invalid, or if file size exceeds limit
        ImportError: If required library not installed
        TypeError: If object type doesn't match expected type for extension
    """
    # Validate filename to prevent path traversal
    validated_filename = _validate_filename(target_path.name)
    if validated_filename != target_path.name:
        # Filename was modified during validation, update path
        target_path = target_path.parent / validated_filename

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if saver is not None:
        # Use custom saver - pass kwargs if it accepts them
        saver(obj, target_path, **kwargs)
    else:
        # Auto-detect and use format handler
        handler = get_handler_for_save(obj, target_path.name)
        handler.saver(obj, target_path, **kwargs)

    # Validate file size after saving
    if target_path.exists():
        _validate_file_size(target_path, max_size)


def load_artifact_from_path(
    source_path: Path,
    loader: Callable[[Path], Any] | None = None,
    format: str | None = None,
) -> Any:
    """Load an artifact from a specific path with automatic format detection.

    Args:
        source_path: Path to artifact to load
        loader: Optional custom loader function (path) -> object
        format: Optional format name for explicit format selection

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
        handler = get_handler_for_load(source_path.name, format=format)
        return handler.loader(source_path)


def copy_artifact_to_path(
    source_path: Path | str,
    target_path: Path,
    filename: str | None = None,
    max_size: int | None = None,
) -> Path:
    """Copy an existing file to a target directory.

    Args:
        source_path: Path to source file
        target_path: Target directory path
        filename: Optional filename to use (defaults to source filename)
        max_size: Maximum file size in bytes (None uses default 100MB, 0 for no limit)

    Returns:
        Path where artifact was saved

    Raises:
        FileNotFoundError: If source doesn't exist
        ValueError: If source is not a file, filename is invalid, or file size exceeds limit
    """
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Source path is not a file: {source_path}")

    # Validate file size before copying
    _validate_file_size(source_path, max_size)

    # Determine target filename
    if filename is None:
        filename = source_path.name

    # Validate filename to prevent path traversal
    filename = _validate_filename(filename)

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

    Raises:
        ValueError: If filename is invalid
    """
    # Validate filename to prevent path traversal
    filename = _validate_filename(filename)

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
