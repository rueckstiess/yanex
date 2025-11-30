"""Storage migrations for yanex experiments."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CURRENT_VERSION = 1


@dataclass
class MigrationResult:
    """Result of a single migration."""

    applied: bool  # True if migration was applied
    description: str  # Human-readable description of what changed
    changes: list[str] = field(default_factory=list)  # List of specific changes made


@dataclass
class Migration:
    """A single migration definition."""

    from_version: int | None  # None = legacy (no version)
    to_version: int
    description: str  # User-facing description
    migrate_fn: Callable[[Path, bool], MigrationResult]


def get_storage_version(metadata: dict[str, Any]) -> int | None:
    """Extract storage version from metadata.

    Args:
        metadata: Experiment metadata dict

    Returns:
        Storage version as int, or None for legacy experiments
    """
    return metadata.get("storage_version")


def get_pending_migrations(current_version: int | None) -> list["Migration"]:
    """Get list of migrations needed to reach CURRENT_VERSION.

    Args:
        current_version: Current storage version (None for legacy)

    Returns:
        List of Migration objects to apply in order
    """
    pending = []
    version = current_version

    for migration in MIGRATIONS:
        if version is None and migration.from_version is None:
            pending.append(migration)
            version = migration.to_version
        elif version is not None and migration.from_version == version:
            pending.append(migration)
            version = migration.to_version

    return pending


def migrate_v0_to_v1(exp_dir: Path, dry_run: bool) -> MigrationResult:
    """Migrate from v0 (no version) to v1.

    Changes:
    - Add storage_version: 1 to metadata.json
    - Convert dependencies.json from {"dependency_ids": [...]}
      to {"dependencies": {"dep1": id1, "dep2": id2, ...}}

    Args:
        exp_dir: Path to experiment directory
        dry_run: If True, report changes without applying

    Returns:
        MigrationResult with details of changes
    """
    changes = []
    metadata_path = exp_dir / "metadata.json"
    dependencies_path = exp_dir / "dependencies.json"

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Check if already migrated (idempotent)
    if metadata.get("storage_version") == 1:
        return MigrationResult(applied=False, description="Already at v1", changes=[])

    # Change 1: Add storage_version to metadata
    changes.append("metadata.json: Add storage_version: 1")

    # Change 2: Migrate dependencies if old format exists
    dep_data = None
    if dependencies_path.exists():
        with open(dependencies_path) as f:
            dep_data = json.load(f)

        # Check for old format (dependency_ids list)
        if "dependency_ids" in dep_data and "dependencies" not in dep_data:
            old_ids = dep_data["dependency_ids"]
            # Convert to new format with auto-generated slot names
            new_deps = {f"dep{i + 1}": dep_id for i, dep_id in enumerate(old_ids)}
            changes.append(
                f"dependencies.json: Convert dependency_ids {old_ids} -> "
                f"dependencies {new_deps}"
            )

            if not dry_run:
                # Update dependencies.json
                dep_data["dependencies"] = new_deps
                del dep_data["dependency_ids"]
                with open(dependencies_path, "w") as f:
                    json.dump(dep_data, f, indent=2)

    if not dry_run:
        # Update metadata.json
        metadata["storage_version"] = 1
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return MigrationResult(
        applied=not dry_run,
        description="Migrate to storage version 1 (named dependency slots)",
        changes=changes,
    )


# Migration registry - ordered list
MIGRATIONS: list[Migration] = [
    Migration(
        from_version=None,
        to_version=1,
        description="Add storage versioning and convert dependencies to named slots",
        migrate_fn=migrate_v0_to_v1,
    ),
]


def migrate_experiment(exp_dir: Path, dry_run: bool = False) -> list[MigrationResult]:
    """Apply all pending migrations to an experiment.

    Args:
        exp_dir: Path to experiment directory
        dry_run: If True, report changes without applying

    Returns:
        List of MigrationResult for each migration applied/checked
    """
    metadata_path = exp_dir / "metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    current_version = get_storage_version(metadata)
    pending = get_pending_migrations(current_version)

    results = []
    for migration in pending:
        result = migration.migrate_fn(exp_dir, dry_run)
        results.append(result)

    return results


def needs_migration(metadata: dict[str, Any]) -> bool:
    """Check if experiment needs migration.

    Args:
        metadata: Experiment metadata dict

    Returns:
        True if migrations are pending
    """
    current_version = get_storage_version(metadata)
    return len(get_pending_migrations(current_version)) > 0
