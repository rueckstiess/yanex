"""Tests for storage migrations."""

import json
from pathlib import Path

from yanex.core.migrations import (
    CURRENT_VERSION,
    Migration,
    MigrationResult,
    get_pending_migrations,
    get_storage_version,
    migrate_experiment,
    migrate_v0_to_v1,
    migrate_v1_to_v2,
    needs_migration,
)


class TestGetStorageVersion:
    """Tests for get_storage_version function."""

    def test_returns_version_when_present(self):
        metadata = {"id": "abc12345", "storage_version": 1}
        assert get_storage_version(metadata) == 1

    def test_returns_none_for_legacy(self):
        metadata = {"id": "abc12345", "status": "completed"}
        assert get_storage_version(metadata) is None

    def test_returns_none_for_empty_metadata(self):
        metadata = {}
        assert get_storage_version(metadata) is None


class TestGetPendingMigrations:
    """Tests for get_pending_migrations function."""

    def test_legacy_to_current_returns_all_migrations(self):
        pending = get_pending_migrations(None)
        assert len(pending) == 2
        assert pending[0].from_version is None
        assert pending[0].to_version == 1
        assert pending[1].from_version == 1
        assert pending[1].to_version == 2

    def test_v1_to_current_returns_remaining(self):
        pending = get_pending_migrations(1)
        assert len(pending) == 1
        assert pending[0].from_version == 1
        assert pending[0].to_version == 2

    def test_current_version_returns_empty(self):
        pending = get_pending_migrations(CURRENT_VERSION)
        assert len(pending) == 0

    def test_unknown_version_returns_empty(self):
        pending = get_pending_migrations(0)
        # Version 0 doesn't exist in our migrations, so nothing pending
        assert len(pending) == 0


class TestNeedsMigration:
    """Tests for needs_migration function."""

    def test_legacy_needs_migration(self):
        metadata = {"id": "abc12345", "status": "completed"}
        assert needs_migration(metadata) is True

    def test_current_version_no_migration(self):
        metadata = {"id": "abc12345", "storage_version": CURRENT_VERSION}
        assert needs_migration(metadata) is False


class TestMigrateV0ToV1:
    """Tests for v0 to v1 migration."""

    def test_adds_storage_version(self, tmp_path):
        """Migration adds storage_version to metadata.json."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        # Create legacy metadata (no storage_version)
        metadata = {"id": "abc12345", "status": "completed", "name": "test"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        # Run migration
        result = migrate_v0_to_v1(exp_dir, dry_run=False)

        assert result.applied is True
        assert "storage_version: 1" in result.changes[0]

        # Verify metadata was updated
        updated = json.loads((exp_dir / "metadata.json").read_text())
        assert updated["storage_version"] == 1

    def test_migrates_dependencies_format(self, tmp_path):
        """Migration converts dependency_ids list to dependencies dict."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        # Create legacy metadata
        metadata = {"id": "abc12345", "status": "completed"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        # Create old-format dependencies
        old_deps = {
            "dependency_ids": ["dep11111", "dep22222"],
            "created_at": "2025-01-15T10:00:00",
            "metadata": {},
        }
        (exp_dir / "dependencies.json").write_text(json.dumps(old_deps))

        # Run migration
        result = migrate_v0_to_v1(exp_dir, dry_run=False)

        assert result.applied is True
        assert len(result.changes) == 2  # metadata + dependencies

        # Verify dependencies were migrated
        updated_deps = json.loads((exp_dir / "dependencies.json").read_text())
        assert "dependencies" in updated_deps
        assert "dependency_ids" not in updated_deps
        assert updated_deps["dependencies"] == {
            "dep1": "dep11111",
            "dep2": "dep22222",
        }

    def test_dry_run_reports_changes_without_applying(self, tmp_path):
        """Dry run shows what would change but doesn't modify files."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        # Create legacy metadata
        metadata = {"id": "abc12345", "status": "completed"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        # Create old-format dependencies
        old_deps = {"dependency_ids": ["dep11111"], "created_at": "2025-01-15T10:00:00"}
        (exp_dir / "dependencies.json").write_text(json.dumps(old_deps))

        # Run dry-run migration
        result = migrate_v0_to_v1(exp_dir, dry_run=True)

        assert result.applied is False
        assert len(result.changes) == 2

        # Verify files unchanged
        unchanged_meta = json.loads((exp_dir / "metadata.json").read_text())
        assert "storage_version" not in unchanged_meta

        unchanged_deps = json.loads((exp_dir / "dependencies.json").read_text())
        assert "dependency_ids" in unchanged_deps
        assert "dependencies" not in unchanged_deps

    def test_idempotent_already_migrated(self, tmp_path):
        """Migration is idempotent - running on already migrated experiment is safe."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        # Create already-migrated metadata
        metadata = {"id": "abc12345", "status": "completed", "storage_version": 1}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        # Run migration
        result = migrate_v0_to_v1(exp_dir, dry_run=False)

        assert result.applied is False
        assert result.description == "Already at v1"
        assert len(result.changes) == 0

    def test_handles_experiment_without_dependencies(self, tmp_path):
        """Migration works for experiments without dependencies.json."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        # Create legacy metadata only (no dependencies.json)
        metadata = {"id": "abc12345", "status": "completed"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        # Run migration
        result = migrate_v0_to_v1(exp_dir, dry_run=False)

        assert result.applied is True
        assert len(result.changes) == 1  # Only metadata change

        # Verify metadata was updated
        updated = json.loads((exp_dir / "metadata.json").read_text())
        assert updated["storage_version"] == 1

    def test_preserves_existing_new_format_dependencies(self, tmp_path):
        """Migration doesn't overwrite dependencies already in new format."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        # Create legacy metadata
        metadata = {"id": "abc12345", "status": "completed"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        # Create new-format dependencies (already migrated manually or from newer version)
        new_deps = {
            "dependencies": {"train": "dep11111", "data": "dep22222"},
            "created_at": "2025-01-15T10:00:00",
        }
        (exp_dir / "dependencies.json").write_text(json.dumps(new_deps))

        # Run migration
        result = migrate_v0_to_v1(exp_dir, dry_run=False)

        assert result.applied is True
        # Only metadata change, not dependencies
        assert len(result.changes) == 1

        # Verify dependencies unchanged
        unchanged_deps = json.loads((exp_dir / "dependencies.json").read_text())
        assert unchanged_deps["dependencies"] == {
            "train": "dep11111",
            "data": "dep22222",
        }


class TestMigrateExperiment:
    """Tests for migrate_experiment function."""

    def test_applies_all_pending_migrations(self, tmp_path):
        """migrate_experiment applies all pending migrations."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        # Create legacy metadata
        metadata = {"id": "abc12345", "status": "completed"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        # Run migration
        results = migrate_experiment(exp_dir, dry_run=False)

        assert len(results) >= 1
        assert results[0].applied is True

        # Verify final state
        updated = json.loads((exp_dir / "metadata.json").read_text())
        assert updated["storage_version"] == CURRENT_VERSION

    def test_dry_run_returns_pending_changes(self, tmp_path):
        """migrate_experiment dry run returns what would change."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        metadata = {"id": "abc12345", "status": "completed"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        results = migrate_experiment(exp_dir, dry_run=True)

        assert len(results) >= 1
        assert results[0].applied is False
        assert len(results[0].changes) > 0

        # Verify no changes made
        unchanged = json.loads((exp_dir / "metadata.json").read_text())
        assert "storage_version" not in unchanged

    def test_skips_already_current(self, tmp_path):
        """migrate_experiment returns empty for current version."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        metadata = {
            "id": "abc12345",
            "status": "completed",
            "storage_version": CURRENT_VERSION,
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        results = migrate_experiment(exp_dir, dry_run=False)

        assert len(results) == 0


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_defaults(self):
        result = MigrationResult(applied=True, description="test")
        assert result.applied is True
        assert result.description == "test"
        assert result.changes == []

    def test_with_changes(self):
        result = MigrationResult(
            applied=True, description="test", changes=["change1", "change2"]
        )
        assert len(result.changes) == 2


class TestMigration:
    """Tests for Migration dataclass."""

    def test_creation(self):
        def dummy_fn(exp_dir: Path, dry_run: bool) -> MigrationResult:
            return MigrationResult(applied=True, description="test")

        migration = Migration(
            from_version=None,
            to_version=1,
            description="Test migration",
            migrate_fn=dummy_fn,
        )

        assert migration.from_version is None
        assert migration.to_version == 1


class TestMigrateV1ToV2:
    """Tests for v1 to v2 migration (add project field)."""

    def test_adds_project_from_git_repo_path(self, tmp_path):
        """Migration derives project from git repo path in metadata."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        metadata = {
            "id": "abc12345",
            "status": "completed",
            "storage_version": 1,
            "environment": {
                "git": {
                    "repository": {
                        "repo_path": "/Users/thomas/code/myproject",
                        "branch": "main",
                    }
                }
            },
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        result = migrate_v1_to_v2(exp_dir, dry_run=False)

        assert result.applied is True
        assert any("project: myproject" in c for c in result.changes)

        updated = json.loads((exp_dir / "metadata.json").read_text())
        assert updated["project"] == "myproject"
        assert updated["storage_version"] == 2

    def test_sets_null_project_when_no_git_info(self, tmp_path):
        """Migration sets project to null when no git repo path exists."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        metadata = {
            "id": "abc12345",
            "status": "completed",
            "storage_version": 1,
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        result = migrate_v1_to_v2(exp_dir, dry_run=False)

        assert result.applied is True
        assert any("null" in c for c in result.changes)

        updated = json.loads((exp_dir / "metadata.json").read_text())
        assert updated["project"] is None
        assert updated["storage_version"] == 2

    def test_dry_run_reports_without_applying(self, tmp_path):
        """Dry run shows changes without modifying files."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        metadata = {
            "id": "abc12345",
            "status": "completed",
            "storage_version": 1,
            "environment": {"git": {"repository": {"repo_path": "/path/to/repo"}}},
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        result = migrate_v1_to_v2(exp_dir, dry_run=True)

        assert result.applied is False
        assert len(result.changes) == 2  # project + version bump

        unchanged = json.loads((exp_dir / "metadata.json").read_text())
        assert "project" not in unchanged
        assert unchanged["storage_version"] == 1

    def test_idempotent_already_at_v2(self, tmp_path):
        """Migration is idempotent for experiments already at v2."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        metadata = {
            "id": "abc12345",
            "status": "completed",
            "storage_version": 2,
            "project": "existing-project",
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        result = migrate_v1_to_v2(exp_dir, dry_run=False)

        assert result.applied is False
        assert result.description == "Already at v2"

    def test_chained_migration_v0_to_v2(self, tmp_path):
        """Full migration chain from legacy (no version) to v2."""
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()

        metadata = {
            "id": "abc12345",
            "status": "completed",
            "environment": {"git": {"repository": {"repo_path": "/path/to/myproject"}}},
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        results = migrate_experiment(exp_dir, dry_run=False)

        assert len(results) == 2  # v0→v1 + v1→v2
        assert results[0].applied is True
        assert results[1].applied is True

        final = json.loads((exp_dir / "metadata.json").read_text())
        assert final["storage_version"] == 2
        assert final["project"] == "myproject"
