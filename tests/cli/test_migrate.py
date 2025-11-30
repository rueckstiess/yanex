"""Tests for yanex CLI migrate command."""

import json
from pathlib import Path

from tests.test_utils import TestDataFactory, create_cli_runner
from yanex.cli.main import cli
from yanex.core.migrations import CURRENT_VERSION
from yanex.core.storage import ExperimentStorage


class TestMigrateCommandHelp:
    """Tests for migrate command help and basic validation."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = create_cli_runner()

    def test_help_displays(self):
        """Test that help text displays correctly."""
        result = self.runner.invoke(cli, ["migrate", "--help"])
        assert result.exit_code == 0
        assert "Migrate experiments" in result.output
        assert "--dry-run" in result.output
        assert "--all" in result.output

    def test_requires_target_specification(self, per_test_experiments_dir):
        """Test that command requires --all, identifiers, or filters."""
        result = self.runner.invoke(cli, ["migrate"])
        assert result.exit_code != 0
        assert "Must specify --all" in result.output

    def test_cannot_combine_all_with_identifiers(self, per_test_experiments_dir):
        """Test that --all cannot be combined with identifiers."""
        result = self.runner.invoke(cli, ["migrate", "--all", "abc12345"])
        assert result.exit_code != 0
        assert "Cannot combine --all with experiment identifiers" in result.output


class TestMigrateCommand:
    """Tests for migrate command functionality."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = create_cli_runner()

    def _create_legacy_experiment(
        self,
        experiments_dir: Path,
        exp_id: str,
        name: str | None = None,
        with_deps: list | None = None,
    ) -> str:
        """Create a legacy experiment (no storage_version)."""
        storage = ExperimentStorage(experiments_dir)

        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id,
            name=name,
            status="completed",
        )
        # Remove storage_version if TestDataFactory adds it
        metadata.pop("storage_version", None)

        storage.create_experiment_directory(exp_id)
        storage.save_metadata(exp_id, metadata)

        # Add old-format dependencies if specified
        if with_deps:
            exp_dir = experiments_dir / exp_id
            old_deps = {
                "dependency_ids": with_deps,
                "created_at": "2025-01-15T10:00:00",
                "metadata": {},
            }
            (exp_dir / "dependencies.json").write_text(json.dumps(old_deps))

        return exp_id

    def _create_current_experiment(
        self, experiments_dir: Path, exp_id: str, name: str | None = None
    ) -> str:
        """Create a current-version experiment."""
        storage = ExperimentStorage(experiments_dir)

        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id,
            name=name,
            status="completed",
        )
        metadata["storage_version"] = CURRENT_VERSION

        storage.create_experiment_directory(exp_id)
        storage.save_metadata(exp_id, metadata)

        return exp_id

    def test_dry_run_shows_pending_migrations(self, per_test_experiments_dir):
        """Test --dry-run shows what would be migrated without changes."""
        # Create legacy experiment
        self._create_legacy_experiment(
            per_test_experiments_dir, "abc12345", "legacy-exp"
        )

        result = self.runner.invoke(cli, ["migrate", "--all", "--dry-run"])

        assert result.exit_code == 0
        assert "abc12345" in result.output
        assert "needs migration" in result.output
        assert "Dry run completed" in result.output
        assert "No changes were made" in result.output

        # Verify no changes were made
        exp_dir = per_test_experiments_dir / "abc12345"
        metadata = json.loads((exp_dir / "metadata.json").read_text())
        assert "storage_version" not in metadata

    def test_dry_run_shows_up_to_date_experiments(self, per_test_experiments_dir):
        """Test --dry-run shows experiments already up to date."""
        self._create_current_experiment(
            per_test_experiments_dir, "abc12345", "current-exp"
        )

        result = self.runner.invoke(cli, ["migrate", "--all", "--dry-run"])

        assert result.exit_code == 0
        assert "up to date" in result.output

    def test_migrate_all_force(self, per_test_experiments_dir):
        """Test --all --force migrates experiments without confirmation."""
        self._create_legacy_experiment(per_test_experiments_dir, "abc12345", "legacy-1")
        self._create_legacy_experiment(per_test_experiments_dir, "def67890", "legacy-2")

        result = self.runner.invoke(cli, ["migrate", "--all", "--force"])

        assert result.exit_code == 0
        assert "abc12345" in result.output or "legacy-1" in result.output
        assert "def67890" in result.output or "legacy-2" in result.output

        # Verify migrations were applied
        for exp_id in ["abc12345", "def67890"]:
            exp_dir = per_test_experiments_dir / exp_id
            metadata = json.loads((exp_dir / "metadata.json").read_text())
            assert metadata["storage_version"] == CURRENT_VERSION

    def test_migrate_specific_experiment(self, per_test_experiments_dir):
        """Test migrating specific experiment by ID."""
        self._create_legacy_experiment(
            per_test_experiments_dir, "abc12345", "legacy-exp"
        )
        self._create_legacy_experiment(
            per_test_experiments_dir, "def67890", "other-exp"
        )

        result = self.runner.invoke(cli, ["migrate", "abc12345", "--force"])

        assert result.exit_code == 0

        # Verify only specified experiment was migrated
        abc_meta = json.loads(
            (per_test_experiments_dir / "abc12345" / "metadata.json").read_text()
        )
        assert abc_meta["storage_version"] == CURRENT_VERSION

        def_meta = json.loads(
            (per_test_experiments_dir / "def67890" / "metadata.json").read_text()
        )
        assert "storage_version" not in def_meta

    def test_migrate_converts_dependencies_format(self, per_test_experiments_dir):
        """Test migration converts old dependency format to new."""
        # Create experiment with old-format dependencies
        self._create_legacy_experiment(
            per_test_experiments_dir,
            "abc12345",
            "with-deps",
            with_deps=["dep11111", "dep22222"],
        )

        result = self.runner.invoke(cli, ["migrate", "abc12345", "--force"])

        assert result.exit_code == 0

        # Verify dependencies were converted
        exp_dir = per_test_experiments_dir / "abc12345"
        deps = json.loads((exp_dir / "dependencies.json").read_text())

        assert "dependencies" in deps
        assert "dependency_ids" not in deps
        assert deps["dependencies"] == {"dep1": "dep11111", "dep2": "dep22222"}

    def test_dry_run_shows_dependency_changes(self, per_test_experiments_dir):
        """Test --dry-run shows dependency format changes."""
        self._create_legacy_experiment(
            per_test_experiments_dir, "abc12345", "with-deps", with_deps=["dep11111"]
        )

        result = self.runner.invoke(cli, ["migrate", "--all", "--dry-run"])

        assert result.exit_code == 0
        assert "dependency_ids" in result.output
        assert "dependencies" in result.output

        # Verify no changes made
        exp_dir = per_test_experiments_dir / "abc12345"
        deps = json.loads((exp_dir / "dependencies.json").read_text())
        assert "dependency_ids" in deps
        assert "dependencies" not in deps

    def test_all_up_to_date_shows_message(self, per_test_experiments_dir):
        """Test message when all experiments are already up to date."""
        self._create_current_experiment(
            per_test_experiments_dir, "abc12345", "current-exp"
        )

        result = self.runner.invoke(cli, ["migrate", "--all", "--force"])

        assert result.exit_code == 0
        assert "up to date" in result.output

    def test_no_experiments_found(self, per_test_experiments_dir):
        """Test message when no experiments found."""
        result = self.runner.invoke(cli, ["migrate", "--all"])

        assert result.exit_code == 0
        assert "No experiments found" in result.output

    def test_migrate_by_status_filter(self, per_test_experiments_dir):
        """Test migrating experiments by status filter."""
        storage = ExperimentStorage(per_test_experiments_dir)

        # Create completed experiment
        self._create_legacy_experiment(
            per_test_experiments_dir, "abc12345", "completed-exp"
        )

        # Create a failed experiment
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id="def67890",
            name="failed-exp",
            status="failed",
        )
        metadata.pop("storage_version", None)
        storage.create_experiment_directory("def67890")
        storage.save_metadata("def67890", metadata)

        result = self.runner.invoke(cli, ["migrate", "-s", "completed", "--force"])

        assert result.exit_code == 0

        # Only completed experiment should be migrated
        abc_meta = json.loads(
            (per_test_experiments_dir / "abc12345" / "metadata.json").read_text()
        )
        assert abc_meta["storage_version"] == CURRENT_VERSION

        def_meta = json.loads(
            (per_test_experiments_dir / "def67890" / "metadata.json").read_text()
        )
        assert "storage_version" not in def_meta

    def test_migrate_archived_experiment_by_id(self, per_test_experiments_dir):
        """Test migrating archived experiment by ID works without --archived flag."""
        storage = ExperimentStorage(per_test_experiments_dir)

        # Create a legacy experiment
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id="abc12345",
            name="archived-exp",
            status="completed",
        )
        metadata.pop("storage_version", None)
        storage.create_experiment_directory("abc12345")
        storage.save_metadata("abc12345", metadata)

        # Archive the experiment
        storage.archive_experiment("abc12345")

        # Migrate by ID should find it without needing --archived flag
        result = self.runner.invoke(cli, ["migrate", "abc12345", "--force"])

        assert result.exit_code == 0
        assert "abc12345" in result.output

        # Verify migration was applied in archived location
        archived_dir = per_test_experiments_dir / "archived" / "abc12345"
        migrated_meta = json.loads((archived_dir / "metadata.json").read_text())
        assert migrated_meta["storage_version"] == CURRENT_VERSION


class TestMigrateCommandOutput:
    """Tests for migrate command output formatting."""

    def setup_method(self):
        """Set up CLI runner."""
        self.runner = create_cli_runner()

    def test_summary_shows_counts(self, per_test_experiments_dir):
        """Test that summary shows correct counts."""
        storage = ExperimentStorage(per_test_experiments_dir)

        # Create mix of legacy and current experiments
        for i in range(3):
            metadata = TestDataFactory.create_experiment_metadata(
                experiment_id=f"legacy{i:02d}",
                name=f"legacy-{i}",
                status="completed",
            )
            metadata.pop("storage_version", None)
            storage.create_experiment_directory(f"legacy{i:02d}")
            storage.save_metadata(f"legacy{i:02d}", metadata)

        for i in range(2):
            metadata = TestDataFactory.create_experiment_metadata(
                experiment_id=f"current{i:01d}",
                name=f"current-{i}",
                status="completed",
            )
            metadata["storage_version"] = CURRENT_VERSION
            storage.create_experiment_directory(f"current{i:01d}")
            storage.save_metadata(f"current{i:01d}", metadata)

        result = self.runner.invoke(cli, ["migrate", "--all", "--dry-run"])

        assert result.exit_code == 0
        assert "3" in result.output  # 3 need migration
        assert "2" in result.output  # 2 up to date
