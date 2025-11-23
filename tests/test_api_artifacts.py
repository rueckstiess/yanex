"""Tests for artifact API functions (copy_artifact, save_artifact, load_artifact)."""

import json
import os
from pathlib import Path

import pytest

import yanex
from yanex.utils.exceptions import StorageError


@pytest.fixture(autouse=True)
def clear_cli_env():
    """Clear YANEX_CLI_ACTIVE environment variable before each test."""
    os.environ.pop("YANEX_CLI_ACTIVE", None)
    yield
    os.environ.pop("YANEX_CLI_ACTIVE", None)


class TestCopyArtifact:
    """Test copy_artifact() function."""

    def test_copy_artifact_with_tracking(self, clean_git_repo, tmp_path):
        """Test copy_artifact with experiment tracking."""
        # Create a source file
        source_file = tmp_path / "source.txt"
        source_file.write_text("Hello, world!")

        # Create experiment
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Copy artifact
            yanex.copy_artifact(source_file)

            # Verify artifact was copied
            artifacts_dir = yanex.get_artifacts_dir()
            copied_file = artifacts_dir / "source.txt"
            assert copied_file.exists()
            assert copied_file.read_text() == "Hello, world!"

    def test_copy_artifact_with_custom_name(self, clean_git_repo, tmp_path):
        """Test copy_artifact with custom filename."""
        source_file = tmp_path / "original.txt"
        source_file.write_text("Content")

        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Copy with custom name
            yanex.copy_artifact(source_file, "renamed.txt")

            # Verify artifact was copied with new name
            artifacts_dir = yanex.get_artifacts_dir()
            copied_file = artifacts_dir / "renamed.txt"
            assert copied_file.exists()
            assert copied_file.read_text() == "Content"

    def test_copy_artifact_standalone_mode(self, tmp_path, monkeypatch):
        """Test copy_artifact in standalone mode (no tracking)."""
        monkeypatch.chdir(tmp_path)

        # Create a source file
        source_file = tmp_path / "source.txt"
        source_file.write_text("Standalone content")

        # Copy artifact in standalone mode
        yanex.copy_artifact(source_file)

        # Verify artifact was copied to ./artifacts/
        artifacts_dir = tmp_path / "artifacts"
        assert artifacts_dir.exists()
        copied_file = artifacts_dir / "source.txt"
        assert copied_file.exists()
        assert copied_file.read_text() == "Standalone content"

    def test_copy_artifact_file_not_found(self, clean_git_repo, tmp_path):
        """Test copy_artifact with nonexistent file."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            with pytest.raises((FileNotFoundError, StorageError)):
                yanex.copy_artifact(tmp_path / "nonexistent.txt")

    def test_copy_artifact_not_a_file(self, clean_git_repo, tmp_path):
        """Test copy_artifact with directory instead of file."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            with pytest.raises((ValueError, StorageError)):
                yanex.copy_artifact(tmp_path)


class TestSaveArtifact:
    """Test save_artifact() function."""

    def test_save_text_artifact(self, clean_git_repo, tmp_path):
        """Test saving text artifact."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Save text artifact
            yanex.save_artifact("Hello, world!", "message.txt")

            # Verify artifact was saved
            artifacts_dir = yanex.get_artifacts_dir()
            artifact_file = artifacts_dir / "message.txt"
            assert artifact_file.exists()
            assert artifact_file.read_text() == "Hello, world!"

    def test_save_json_artifact(self, clean_git_repo, tmp_path):
        """Test saving JSON artifact."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            data = {"key": "value", "number": 42}
            yanex.save_artifact(data, "data.json")

            # Verify artifact was saved
            artifacts_dir = yanex.get_artifacts_dir()
            artifact_file = artifacts_dir / "data.json"
            assert artifact_file.exists()

            loaded = json.loads(artifact_file.read_text())
            assert loaded == data

    def test_save_pickle_artifact(self, clean_git_repo, tmp_path):
        """Test saving pickle artifact."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            data = {"key": "value", "numbers": [1, 2, 3]}
            yanex.save_artifact(data, "data.pkl")

            # Verify artifact was saved
            artifacts_dir = yanex.get_artifacts_dir()
            artifact_file = artifacts_dir / "data.pkl"
            assert artifact_file.exists()

    def test_save_artifact_standalone_mode(self, tmp_path, monkeypatch):
        """Test save_artifact in standalone mode."""
        monkeypatch.chdir(tmp_path)

        # Save artifact in standalone mode
        yanex.save_artifact({"key": "value"}, "data.json")

        # Verify artifact was saved to ./artifacts/
        artifacts_dir = tmp_path / "artifacts"
        assert artifacts_dir.exists()
        artifact_file = artifacts_dir / "data.json"
        assert artifact_file.exists()

        loaded = json.loads(artifact_file.read_text())
        assert loaded == {"key": "value"}

    def test_save_artifact_custom_saver(self, clean_git_repo, tmp_path):
        """Test save_artifact with custom saver function."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        def custom_saver(obj, path):
            """Custom saver that saves as uppercase text."""
            path.write_text(obj.upper())

        with yanex.create_experiment(script_path):
            yanex.save_artifact("hello", "custom.dat", saver=custom_saver)

            # Verify custom saver was used
            artifacts_dir = yanex.get_artifacts_dir()
            artifact_file = artifacts_dir / "custom.dat"
            assert artifact_file.read_text() == "HELLO"

    def test_save_artifact_unsupported_extension(self, clean_git_repo, tmp_path):
        """Test save_artifact with unsupported extension."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            with pytest.raises((ValueError, StorageError)):
                yanex.save_artifact("data", "file.unknown")

    def test_save_artifact_wrong_type(self, clean_git_repo, tmp_path):
        """Test save_artifact with wrong object type for extension."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Try to save dict as .txt (expects string)
            with pytest.raises((TypeError, StorageError)):
                yanex.save_artifact({"key": "value"}, "file.txt")

    def test_save_artifact_overwrite(self, clean_git_repo, tmp_path):
        """Test that save_artifact overwrites existing file."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Save first version
            yanex.save_artifact("Version 1", "file.txt")

            # Overwrite with second version
            yanex.save_artifact("Version 2", "file.txt")

            # Verify second version is saved
            artifacts_dir = yanex.get_artifacts_dir()
            artifact_file = artifacts_dir / "file.txt"
            assert artifact_file.read_text() == "Version 2"


class TestLoadArtifact:
    """Test load_artifact() function."""

    def test_load_artifact_with_tracking(self, clean_git_repo, tmp_path):
        """Test load_artifact with experiment tracking."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Save then load
            yanex.save_artifact("Hello, world!", "message.txt")
            loaded = yanex.load_artifact("message.txt")
            assert loaded == "Hello, world!"

    def test_load_json_artifact(self, clean_git_repo, tmp_path):
        """Test loading JSON artifact."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            data = {"key": "value", "number": 42}
            yanex.save_artifact(data, "data.json")

            loaded = yanex.load_artifact("data.json")
            assert loaded == data

    def test_load_artifact_not_found(self, clean_git_repo, tmp_path):
        """Test load_artifact returns None for missing artifact."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            loaded = yanex.load_artifact("nonexistent.txt")
            assert loaded is None

    def test_load_artifact_standalone_mode(self, tmp_path, monkeypatch):
        """Test load_artifact in standalone mode."""
        monkeypatch.chdir(tmp_path)

        # Save in standalone mode
        yanex.save_artifact({"key": "value"}, "data.json")

        # Load in standalone mode
        loaded = yanex.load_artifact("data.json")
        assert loaded == {"key": "value"}

    def test_load_artifact_custom_loader(self, clean_git_repo, tmp_path):
        """Test load_artifact with custom loader function."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        def custom_loader(path):
            """Custom loader that returns uppercase text."""
            return path.read_text().upper()

        with yanex.create_experiment(script_path):
            yanex.save_artifact("hello", "message.txt")

            loaded = yanex.load_artifact("message.txt", loader=custom_loader)
            assert loaded == "HELLO"

    def test_load_artifact_missing_in_standalone(self, tmp_path, monkeypatch):
        """Test load_artifact returns None for missing artifact in standalone mode."""
        monkeypatch.chdir(tmp_path)

        loaded = yanex.load_artifact("nonexistent.txt")
        assert loaded is None


class TestArtifactExists:
    """Test artifact_exists() function."""

    def test_artifact_exists_with_tracking(self, clean_git_repo, tmp_path):
        """Test artifact_exists with experiment tracking."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Check nonexistent artifact
            assert not yanex.artifact_exists("missing.txt")

            # Save artifact
            yanex.save_artifact("content", "file.txt")

            # Check existing artifact
            assert yanex.artifact_exists("file.txt")

    def test_artifact_exists_standalone_mode(self, tmp_path, monkeypatch):
        """Test artifact_exists in standalone mode."""
        monkeypatch.chdir(tmp_path)

        # Check nonexistent artifact
        assert not yanex.artifact_exists("missing.txt")

        # Save artifact
        yanex.save_artifact("content", "file.txt")

        # Check existing artifact
        assert yanex.artifact_exists("file.txt")


class TestListArtifacts:
    """Test list_artifacts() function."""

    def test_list_artifacts_with_tracking(self, clean_git_repo, tmp_path):
        """Test list_artifacts with experiment tracking."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Save artifacts
            yanex.save_artifact("content1", "file1.txt")
            yanex.save_artifact("content2", "file2.txt")
            yanex.save_artifact({"key": "value"}, "data.json")

            # List artifacts (filter out git_diff.patch which is auto-created)
            artifacts = [a for a in yanex.list_artifacts() if a != "git_diff.patch"]
            assert sorted(artifacts) == ["data.json", "file1.txt", "file2.txt"]

    def test_list_artifacts_standalone_mode(self, tmp_path, monkeypatch):
        """Test list_artifacts in standalone mode."""
        monkeypatch.chdir(tmp_path)

        # Initially empty
        assert yanex.list_artifacts() == []

        # Save artifacts
        yanex.save_artifact("content1", "file1.txt")
        yanex.save_artifact("content2", "file2.txt")

        # List artifacts
        artifacts = yanex.list_artifacts()
        assert sorted(artifacts) == ["file1.txt", "file2.txt"]


class TestArtifactIntegration:
    """Integration tests for artifact API."""

    def test_save_load_roundtrip(self, clean_git_repo, tmp_path):
        """Test saving and loading various formats."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Text
            yanex.save_artifact("Text content", "text.txt")
            assert yanex.load_artifact("text.txt") == "Text content"

            # JSON
            json_data = {"key": "value", "numbers": [1, 2, 3]}
            yanex.save_artifact(json_data, "data.json")
            assert yanex.load_artifact("data.json") == json_data

            # Pickle
            pickle_data = {"set": {1, 2, 3}, "tuple": (1, 2)}
            yanex.save_artifact(pickle_data, "data.pkl")
            loaded_pickle = yanex.load_artifact("data.pkl")
            assert loaded_pickle["set"] == pickle_data["set"]
            assert loaded_pickle["tuple"] == pickle_data["tuple"]

    def test_optional_artifact_pattern(self, clean_git_repo, tmp_path):
        """Test optional artifact loading pattern."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Try to load optional artifact
            checkpoint = yanex.load_artifact("checkpoint.pkl")
            assert checkpoint is None

            # Save checkpoint
            yanex.save_artifact({"epoch": 5}, "checkpoint.pkl")

            # Now it exists
            checkpoint = yanex.load_artifact("checkpoint.pkl")
            assert checkpoint == {"epoch": 5}

    def test_artifact_workflow(self, clean_git_repo, tmp_path):
        """Test complete artifact workflow."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Save multiple artifacts
            yanex.save_artifact("Log content", "log.txt")
            yanex.save_artifact({"accuracy": 0.95}, "metrics.json")
            yanex.save_artifact([1, 2, 3], "results.pkl")

            # Check they exist
            assert yanex.artifact_exists("log.txt")
            assert yanex.artifact_exists("metrics.json")
            assert yanex.artifact_exists("results.pkl")

            # List all artifacts (filter out git_diff.patch)
            artifacts = [a for a in yanex.list_artifacts() if a != "git_diff.patch"]
            assert sorted(artifacts) == ["log.txt", "metrics.json", "results.pkl"]

            # Load them back
            log = yanex.load_artifact("log.txt")
            metrics = yanex.load_artifact("metrics.json")
            results = yanex.load_artifact("results.pkl")

            assert log == "Log content"
            assert metrics == {"accuracy": 0.95}
            assert results == [1, 2, 3]

    def test_standalone_to_tracking_workflow(
        self, clean_git_repo, tmp_path, monkeypatch
    ):
        """Test using artifacts in both standalone and tracking mode."""
        # Use clean_git_repo working directory
        git_dir = Path(clean_git_repo.working_dir)
        monkeypatch.chdir(git_dir)

        # Start in standalone mode
        yanex.save_artifact("Standalone data", "data.txt")
        assert yanex.load_artifact("data.txt") == "Standalone data"

        # Switch to tracking mode
        script_path = git_dir / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Save in tracking mode
            yanex.save_artifact("Tracking data", "experiment_data.txt")

            # Verify isolated from standalone artifacts
            assert yanex.artifact_exists("experiment_data.txt")
            assert not yanex.artifact_exists("data.txt")  # Standalone artifact

            # List only shows tracking artifacts (filter out git_diff.patch)
            artifacts = [a for a in yanex.list_artifacts() if a != "git_diff.patch"]
            assert artifacts == ["experiment_data.txt"]


class TestArtifactSecurity:
    """Test security features of artifact API."""

    def test_save_artifact_path_traversal_blocked(self, clean_git_repo, tmp_path):
        """Test that path traversal attempts are blocked in save_artifact."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Try path traversal with .. (use valid extension)
            with pytest.raises((ValueError, StorageError), match="path traversal"):
                yanex.save_artifact("malicious", "../../../etc/passwd.txt")

            # Try absolute path (use valid extension)
            with pytest.raises((ValueError, StorageError), match="absolute paths not allowed"):
                yanex.save_artifact("malicious", "/etc/passwd.txt")

    def test_copy_artifact_path_traversal_blocked(self, clean_git_repo, tmp_path):
        """Test that path traversal attempts are blocked in copy_artifact."""
        # Create a source file
        source_file = tmp_path / "source.txt"
        source_file.write_text("Content")

        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Try path traversal with custom filename
            with pytest.raises((ValueError, StorageError), match="path traversal"):
                yanex.copy_artifact(source_file, "../../../malicious.txt")

            # Try absolute path as filename
            with pytest.raises((ValueError, StorageError), match="absolute paths not allowed"):
                yanex.copy_artifact(source_file, "/tmp/malicious.txt")

    def test_artifact_exists_path_traversal_blocked(self, clean_git_repo, tmp_path):
        """Test that path traversal attempts are blocked in artifact_exists."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Try path traversal
            with pytest.raises((ValueError, StorageError), match="path traversal"):
                yanex.artifact_exists("../../../etc/passwd")

    def test_load_artifact_path_traversal_blocked(self, clean_git_repo, tmp_path):
        """Test that path traversal attempts are blocked in load_artifact."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Try path traversal
            with pytest.raises((ValueError, StorageError), match="path traversal"):
                yanex.load_artifact("../../../etc/passwd")

    def test_save_artifact_filename_sanitized(self, clean_git_repo, tmp_path):
        """Test that filenames are sanitized (basename extraction)."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Save with path-like filename (should extract basename)
            yanex.save_artifact("content", "subdir/file.txt")

            # Should be saved as just "file.txt" in artifacts dir
            artifacts_dir = yanex.get_artifacts_dir()
            assert (artifacts_dir / "file.txt").exists()
            assert not (artifacts_dir / "subdir").exists()

    def test_copy_artifact_filename_sanitized(self, clean_git_repo, tmp_path):
        """Test that filenames are sanitized in copy_artifact."""
        source_file = tmp_path / "source.txt"
        source_file.write_text("Content")

        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path):
            # Copy with path-like filename (should extract basename)
            yanex.copy_artifact(source_file, "subdir/renamed.txt")

            # Should be saved as just "renamed.txt" in artifacts dir
            artifacts_dir = yanex.get_artifacts_dir()
            assert (artifacts_dir / "renamed.txt").exists()
            assert not (artifacts_dir / "subdir").exists()

    def test_save_artifact_standalone_path_traversal_blocked(
        self, tmp_path, monkeypatch
    ):
        """Test path traversal blocked in standalone mode."""
        monkeypatch.chdir(tmp_path)

        # Try path traversal in standalone mode (use valid extension)
        with pytest.raises(ValueError, match="path traversal"):
            yanex.save_artifact("malicious", "../../../etc/passwd.txt")

    def test_copy_artifact_standalone_path_traversal_blocked(
        self, tmp_path, monkeypatch
    ):
        """Test path traversal blocked in standalone mode for copy_artifact."""
        monkeypatch.chdir(tmp_path)

        source_file = tmp_path / "source.txt"
        source_file.write_text("Content")

        # Try path traversal in standalone mode
        with pytest.raises(ValueError, match="path traversal"):
            yanex.copy_artifact(source_file, "../../../malicious.txt")
