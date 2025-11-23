"""Tests for Experiment artifact methods (load_artifact, artifact_exists, list_artifacts)."""

import os

import pytest

import yanex
from yanex.results import ResultsManager


@pytest.fixture(autouse=True)
def clear_cli_env():
    """Clear YANEX_CLI_ACTIVE environment variable before each test."""
    os.environ.pop("YANEX_CLI_ACTIVE", None)
    yield
    os.environ.pop("YANEX_CLI_ACTIVE", None)


class TestExperimentLoadArtifact:
    """Test Experiment.load_artifact() method."""

    def test_load_artifact(self, clean_git_repo, tmp_path):
        """Test loading artifact from experiment."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        # Create experiment and save artifacts
        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("Hello, world!", "message.txt")
            yanex.save_artifact({"key": "value"}, "data.json")

        # Load artifacts via Results API
        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        loaded_text = exp.load_artifact("message.txt")
        assert loaded_text == "Hello, world!"

        loaded_json = exp.load_artifact("data.json")
        assert loaded_json == {"key": "value"}

    def test_load_artifact_not_found(self, clean_git_repo, tmp_path):
        """Test loading nonexistent artifact returns None."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        loaded = exp.load_artifact("nonexistent.txt")
        assert loaded is None

    def test_load_artifact_auto_detection(self, clean_git_repo, tmp_path):
        """Test automatic format detection when loading."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("Text", "file.txt")
            yanex.save_artifact({"data": [1, 2, 3]}, "file.json")
            yanex.save_artifact([{"id": 1}, {"id": 2}], "file.jsonl")

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        # Each format loaded correctly
        assert exp.load_artifact("file.txt") == "Text"
        assert exp.load_artifact("file.json") == {"data": [1, 2, 3]}
        assert exp.load_artifact("file.jsonl") == [{"id": 1}, {"id": 2}]

    def test_load_artifact_custom_loader(self, clean_git_repo, tmp_path):
        """Test loading artifact with custom loader."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("hello", "message.txt")

        def custom_loader(path):
            return path.read_text().upper()

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        loaded = exp.load_artifact("message.txt", loader=custom_loader)
        assert loaded == "HELLO"


class TestExperimentArtifactExists:
    """Test Experiment.artifact_exists() method."""

    def test_artifact_exists(self, clean_git_repo, tmp_path):
        """Test checking artifact existence."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("content", "file.txt")

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        assert exp.artifact_exists("file.txt")
        assert not exp.artifact_exists("nonexistent.txt")

    def test_artifact_exists_multiple(self, clean_git_repo, tmp_path):
        """Test checking multiple artifacts."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("content1", "file1.txt")
            yanex.save_artifact("content2", "file2.txt")
            yanex.save_artifact("content3", "file3.txt")

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        assert exp.artifact_exists("file1.txt")
        assert exp.artifact_exists("file2.txt")
        assert exp.artifact_exists("file3.txt")
        assert not exp.artifact_exists("file4.txt")


class TestExperimentListArtifacts:
    """Test Experiment.list_artifacts() method."""

    def test_list_artifacts(self, clean_git_repo, tmp_path):
        """Test listing artifacts."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("content1", "file1.txt")
            yanex.save_artifact("content2", "file2.txt")
            yanex.save_artifact({"key": "value"}, "data.json")

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        artifacts = [a for a in exp.list_artifacts() if a != "git_diff.patch"]
        assert sorted(artifacts) == ["data.json", "file1.txt", "file2.txt"]

    def test_list_artifacts_empty(self, clean_git_repo, tmp_path):
        """Test listing artifacts when there are none."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        # Filter out git_diff.patch which may be auto-created
        artifacts = [a for a in exp.list_artifacts() if a != "git_diff.patch"]
        assert artifacts == []

    def test_list_artifacts_sorted(self, clean_git_repo, tmp_path):
        """Test that artifacts are returned sorted."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            # Save in random order
            yanex.save_artifact("c", "c.txt")
            yanex.save_artifact("a", "a.txt")
            yanex.save_artifact("b", "b.txt")

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        artifacts = [a for a in exp.list_artifacts() if a != "git_diff.patch"]
        assert artifacts == ["a.txt", "b.txt", "c.txt"]


class TestExperimentArtifactsIntegration:
    """Integration tests for Experiment artifact methods."""

    def test_complete_artifact_workflow(self, clean_git_repo, tmp_path):
        """Test complete workflow: save, load, check, list."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        # Create experiment and save artifacts
        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("Log content", "log.txt")
            yanex.save_artifact({"accuracy": 0.95}, "metrics.json")
            yanex.save_artifact([1, 2, 3], "results.pkl")

        # Access via Results API
        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        # List artifacts (filter out git_diff.patch)
        artifacts = [a for a in exp.list_artifacts() if a != "git_diff.patch"]
        assert sorted(artifacts) == ["log.txt", "metrics.json", "results.pkl"]

        # Check existence
        assert exp.artifact_exists("log.txt")
        assert exp.artifact_exists("metrics.json")
        assert exp.artifact_exists("results.pkl")
        assert not exp.artifact_exists("missing.txt")

        # Load artifacts
        log = exp.load_artifact("log.txt")
        metrics = exp.load_artifact("metrics.json")
        results = exp.load_artifact("results.pkl")

        assert log == "Log content"
        assert metrics == {"accuracy": 0.95}
        assert results == [1, 2, 3]

    def test_optional_artifact_loading(self, clean_git_repo, tmp_path):
        """Test optional artifact loading pattern."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact({"required": True}, "config.json")
            # Don't save optional checkpoint

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        # Required artifact exists
        config = exp.load_artifact("config.json")
        assert config == {"required": True}

        # Optional artifact returns None
        checkpoint = exp.load_artifact("checkpoint.pkl")
        assert checkpoint is None

    def test_experiment_to_dict_uses_list_artifacts(self, clean_git_repo, tmp_path):
        """Test that Experiment.to_dict() uses new list_artifacts method."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("content1", "file1.txt")
            yanex.save_artifact("content2", "file2.txt")

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        exp_dict = exp.to_dict()
        assert "artifacts" in exp_dict
        artifacts = [a for a in exp_dict["artifacts"] if a != "git_diff.patch"]
        assert sorted(artifacts) == ["file1.txt", "file2.txt"]

    def test_artifacts_dir_property_still_works(self, clean_git_repo, tmp_path):
        """Test that artifacts_dir property still works for manual access."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        with yanex.create_experiment(script_path) as ctx:
            exp_id = ctx.experiment_id
            yanex.save_artifact("content", "file.txt")

        manager = ResultsManager()
        exp = manager.get_experiment(exp_id)

        # Access artifacts_dir for manual path construction
        artifacts_dir = exp.artifacts_dir
        assert artifacts_dir.exists()
        assert (artifacts_dir / "file.txt").exists()

    def test_multiple_experiments_isolated_artifacts(self, clean_git_repo, tmp_path):
        """Test that artifacts are isolated between experiments."""
        script_path = tmp_path / "script.py"
        script_path.write_text("import yanex")

        # Create first experiment
        with yanex.create_experiment(script_path) as ctx:
            exp1_id = ctx.experiment_id
            yanex.save_artifact("Experiment 1", "data.txt")

        # Create second experiment
        with yanex.create_experiment(script_path) as ctx:
            exp2_id = ctx.experiment_id
            yanex.save_artifact("Experiment 2", "data.txt")

        manager = ResultsManager()
        exp1 = manager.get_experiment(exp1_id)
        exp2 = manager.get_experiment(exp2_id)

        # Each experiment has its own artifacts
        assert exp1.load_artifact("data.txt") == "Experiment 1"
        assert exp2.load_artifact("data.txt") == "Experiment 2"

        # Each only sees its own artifacts (filter out git_diff.patch)
        artifacts1 = [a for a in exp1.list_artifacts() if a != "git_diff.patch"]
        artifacts2 = [a for a in exp2.list_artifacts() if a != "git_diff.patch"]
        assert artifacts1 == ["data.txt"]
        assert artifacts2 == ["data.txt"]
