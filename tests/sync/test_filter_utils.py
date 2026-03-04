"""Tests for remote experiment filtering."""

from yanex.sync.filter_utils import filter_experiment_dicts

# Sample metadata dicts (mimicking what we'd get from remote)
EXPERIMENTS = [
    {
        "id": "abc12345",
        "name": "train-resnet",
        "status": "completed",
        "tags": ["training", "cv"],
        "project": "image-cls",
        "script_path": "train.py",
        "started_at": "2025-06-01T10:00:00",
        "ended_at": "2025-06-01T12:00:00",
    },
    {
        "id": "def67890",
        "name": "eval-resnet",
        "status": "completed",
        "tags": ["evaluation"],
        "project": "image-cls",
        "script_path": "eval.py",
        "started_at": "2025-06-02T10:00:00",
        "ended_at": "2025-06-02T11:00:00",
    },
    {
        "id": "ghi11111",
        "name": "train-bert",
        "status": "failed",
        "tags": ["training", "nlp"],
        "project": "text-cls",
        "script_path": "train.py",
        "started_at": "2025-06-03T10:00:00",
        "ended_at": "2025-06-03T10:30:00",
    },
    {
        "id": "jkl22222",
        "name": "data-prep",
        "status": "running",
        "tags": [],
        "project": "image-cls",
        "script_path": "prepare.py",
        "started_at": "2025-06-04T10:00:00",
    },
]


class TestFilterExperimentDicts:
    """Tests for filter_experiment_dicts()."""

    def test_no_filters_returns_all(self):
        result = filter_experiment_dicts(EXPERIMENTS)
        assert len(result) == len(EXPERIMENTS)

    def test_filter_by_status(self):
        result = filter_experiment_dicts(EXPERIMENTS, status="completed")
        assert len(result) == 2
        assert all(e["status"] == "completed" for e in result)

    def test_filter_by_name_pattern(self):
        result = filter_experiment_dicts(EXPERIMENTS, name="train*")
        assert len(result) == 2
        assert {e["name"] for e in result} == {"train-resnet", "train-bert"}

    def test_filter_by_tags_and_logic(self):
        result = filter_experiment_dicts(EXPERIMENTS, tags=["training"])
        assert len(result) == 2

        result = filter_experiment_dicts(EXPERIMENTS, tags=["training", "cv"])
        assert len(result) == 1
        assert result[0]["id"] == "abc12345"

    def test_filter_by_ids(self):
        result = filter_experiment_dicts(EXPERIMENTS, ids=["abc12345", "ghi11111"])
        assert len(result) == 2
        assert {e["id"] for e in result} == {"abc12345", "ghi11111"}

    def test_filter_by_project(self):
        result = filter_experiment_dicts(EXPERIMENTS, project="text-cls")
        assert len(result) == 1
        assert result[0]["id"] == "ghi11111"

    def test_filter_by_script_pattern(self):
        result = filter_experiment_dicts(EXPERIMENTS, script_pattern="train*")
        assert len(result) == 2

    def test_combined_filters(self):
        result = filter_experiment_dicts(EXPERIMENTS, status="completed", name="train*")
        assert len(result) == 1
        assert result[0]["id"] == "abc12345"

    def test_no_matches(self):
        result = filter_experiment_dicts(EXPERIMENTS, name="nonexistent*")
        assert result == []

    def test_empty_input(self):
        result = filter_experiment_dicts([])
        assert result == []
