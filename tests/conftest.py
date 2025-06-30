"""
Pytest configuration and shared fixtures.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import git
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def git_repo(temp_dir: Path) -> git.Repo:
    """Create a temporary git repository for tests."""
    repo = git.Repo.init(temp_dir)

    # Configure git user for tests
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Create initial commit
    test_file = temp_dir / "test.txt"
    test_file.write_text("initial content")
    repo.index.add([str(test_file.relative_to(temp_dir))])
    repo.index.commit("Initial commit")

    return repo


@pytest.fixture
def clean_git_repo(git_repo: git.Repo) -> git.Repo:
    """Ensure git repo is in clean state."""
    # Make sure working directory is clean
    assert not git_repo.is_dirty()
    return git_repo


@pytest.fixture
def sample_config_yaml(temp_dir: Path) -> Path:
    """Create a sample config.yaml file."""
    config_path = temp_dir / "config.yaml"
    config_content = """
n_docs: 1000
batch_size: 32
learning_rate: 0.001
model_type: "transformer"
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_experiment_script(temp_dir: Path) -> Path:
    """Create a sample experiment script."""
    script_path = temp_dir / "experiment.py"
    script_content = """
import yanex

params = yanex.get_params()

result = {
    "accuracy": 0.95,
    "loss": 0.05,
    "docs_processed": params.get("n_docs", 1000)
}
yanex.log_results(result)
"""
    script_path.write_text(script_content)
    return script_path


# Additional fixtures for test infrastructure consolidation
# These supplement existing fixtures without replacing them

@pytest.fixture
def isolated_experiments_dir(temp_dir: Path) -> Path:
    """Create an isolated experiments directory for testing."""
    experiments_dir = temp_dir / "experiments"
    experiments_dir.mkdir()
    return experiments_dir


@pytest.fixture
def isolated_storage(isolated_experiments_dir: Path):
    """Create an isolated ExperimentStorage instance."""
    from yanex.core.storage import ExperimentStorage
    return ExperimentStorage(isolated_experiments_dir)


@pytest.fixture
def isolated_manager(isolated_experiments_dir: Path):
    """Create an isolated ExperimentManager instance."""
    from yanex.core.manager import ExperimentManager
    return ExperimentManager(experiments_dir=isolated_experiments_dir)


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner for testing."""
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def sample_experiment_metadata():
    """Create sample experiment metadata for testing."""
    from tests.test_utils import TestDataFactory
    return TestDataFactory.create_experiment_metadata(
        experiment_id="test001",
        name="Test Experiment",
        tags=["test", "sample"]
    )
