"""
Pytest configuration and shared fixtures.
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import git
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_experiments_directory():
    """Automatically isolate experiments directory for entire test session.

    This fixture ensures ALL tests use a temporary experiments directory
    instead of the production ~/.yanex/experiments folder.

    The fixture:
    - Creates a temporary directory for test experiments
    - Sets YANEX_EXPERIMENTS_DIR environment variable
    - Automatically cleans up after all tests complete

    This runs automatically for every test session (autouse=True)
    and does not need to be explicitly requested by individual tests.
    """
    # Create temporary directory for test experiments
    test_experiments_dir = tempfile.mkdtemp(prefix="yanex_test_experiments_")

    # Save original value (if exists)
    original_env = os.environ.get("YANEX_EXPERIMENTS_DIR")

    # Set environment variable for entire test session
    os.environ["YANEX_EXPERIMENTS_DIR"] = test_experiments_dir

    try:
        yield test_experiments_dir
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ["YANEX_EXPERIMENTS_DIR"] = original_env
        else:
            os.environ.pop("YANEX_EXPERIMENTS_DIR", None)

        # Clean up temporary directory
        shutil.rmtree(test_experiments_dir, ignore_errors=True)


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

# Access all params to ensure they're tracked
for key in params:
    _ = params[key]

result = {
    "accuracy": 0.95,
    "loss": 0.05,
    "docs_processed": params.get("n_docs", 1000)
}
yanex.log_metrics(result)
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
def per_test_experiments_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide per-test experiment directory isolation with environment override.

    This fixture creates a completely isolated experiments directory for a single test
    by temporarily overriding the YANEX_EXPERIMENTS_DIR environment variable.

    Use this fixture when your test needs complete isolation from other tests
    (e.g., when testing default ExperimentManager() initialization or when you need
    an empty experiments directory).

    This works on top of the session-wide isolation - it temporarily overrides
    the session-wide temp directory with a test-specific one.

    Example:
        def test_something(self, per_test_experiments_dir):
            # ExperimentManager() will use per_test_experiments_dir
            manager = ExperimentManager()
            # Only experiments created in THIS test will be in the directory
    """
    import yanex.results as yr

    test_exp_dir = tmp_path / "experiments"
    test_exp_dir.mkdir()

    # Save the current (session-wide) value
    old_env = os.environ.get("YANEX_EXPERIMENTS_DIR")

    # Reset cached results manager to ensure it picks up the new directory
    old_manager = yr._default_manager
    yr._default_manager = None

    # Override with test-specific directory
    os.environ["YANEX_EXPERIMENTS_DIR"] = str(test_exp_dir)

    try:
        yield test_exp_dir
    finally:
        # Restore session-wide directory
        if old_env:
            os.environ["YANEX_EXPERIMENTS_DIR"] = old_env

        # Restore cached results manager (or keep it None to force re-creation)
        yr._default_manager = old_manager

        # tmp_path cleanup is handled by pytest's tmp_path fixture


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
        experiment_id="test001", name="Test Experiment", tags=["test", "sample"]
    )


@pytest.fixture
def clean_api_state():
    """Clean up yanex API global state before and after each test.

    This fixture ensures that the API's cached TrackedDict and atexit handler
    registration are cleared between tests to prevent state leakage. It also
    disables atexit parameter saving after tests to prevent warnings when temp
    directories are cleaned up before Python exits.

    Use this fixture for any test that calls yanex.get_params() or yanex.get_param()
    to ensure test isolation.

    Example:
        def test_something(self, clean_api_state):
            # API state is clean at start
            params = yanex.get_params()
            # API state will be cleaned up after test
    """
    import yanex

    # Clean before test
    if hasattr(yanex.api._local, "experiment_id"):
        del yanex.api._local.experiment_id
    yanex.api._tracked_params = None
    yanex.api._atexit_registered = False
    yanex.api._should_save_on_exit = True  # Enable saving for test

    yield

    # Clean after test and disable atexit saving to prevent warnings
    yanex.api._should_save_on_exit = False  # Disable to prevent atexit errors
    if hasattr(yanex.api._local, "experiment_id"):
        del yanex.api._local.experiment_id
    yanex.api._tracked_params = None
    yanex.api._atexit_registered = False
