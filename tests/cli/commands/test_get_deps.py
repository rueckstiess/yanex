"""Dependency argument output and shell aggregation integration tests."""

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_utils import create_cli_runner
from yanex.cli.formatters import OutputFormat
from yanex.cli.formatters.output import is_machine_format
from yanex.cli.main import cli
from yanex.core.manager import ExperimentManager


@pytest.fixture
def dependency_runs(per_test_experiments_dir, git_repo):
    manager = ExperimentManager(per_test_experiments_dir)
    script = Path(git_repo.working_dir) / "aggregate.py"
    script.write_text("print('aggregate')\n")
    ids = []
    for i in range(3):
        exp_id = manager.create_experiment(script, config={}, name=f"input-{i}")
        manager.complete_experiment(exp_id)
        ids.append(exp_id)
    return manager, script, ids


@pytest.mark.parametrize("field", ["id", "meta:id"])
def test_deps_single_and_multiple(dependency_runs, field):
    _, _, ids = dependency_runs
    runner = create_cli_runner()
    single = runner.invoke(cli, ["get", field, ids[0][:6], "-F", "deps"])
    assert single.exit_code == 0, single.output
    assert single.output == f"-D {ids[0]}"
    filters = ["--ids", ",".join(ids), "--global"]
    sweep = runner.invoke(cli, ["get", field, *filters, "-F", "sweep"])
    deps = runner.invoke(cli, ["get", field, *filters, "--format", "deps"])
    assert sweep.exit_code == deps.exit_code == 0
    ordered_ids = sweep.output.split(",")
    assert set(ordered_ids) == set(ids)
    assert deps.output == " ".join(f"-D {exp_id}" for exp_id in ordered_ids)
    limited = runner.invoke(cli, ["get", field, *filters, "-l", "1", "-F", "deps"])
    assert limited.exit_code == 0
    assert limited.output == f"-D {ordered_ids[0]}"


@pytest.mark.parametrize("field", ["id", "meta:id"])
def test_deps_empty_results(per_test_experiments_dir, field):
    result = create_cli_runner().invoke(
        cli, ["get", field, "-n", "missing-*", "-F", "deps"]
    )
    assert result.exit_code == 0
    assert result.output == ""


@pytest.mark.parametrize(
    "field", ["status", "param:lr", "stdout", "upstream", "dependencies"]
)
def test_deps_rejects_other_fields_before_empty_return(per_test_experiments_dir, field):
    result = create_cli_runner().invoke(
        cli, ["get", field, "-n", "missing-*", "-F", "deps"]
    )
    assert result.exit_code != 0
    assert "--format deps only supports 'id' and 'meta:id' fields" in result.output


def test_deps_is_machine_output_and_get_only():
    assert is_machine_format(OutputFormat.DEPS)
    runner = create_cli_runner()
    assert "deps" in runner.invoke(cli, ["get", "--help"]).output
    result = runner.invoke(cli, ["list", "-F", "deps"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output


@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_shell_substitution_creates_one_run_with_all_dependencies(
    dependency_runs, shell
):
    shell_path = shutil.which(shell)
    if shell_path is None:
        pytest.skip(f"{shell} is not installed")
    manager, script, ids = dependency_runs
    yanex = shlex.join([sys.executable, "-m", "yanex"])
    command = (
        f"{yanex} run {shlex.quote(str(script))} --name aggregated "
        f'$({yanex} get id -n "input-*" -s completed --global -F deps)'
    )
    result = subprocess.run(
        [shell_path, "-c", command],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    new_ids = set(manager.list_experiments()) - set(ids)
    assert len(new_ids) == 1
    aggregate_id = new_ids.pop()
    assert manager.get_experiment_status(aggregate_id) == "completed"
    dependencies = manager.storage.dependency_storage.load_dependencies(aggregate_id)[
        "dependencies"
    ]
    assert set(dependencies) == {"dep1", "dep2", "dep3"}
    assert set(dependencies.values()) == set(ids)
