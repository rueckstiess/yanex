"""Upstream metadata lookup uses only the selected dependencies and ancestors."""

import pytest

from yanex.cli.main import cli
from yanex.core.manager import ExperimentManager


@pytest.fixture
def pipeline(per_test_experiments_dir, git_repo):
    from pathlib import Path

    manager = ExperimentManager(per_test_experiments_dir)
    script = Path(git_repo.working_dir) / "template_run.py"
    script.write_text("print('completed')\n")

    def create(config=None, dependencies=None, staged=False):
        exp_id = manager.create_experiment(
            script, config=config or {}, dependencies=dependencies, stage_only=staged
        )
        if not staged:
            manager.complete_experiment(exp_id)
        return exp_id

    return manager, script, create


def test_transitive_diamond_ignores_other_branches(cli_runner, pipeline):
    manager, script, create = pipeline
    root = create({"model": {"lr": 0.125}})
    left = create(dependencies={"data": root})
    right = create(dependencies={"data": root})
    create({"model": {"lr": 99}}, {"data": root})  # sibling branch
    create({"model": {"lr": 98}}, {"data": left})  # existing descendant
    manager.archive_experiment(root)
    result = cli_runner.invoke(
        cli,
        [
            "run",
            str(script),
            "-D",
            f"left={left[:6]}",
            "-D",
            f"right={right}",
            "-n",
            "lr-{lr:.2f}",
            "-t",
            "lr={model.lr}",
            "-d",
            "rate={lr}",
            "--stage",
        ],
    )
    assert result.exit_code == 0, result.output
    exp_id = manager.get_staged_experiments()[0]
    metadata = manager.get_experiment_metadata(exp_id)
    assert metadata["name"] == "lr-0.12"
    assert metadata["tags"] == ["lr=0.125"]
    assert metadata["description"] == "rate=0.125"


@pytest.mark.parametrize("mode", ["--dry-run", "--stage", "execute"])
def test_agreeing_dependencies_and_local_precedence(cli_runner, pipeline, mode):
    manager, script, create = pipeline
    first = create({"model": {"lr": 0.01}, "seed": 1})
    second = create({"model": {"lr": 0.01}, "seed": 2})
    before = set(manager.list_experiments())
    args = [
        "run",
        str(script),
        "-D",
        f"first={first}",
        "-D",
        f"second={second}",
        "-p",
        "seed=7",
        "-n",
        "lr-{lr}-seed-{seed}",
    ]
    if mode != "execute":
        args.append(mode)
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    created = set(manager.list_experiments()) - before
    if mode == "--dry-run":
        assert not created
        assert "lr-0.01-seed-7" in result.output
    else:
        assert len(created) == 1
        metadata = manager.get_experiment_metadata(created.pop())
        assert metadata["name"] == "lr-0.01-seed-7"
        assert metadata["status"] == ("staged" if mode == "--stage" else "completed")


@pytest.mark.parametrize("mode", ["--dry-run", "--stage", "execute"])
@pytest.mark.parametrize("problem", ["conflict", "missing", "ambiguous"])
def test_later_dependency_combination_fails_before_creation(
    cli_runner, pipeline, mode, problem
):
    manager, script, create = pipeline
    first = create({"lr": 0.01})
    good = create({"lr": 0.01})
    bad_config = {
        "conflict": {"lr": 0.02},
        "missing": {},
        "ambiguous": {"other": {"lr": 0.01}},
    }[problem]
    bad = create(bad_config)
    before = set(manager.list_experiments())
    args = ["run", str(script), "-D", f"model={good},{bad}", "-n", "lr-{lr}"]
    if problem != "missing":
        args += ["-D", f"base={first}"]
    if mode != "execute":
        args.append(mode)
    result = cli_runner.invoke(cli, args)
    assert result.exit_code != 0
    assert "combination 2" in result.output
    assert "lr" in result.output
    if problem == "conflict":
        assert first in result.output and bad in result.output
        assert "different values" in result.output
    assert set(manager.list_experiments()) == before


@pytest.mark.parametrize("parallel", [None, "2"])
def test_cartesian_sweep_formats_selected_upstream_values(
    cli_runner, pipeline, parallel
):
    manager, script, create = pipeline
    first = create({"lr": 0.01})
    second = create({"lr": 0.02})
    before = set(manager.list_experiments())
    args = [
        "run",
        str(script),
        "-D",
        f"model={first},{second}",
        "-p",
        "seed=1,2",
        "-n",
        "lr-{lr:.2f}-seed-{seed}",
    ]
    if parallel:
        args += ["-j", parallel]
    result = cli_runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    metadata = [
        manager.get_experiment_metadata(e)
        for e in set(manager.list_experiments()) - before
    ]
    assert {m["name"] for m in metadata} == {
        f"lr-{lr}-seed-{seed}" for lr in ["0.01", "0.02"] for seed in [1, 2]
    }
    assert all(m["status"] == "completed" for m in metadata)


def test_staged_templates_are_frozen(cli_runner, pipeline):
    manager, script, create = pipeline
    dep = create({"lr": 0.01}, staged=True)
    result = cli_runner.invoke(
        cli,
        [
            "run",
            str(script),
            "-D",
            f"model={dep}",
            "-p",
            "seed=1,2",
            "-n",
            "lr-{lr}-seed-{seed}",
            "--stage",
        ],
    )
    assert result.exit_code == 0, result.output
    children = set(manager.get_staged_experiments()) - {dep}
    assert len(children) == 2
    manager.storage.save_config(dep, {"lr": 0.99})
    manager.complete_experiment(dep)
    for child in children:
        result = cli_runner.invoke(cli, ["run", "--id", child])
        assert result.exit_code == 0, result.output
        assert manager.get_experiment_metadata(child)["name"].startswith("lr-0.01-")


def test_local_ambiguity_does_not_fall_back(cli_runner, pipeline):
    manager, script, create = pipeline
    dep = create({"lr": 0.01})
    result = cli_runner.invoke(
        cli,
        [
            "run",
            str(script),
            "-D",
            f"model={dep}",
            "-p",
            "one.lr=1",
            "-p",
            "two.lr=1",
            "-n",
            "lr-{lr}",
            "--stage",
        ],
    )
    assert result.exit_code != 0
    assert "matches multiple keys" in result.output
    assert not manager.get_staged_experiments()


@pytest.mark.parametrize(
    "metadata",
    ["{broken", "[]", "{}", '{"dependencies": []}', '{"dependencies": {"data": 42}}'],
)
@pytest.mark.parametrize("mode", ["--dry-run", "--stage", "execute"])
def test_malformed_ancestor_metadata_aborts_preflight(
    cli_runner, pipeline, metadata, mode
):
    manager, script, create = pipeline
    first = create({"lr": 1})
    conflicting = create({"lr": 2})
    intermediate = create({}, {"data": conflicting})
    path = manager.storage.get_experiment_directory(intermediate) / "dependencies.json"
    path.write_text(metadata)
    before = set(manager.list_experiments())
    args = [
        "run",
        str(script),
        "-D",
        f"base={first}",
        "-D",
        f"model={first},{intermediate}",
        "-n",
        "lr-{lr}",
    ]
    if mode != "execute":
        args.append(mode)
    result = cli_runner.invoke(cli, args)
    assert result.exit_code != 0
    assert "combination 2" in result.output
    assert "Malformed dependency metadata" in result.output
    assert intermediate in result.output
    assert set(manager.list_experiments()) == before


def test_comma_in_rendered_tag_rejected(cli_runner, pipeline):
    manager, script, create = pipeline
    result = cli_runner.invoke(
        cli, ["run", str(script), "-p", "size=1000", "-t", "size={size:,d}", "--stage"]
    )
    assert result.exit_code != 0
    assert "commas" in result.output
    assert not manager.list_experiments()
