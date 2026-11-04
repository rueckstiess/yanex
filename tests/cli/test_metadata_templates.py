"""CLI preflight and persisted metadata regression tests."""

import json

import pytest

from yanex.cli.main import cli


@pytest.mark.parametrize("stage", [False, True])
def test_sweep_preflight_does_not_create_partial_runs(
    cli_runner, tmp_path, per_test_experiments_dir, stage
):
    script = tmp_path / "script.py"
    script.write_text("raise RuntimeError('must not run')")
    args = ["run", str(script), "-p", "value=list(1,bad)", "-n", "run-{value:03d}"]
    if stage:
        args.append("--stage")
    result = cli_runner.invoke(cli, args)
    assert result.exit_code != 0
    assert "combination 2" in result.output
    assert not list(per_test_experiments_dir.glob("*/metadata.json"))


def test_staged_sweep_renders_all_fields(
    cli_runner, tmp_path, per_test_experiments_dir
):
    script = tmp_path / "script.py"
    script.write_text("print('hello')")
    args = [
        "run",
        str(script),
        "-p",
        "batch_size=10,50",
        "-n",
        "bs-{batch_size}",
        "-t",
        "batch-{batch_size}",
        "-d",
        "Size {batch_size}",
    ]
    preview = cli_runner.invoke(cli, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert "bs-10" in preview.output and "bs-50" in preview.output
    assert not list(per_test_experiments_dir.glob("*/metadata.json"))
    result = cli_runner.invoke(cli, [*args, "--stage"])
    assert result.exit_code == 0, result.output
    metadata = [
        json.loads(p.read_text())
        for p in per_test_experiments_dir.glob("*/metadata.json")
    ]
    assert {m["name"] for m in metadata} == {"bs-10", "bs-50"}
    for m in metadata:
        size = m["name"].split("-")[1]
        assert f"batch-{size}" in m["tags"]
        assert m["description"] == f"Size {size}"


def test_yaml_single_run_renders_once(cli_runner, tmp_path, per_test_experiments_dir):
    script = tmp_path / "script.py"
    script.write_text("print('hello')")
    config = tmp_path / "settings.yaml"
    config.write_text(
        'value: "{literal}"\nyanex:\n  name: "run-{value}"\n  tag: ["value-{value}"]\n  description: "value={value}"\n'
    )
    result = cli_runner.invoke(cli, ["run", str(script), "-c", str(config), "--stage"])
    assert result.exit_code == 0, result.output
    metadata = json.loads(
        next(per_test_experiments_dir.glob("*/metadata.json")).read_text()
    )
    assert metadata["name"] == "run-{literal}"
    assert metadata["tags"] == ["value-{literal}"]
