import typer

app = typer.Typer(help="Yet Another Experiment harness (yanex)")


@app.command()
def run(
    name: str = typer.Argument(..., help="Name of the experiment"),
    script: str = typer.Argument(..., help="Path to the Python script to run"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to YAML config file"),
    tags: list[str] = typer.Option([], "--tag", "-t", help="Tags for this experiment"),
    desc: str = typer.Option("", "--desc", "-d", help="Optional description"),
    param_override: list[str] = typer.Option([], "--param", help="Override parameters, e.g. --param=batch_size=64"),
):
    """Run an experiment with given parameters."""
    import os
    import subprocess
    from datetime import datetime
    from pathlib import Path
    import yaml

    from yanex.utils import (
        ensure_git_clean,
        get_git_commit_hash,
        parse_param_overrides,
        deep_update_dict,
        slugify,
    )

    base_dir = Path("./experiments")
    base_dir.mkdir(exist_ok=True)

    # Ensure git is clean
    ensure_git_clean()

    # Resolve script path
    script_path = Path(script).resolve()
    if not script_path.exists():
        raise typer.BadParameter(f"Script not found: {script_path}")

    # Load config
    with open(config, "r") as f:
        config_params = yaml.safe_load(f)

    overrides = parse_param_overrides(param_override)
    params = deep_update_dict(config_params, overrides)

    # Create experiment folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(name)
    exp_id = f"{timestamp}-{slug}"
    exp_dir = base_dir / exp_id
    exp_dir.mkdir()

    # Save parameters
    with open(exp_dir / "parameters.yaml", "w") as f:
        yaml.safe_dump(params, f)

    # Save metadata
    metadata = {
        "id": exp_id,
        "name": name,
        "tags": tags,
        "description": desc,
        "script": str(script_path),
        "git_commit": get_git_commit_hash(),
        "started_at": datetime.now().isoformat(),
    }
    with open(exp_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)

    # Run script
    log_path = exp_dir / "log.txt"
    env = os.environ.copy()
    env = os.environ.copy()
    env["YANEX_CONFIG_PATH"] = str(exp_dir / "parameters.yaml")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")

    with open(log_path, "w") as log_file:
        process = subprocess.run(
            ["python", str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=script_path.parent,
            env=env,
        )
        metadata["exit_code"] = process.returncode
        metadata["finished_at"] = datetime.now().isoformat()
        metadata["duration_sec"] = (
            datetime.fromisoformat(metadata["finished_at"]) - datetime.fromisoformat(metadata["started_at"])
        ).total_seconds()

    with open(exp_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)

    typer.echo(f"Experiment {exp_id} completed with exit code {process.returncode}")


@app.command(name="list")
def list_cmd(
    name: str = typer.Option(None),
    tag: list[str] = typer.Option(None),
    after: str = typer.Option(None, help="Start time after (e.g. '3 days ago')"),
    before: str = typer.Option(None),
    param: list[str] = typer.Option(None),
):
    """List experiments with optional filters."""
    pass


@app.command(name="archive")
def archive(
    experiment_id: str = typer.Argument(..., help="ID of experiment to archive"),
):
    """Move an experiment to archive storage."""
    pass


@app.command(name="compare")
def compare_cmd(ids: list[str] = typer.Argument(..., help="Experiment IDs to compare")):
    """Compare results of multiple experiments in a table."""
    pass


if __name__ == "__main__":
    app()
