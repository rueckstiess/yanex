import subprocess
import re
from pathlib import Path
import os
import time
import traceback
import yaml
from contextlib import contextmanager
from datetime import datetime

_exp_dir = None
_metadata_path = None
_results_path = None
_artifacts_dir = None
_params = None


def _ensure_initialized():
    global _exp_dir, _metadata_path, _results_path, _artifacts_dir, _params

    if _exp_dir is not None:
        return

    config_path = os.environ.get("YANEX_CONFIG_PATH")
    if not config_path or not Path(config_path).exists():
        raise RuntimeError(
            "YANEX_CONFIG_PATH not set or does not point to a valid file"
        )

    _params = yaml.safe_load(Path(config_path).read_text())
    _exp_dir = Path(config_path).parent
    _metadata_path = _exp_dir / "metadata.yaml"
    _results_path = _exp_dir / "results.yaml"
    _artifacts_dir = _exp_dir / "artefacts"
    _artifacts_dir.mkdir(exist_ok=True)


def get_params() -> dict:
    _ensure_initialized()
    return _params


def log_result(key: str, value):
    _ensure_initialized()
    results = {}
    if _results_path.exists():
        results = yaml.safe_load(_results_path.read_text()) or {}
    results[key] = value
    _results_path.write_text(yaml.safe_dump(results))


def log_results(results_dict: dict):
    _ensure_initialized()
    results = {}
    if _results_path.exists():
        results = yaml.safe_load(_results_path.read_text()) or {}
    results.update(results_dict)
    _results_path.write_text(yaml.safe_dump(results))


def log_artifact(name: str, path: str, copy: bool = True):
    _ensure_initialized()
    target = _artifacts_dir / name
    if copy:
        from shutil import copyfile

        copyfile(path, target)
    else:
        # Just record the reference
        target.write_text(f"Linked artifact path: {os.path.abspath(path)}")


@contextmanager
def run():
    _ensure_initialized()
    meta = yaml.safe_load(_metadata_path.read_text()) or {}
    meta["run_started_at"] = datetime.now().isoformat()
    start = time.time()
    try:
        yield
    except Exception as e:
        meta["exception"] = str(e)
        meta["traceback"] = traceback.format_exc()
        raise
    finally:
        meta["run_finished_at"] = datetime.now().isoformat()
        meta["run_duration_sec"] = round(time.time() - start, 3)
        _metadata_path.write_text(yaml.safe_dump(meta))


def ensure_git_clean():
    """Raises an error if the git working directory is not clean."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError("Not a git repository or git is not available.")
    if result.stdout.strip():
        raise RuntimeError(
            "Git working directory is not clean. Commit or stash changes first."
        )


def get_git_commit_hash() -> str:
    """Returns the current git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to get git commit hash.")
    return result.stdout.decode("utf-8").strip()


def slugify(value: str) -> str:
    """Converts a string into a filesystem-safe slug."""
    value = value.lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value)
    value = re.sub(r"^-+|-+$", "", value)
    return value


def parse_param_overrides(pairs: list[str]) -> dict:
    """Parses flat key=value overrides into a nested dict."""
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid param override: '{pair}'")
        key, value = pair.split("=", 1)
        keys = key.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = yaml.safe_load(value)
    return result


def deep_update_dict(base: dict, updates: dict) -> dict:
    """Recursively updates a dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base
