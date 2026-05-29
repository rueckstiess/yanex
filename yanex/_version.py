"""Package version helpers."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_version() -> str:
    """Return the package version from installed metadata or pyproject.toml."""
    try:
        return version("yanex")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject = tomllib.load(f)
        return pyproject["project"]["version"]


__version__ = get_version()
