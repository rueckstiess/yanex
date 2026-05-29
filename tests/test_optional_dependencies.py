"""Regression tests for optional analysis dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_with_blocked_optional_deps(statement: str) -> subprocess.CompletedProcess[str]:
    """Run a Python statement while pretending optional deps are not installed."""
    import_blocker = textwrap.dedent(
        """
        import builtins

        blocked = {"matplotlib", "numpy", "pandas"}
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked:
                raise ModuleNotFoundError(
                    f"No module named '{name.split('.', 1)[0]}'",
                    name=name.split(".", 1)[0],
                )
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        """
    )
    code = f"{import_blocker}\n{textwrap.dedent(statement)}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_top_level_import_does_not_require_analysis_dependencies() -> None:
    result = _run_with_blocked_optional_deps("import yanex")

    assert result.returncode == 0, result.stderr


def test_cli_import_does_not_require_analysis_dependencies() -> None:
    result = _run_with_blocked_optional_deps("from yanex.cli.main import cli")

    assert result.returncode == 0, result.stderr


def test_results_import_does_not_require_plotting_dependencies() -> None:
    result = _run_with_blocked_optional_deps("import yanex.results as yr")

    assert result.returncode == 0, result.stderr


def test_plot_metrics_reports_missing_optional_dependency() -> None:
    result = _run_with_blocked_optional_deps(
        textwrap.dedent(
            """
            import yanex.results as yr

            try:
                yr.plot_metrics(None)
            except ImportError as exc:
                message = str(exc)
                assert "plot_metrics requires optional plotting dependencies" in message
                assert "yanex[results]" in message
            else:
                raise AssertionError("plot_metrics should fail without plotting deps")
            """
        )
    )

    assert result.returncode == 0, result.stderr
