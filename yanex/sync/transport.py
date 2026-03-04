"""Transport layer for syncing experiments via rsync and AWS S3.

Wraps subprocess calls to rsync and aws s3 sync. No new Python dependencies.
"""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import click

from .target import SyncProtocol, SyncTarget


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success_count: int = 0
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.success_count + self.failed_count

    @property
    def all_succeeded(self) -> bool:
        return self.failed_count == 0


def check_tool(name: str) -> None:
    """Check that a required external tool is installed.

    Args:
        name: Tool name (e.g. "rsync", "aws")

    Raises:
        click.ClickException: If tool is not found
    """
    if shutil.which(name) is None:
        install_hints = {
            "rsync": "Install rsync: apt install rsync (Linux) or brew install rsync (macOS)",
            "aws": "Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html",
        }
        hint = install_hints.get(name, f"Install {name} and ensure it's on your PATH")
        raise click.ClickException(f"'{name}' is not installed. {hint}")


def sync_experiments_push(
    experiment_ids: list[str],
    target: SyncTarget,
    local_dir: Path,
    progress: bool = False,
) -> SyncResult:
    """Push local experiments to a remote target.

    Args:
        experiment_ids: List of experiment IDs to push
        target: Parsed sync target
        local_dir: Local experiments directory
        progress: Show transfer progress

    Returns:
        SyncResult with outcome
    """
    if target.protocol == SyncProtocol.SSH:
        return _rsync_push(experiment_ids, target, local_dir, progress)
    return _s3_push(experiment_ids, target, local_dir, progress)


def sync_experiments_pull(
    experiment_ids: list[str],
    target: SyncTarget,
    local_dir: Path,
    progress: bool = False,
) -> SyncResult:
    """Pull remote experiments to local storage.

    Args:
        experiment_ids: List of experiment IDs to pull
        target: Parsed sync target
        local_dir: Local experiments directory
        progress: Show transfer progress

    Returns:
        SyncResult with outcome
    """
    if target.protocol == SyncProtocol.SSH:
        return _rsync_pull(experiment_ids, target, local_dir, progress)
    return _s3_pull(experiment_ids, target, local_dir, progress)


def _rsync_push(
    experiment_ids: list[str],
    target: SyncTarget,
    local_dir: Path,
    progress: bool,
) -> SyncResult:
    """Push experiments via rsync."""
    check_tool("rsync")

    # Build rsync command: rsync -az src1 src2 ... dest/
    cmd = ["rsync", "-az"]
    if progress:
        cmd.append("--progress")

    # Add each experiment directory as a source
    for exp_id in experiment_ids:
        cmd.append(str(local_dir / exp_id))

    # Destination: host:path/
    cmd.append(target.ssh_dest)

    return _run_rsync(cmd, progress=progress)


def _rsync_pull(
    experiment_ids: list[str],
    target: SyncTarget,
    local_dir: Path,
    progress: bool,
) -> SyncResult:
    """Pull experiments via rsync."""
    check_tool("rsync")

    # Build rsync command: rsync -az remote:path/id1 remote:path/id2 ... local/
    cmd = ["rsync", "-az"]
    if progress:
        cmd.append("--progress")

    # Add each remote experiment directory as a source
    user_prefix = f"{target.user}@" if target.user else ""
    for exp_id in experiment_ids:
        cmd.append(f"{user_prefix}{target.host}:{target.path}{exp_id}")

    # Destination: local experiments directory
    cmd.append(str(local_dir) + "/")

    return _run_rsync(cmd, progress=progress)


def _run_rsync(cmd: list[str], progress: bool = False) -> SyncResult:
    """Execute an rsync command and return the result."""
    try:
        if progress:
            # Let stdout pass through to terminal for progress display
            proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            # rsync is all-or-nothing per invocation
            return SyncResult(success_count=1)
        else:
            error_msg = (
                proc.stderr.strip() or f"rsync exited with code {proc.returncode}"
            )
            return SyncResult(failed_count=1, errors=[error_msg])
    except OSError as e:
        return SyncResult(failed_count=1, errors=[str(e)])


def _s3_push(
    experiment_ids: list[str],
    target: SyncTarget,
    local_dir: Path,
    progress: bool,
) -> SyncResult:
    """Push experiments to S3 (one aws s3 sync per experiment)."""
    check_tool("aws")
    result = SyncResult()

    for exp_id in experiment_ids:
        local_path = str(local_dir / exp_id)
        s3_path = f"{target.s3_url}/{exp_id}"
        outcome = _run_s3_sync(local_path, s3_path, progress)
        result.success_count += outcome.success_count
        result.failed_count += outcome.failed_count
        result.errors.extend(outcome.errors)

    return result


def _s3_pull(
    experiment_ids: list[str],
    target: SyncTarget,
    local_dir: Path,
    progress: bool,
) -> SyncResult:
    """Pull experiments from S3 (one aws s3 sync per experiment)."""
    check_tool("aws")
    result = SyncResult()

    for exp_id in experiment_ids:
        s3_path = f"{target.s3_url}/{exp_id}"
        local_path = str(local_dir / exp_id)
        outcome = _run_s3_sync(s3_path, local_path, progress)
        result.success_count += outcome.success_count
        result.failed_count += outcome.failed_count
        result.errors.extend(outcome.errors)

    return result


def _run_s3_sync(source: str, dest: str, progress: bool) -> SyncResult:
    """Execute an aws s3 sync command for a single experiment."""
    cmd = ["aws", "s3", "sync", source, dest]
    if not progress:
        cmd.append("--quiet")

    try:
        if progress:
            # Let stdout pass through to terminal for file listing
            proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return SyncResult(success_count=1)
        else:
            error_msg = (
                proc.stderr.strip() or f"aws s3 sync exited with code {proc.returncode}"
            )
            return SyncResult(failed_count=1, errors=[error_msg])
    except OSError as e:
        return SyncResult(failed_count=1, errors=[str(e)])
