"""Fetch experiment metadata from remote targets.

Used by `yanex pull` to discover and filter remote experiments
before transferring them.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import click

from .target import SyncProtocol, SyncTarget
from .transport import check_tool

# Separator used to delimit metadata.json entries in SSH output
_SSH_SEPARATOR = "===YANEX_SEP==="


def fetch_remote_metadata(target: SyncTarget) -> list[dict[str, Any]]:
    """Fetch experiment metadata from a remote target.

    Args:
        target: Parsed sync target

    Returns:
        List of experiment metadata dicts, each with an 'id' key

    Raises:
        click.ClickException: On connection or tool errors
    """
    if target.protocol == SyncProtocol.SSH:
        return _fetch_ssh_metadata(target)
    return _fetch_s3_metadata(target)


def _fetch_ssh_metadata(target: SyncTarget) -> list[dict[str, Any]]:
    """Fetch metadata from an SSH target by reading remote metadata.json files."""
    check_tool("rsync")  # We also need SSH, but rsync implies SSH availability

    # Build SSH command to cat all metadata.json files with separators
    remote_cmd = (
        f"for f in {target.path}*/metadata.json; do "
        f'[ -f "$f" ] && echo "{_SSH_SEPARATOR}" && cat "$f"; '
        f"done"
    )

    ssh_target = f"{target.user}@{target.host}" if target.user else target.host
    cmd = ["ssh", ssh_target, remote_cmd]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise click.ClickException(
            f"SSH connection to {ssh_target} timed out while fetching metadata"
        )
    except OSError as e:
        raise click.ClickException(f"Failed to connect to {ssh_target}: {e}")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise click.ClickException(
            f"Failed to fetch metadata from {ssh_target}: {stderr}"
        )

    return _parse_ssh_output(proc.stdout, target.path)


def _parse_ssh_output(output: str, remote_path: str) -> list[dict[str, Any]]:
    """Parse SSH output containing separator-delimited metadata JSON."""
    if not output.strip():
        return []

    experiments = []
    chunks = output.split(_SSH_SEPARATOR)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            metadata = json.loads(chunk)
            # Ensure the metadata has an ID
            if "id" in metadata:
                experiments.append(metadata)
        except json.JSONDecodeError:
            # Skip corrupted metadata entries
            click.echo(
                "Warning: Skipping corrupted metadata entry on remote",
                err=True,
            )
            continue

    return experiments


def _fetch_s3_metadata(target: SyncTarget) -> list[dict[str, Any]]:
    """Fetch metadata from an S3 target by downloading metadata.json files."""
    check_tool("aws")

    with tempfile.TemporaryDirectory(prefix="yanex-sync-") as tmpdir:
        # Download all metadata.json files
        s3_path = target.s3_url + "/"
        cmd = [
            "aws",
            "s3",
            "cp",
            "--recursive",
            "--exclude",
            "*",
            "--include",
            "*/metadata.json",
            "--quiet",
            s3_path,
            tmpdir,
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise click.ClickException(
                f"S3 operation timed out while fetching metadata from {target.s3_url}"
            )
        except OSError as e:
            raise click.ClickException(f"Failed to access S3: {e}")

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise click.ClickException(
                f"Failed to fetch metadata from {target.s3_url}: {stderr}"
            )

        return _parse_s3_metadata_dir(Path(tmpdir))


def _parse_s3_metadata_dir(tmpdir: Path) -> list[dict[str, Any]]:
    """Parse metadata.json files downloaded from S3."""
    experiments = []

    for metadata_file in tmpdir.rglob("metadata.json"):
        # Extract experiment ID from parent directory name
        exp_id = metadata_file.parent.name
        if len(exp_id) != 8:
            continue

        try:
            metadata = json.loads(metadata_file.read_text())
            metadata["id"] = exp_id
            experiments.append(metadata)
        except (json.JSONDecodeError, OSError):
            click.echo(
                f"Warning: Skipping corrupted metadata for experiment {exp_id}",
                err=True,
            )
            continue

    return experiments
