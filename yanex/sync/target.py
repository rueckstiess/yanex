"""Target parsing for sync commands.

Parses target strings into structured SyncTarget objects.
Protocol is inferred from the target string:
- s3://bucket/prefix → S3
- [user@]host[:path] → SSH
"""

from dataclasses import dataclass
from enum import Enum


class SyncProtocol(Enum):
    SSH = "ssh"
    S3 = "s3"


DEFAULT_SSH_PATH = "~/.yanex/experiments/"


@dataclass(frozen=True)
class SyncTarget:
    """Parsed sync target."""

    protocol: SyncProtocol

    # SSH fields
    host: str | None = None
    user: str | None = None
    path: str | None = None

    # S3 fields
    bucket: str | None = None
    prefix: str | None = None

    @property
    def ssh_dest(self) -> str:
        """Build the SSH destination string (e.g. user@host:path)."""
        if self.protocol != SyncProtocol.SSH:
            raise ValueError("ssh_dest only valid for SSH targets")
        user_prefix = f"{self.user}@" if self.user else ""
        return f"{user_prefix}{self.host}:{self.path}"

    @property
    def s3_url(self) -> str:
        """Build the S3 URL (e.g. s3://bucket/prefix)."""
        if self.protocol != SyncProtocol.S3:
            raise ValueError("s3_url only valid for S3 targets")
        if self.prefix:
            return f"s3://{self.bucket}/{self.prefix}"
        return f"s3://{self.bucket}"

    @property
    def display_name(self) -> str:
        """Human-readable display name for the target."""
        if self.protocol == SyncProtocol.S3:
            return self.s3_url
        return self.ssh_dest


def parse_target(target_str: str) -> SyncTarget:
    """Parse a target string into a SyncTarget.

    Args:
        target_str: Target string, e.g. "sky-dev", "user@host:/path",
                    "s3://bucket/prefix"

    Returns:
        Parsed SyncTarget

    Raises:
        ValueError: If target string is empty or malformed
    """
    if not target_str or not target_str.strip():
        raise ValueError("Target cannot be empty")

    target_str = target_str.strip()

    if target_str.startswith("s3://"):
        return _parse_s3_target(target_str)
    return _parse_ssh_target(target_str)


def _parse_s3_target(target_str: str) -> SyncTarget:
    """Parse an S3 target string like s3://bucket/prefix."""
    # Remove s3:// prefix
    remainder = target_str[5:]
    if not remainder:
        raise ValueError("S3 target must include a bucket name: s3://bucket[/prefix]")

    # Split into bucket and prefix
    parts = remainder.split("/", 1)
    bucket = parts[0]
    if not bucket:
        raise ValueError("S3 target must include a bucket name: s3://bucket[/prefix]")

    prefix = parts[1].rstrip("/") if len(parts) > 1 and parts[1] else None

    return SyncTarget(
        protocol=SyncProtocol.S3,
        bucket=bucket,
        prefix=prefix,
    )


def _parse_ssh_target(target_str: str) -> SyncTarget:
    """Parse an SSH target string like [user@]host[:path]."""
    user = None
    path = DEFAULT_SSH_PATH

    # Extract user if present
    if "@" in target_str:
        user_part, remainder = target_str.split("@", 1)
        if not user_part:
            raise ValueError("SSH target has empty username before '@'")
        user = user_part
    else:
        remainder = target_str

    # Extract host and optional path
    if ":" in remainder:
        host, path_part = remainder.split(":", 1)
        if path_part:
            # Ensure path ends with /
            path = path_part if path_part.endswith("/") else path_part + "/"
        # If empty after colon (e.g. "host:"), use default path
    else:
        host = remainder

    if not host:
        raise ValueError("SSH target must include a hostname")

    return SyncTarget(
        protocol=SyncProtocol.SSH,
        host=host,
        user=user,
        path=path,
    )
