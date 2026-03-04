"""Tests for sync transport layer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yanex.sync.target import SyncProtocol, SyncTarget
from yanex.sync.transport import (
    SyncResult,
    check_tool,
    sync_experiments_pull,
    sync_experiments_push,
)


class TestCheckTool:
    """Tests for check_tool()."""

    def test_tool_found(self):
        with patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/rsync"):
            check_tool("rsync")  # Should not raise

    def test_tool_not_found_rsync(self):
        import click

        with patch("yanex.sync.transport.shutil.which", return_value=None):
            with pytest.raises(click.ClickException, match="rsync.*not installed"):
                check_tool("rsync")

    def test_tool_not_found_aws(self):
        import click

        with patch("yanex.sync.transport.shutil.which", return_value=None):
            with pytest.raises(click.ClickException, match="aws.*not installed"):
                check_tool("aws")


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_defaults(self):
        r = SyncResult()
        assert r.success_count == 0
        assert r.failed_count == 0
        assert r.errors == []
        assert r.total == 0
        assert r.all_succeeded

    def test_success(self):
        r = SyncResult(success_count=3)
        assert r.total == 3
        assert r.all_succeeded

    def test_failure(self):
        r = SyncResult(success_count=2, failed_count=1, errors=["oops"])
        assert r.total == 3
        assert not r.all_succeeded


class TestRsyncPush:
    """Tests for rsync push commands."""

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/rsync")
    @patch("yanex.sync.transport.subprocess.run")
    def test_push_builds_correct_command(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        local_dir = Path("/home/user/.yanex/experiments")

        result = sync_experiments_push(["abc12345", "def67890"], target, local_dir)

        assert result.all_succeeded
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "rsync"
        assert "-az" in cmd
        assert str(local_dir / "abc12345") in cmd
        assert str(local_dir / "def67890") in cmd
        assert "sky-dev:~/.yanex/experiments/" in cmd

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/rsync")
    @patch("yanex.sync.transport.subprocess.run")
    def test_push_with_user(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="gpu-box", user="root", path="/data/"
        )
        local_dir = Path("/home/user/.yanex/experiments")

        sync_experiments_push(["abc12345"], target, local_dir)

        cmd = mock_run.call_args[0][0]
        assert "root@gpu-box:/data/" in cmd

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/rsync")
    @patch("yanex.sync.transport.subprocess.run")
    def test_push_progress(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        target = SyncTarget(protocol=SyncProtocol.SSH, host="h", path="/")
        sync_experiments_push(["abc12345"], target, Path("/tmp"), progress=True)

        cmd = mock_run.call_args[0][0]
        assert "--progress" in cmd

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/rsync")
    @patch("yanex.sync.transport.subprocess.run")
    def test_push_failure(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=1, stderr="Connection refused")

        target = SyncTarget(protocol=SyncProtocol.SSH, host="h", path="/")
        result = sync_experiments_push(["abc12345"], target, Path("/tmp"))

        assert not result.all_succeeded
        assert result.failed_count == 1
        assert "Connection refused" in result.errors[0]


class TestRsyncPull:
    """Tests for rsync pull commands."""

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/rsync")
    @patch("yanex.sync.transport.subprocess.run")
    def test_pull_builds_correct_command(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        local_dir = Path("/home/user/.yanex/experiments")

        result = sync_experiments_pull(["abc12345", "def67890"], target, local_dir)

        assert result.all_succeeded
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "rsync"
        assert "-az" in cmd
        # Sources are remote paths
        assert "sky-dev:~/.yanex/experiments/abc12345" in cmd
        assert "sky-dev:~/.yanex/experiments/def67890" in cmd
        # Destination is local with trailing slash
        assert cmd[-1] == str(local_dir) + "/"

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/rsync")
    @patch("yanex.sync.transport.subprocess.run")
    def test_pull_with_user(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="h", user="u", path="/data/"
        )
        sync_experiments_pull(["abc12345"], target, Path("/tmp"))

        cmd = mock_run.call_args[0][0]
        assert "u@h:/data/abc12345" in cmd


class TestS3Push:
    """Tests for S3 push commands."""

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/aws")
    @patch("yanex.sync.transport.subprocess.run")
    def test_push_per_experiment(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        target = SyncTarget(
            protocol=SyncProtocol.S3, bucket="my-bucket", prefix="experiments"
        )
        local_dir = Path("/home/user/.yanex/experiments")

        result = sync_experiments_push(["abc12345", "def67890"], target, local_dir)

        assert result.success_count == 2
        assert mock_run.call_count == 2

        # Check first call
        cmd1 = mock_run.call_args_list[0][0][0]
        assert cmd1[:3] == ["aws", "s3", "sync"]
        assert str(local_dir / "abc12345") in cmd1
        assert "s3://my-bucket/experiments/abc12345" in cmd1

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/aws")
    @patch("yanex.sync.transport.subprocess.run")
    def test_push_partial_failure(self, mock_run, _mock_which):
        # First succeeds, second fails
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=1, stderr="Access Denied"),
        ]

        target = SyncTarget(protocol=SyncProtocol.S3, bucket="b", prefix="p")

        result = sync_experiments_push(["id1", "id2"], target, Path("/tmp"))

        assert result.success_count == 1
        assert result.failed_count == 1
        assert "Access Denied" in result.errors[0]


class TestS3Pull:
    """Tests for S3 pull commands."""

    @patch("yanex.sync.transport.shutil.which", return_value="/usr/bin/aws")
    @patch("yanex.sync.transport.subprocess.run")
    def test_pull_per_experiment(self, mock_run, _mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        target = SyncTarget(
            protocol=SyncProtocol.S3, bucket="my-bucket", prefix="experiments"
        )
        local_dir = Path("/home/user/.yanex/experiments")

        result = sync_experiments_pull(["abc12345"], target, local_dir)

        assert result.success_count == 1
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["aws", "s3", "sync"]
        assert "s3://my-bucket/experiments/abc12345" in cmd
        assert str(local_dir / "abc12345") in cmd
