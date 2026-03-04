"""Tests for remote metadata fetching."""

import json
from unittest.mock import MagicMock, patch

import click
import pytest

from yanex.sync.remote_metadata import (
    _SSH_SEPARATOR,
    _parse_s3_metadata_dir,
    _parse_ssh_output,
    fetch_remote_metadata,
)
from yanex.sync.target import SyncProtocol, SyncTarget


class TestParseSSHOutput:
    """Tests for _parse_ssh_output()."""

    def test_single_experiment(self):
        metadata = {"id": "abc12345", "name": "test", "status": "completed"}
        output = f"{_SSH_SEPARATOR}\n{json.dumps(metadata)}\n"

        result = _parse_ssh_output(output, "~/.yanex/experiments/")
        assert len(result) == 1
        assert result[0]["id"] == "abc12345"

    def test_multiple_experiments(self):
        meta1 = {"id": "abc12345", "name": "exp1", "status": "completed"}
        meta2 = {"id": "def67890", "name": "exp2", "status": "failed"}
        output = (
            f"{_SSH_SEPARATOR}\n{json.dumps(meta1)}\n"
            f"{_SSH_SEPARATOR}\n{json.dumps(meta2)}\n"
        )

        result = _parse_ssh_output(output, "/path/")
        assert len(result) == 2
        assert result[0]["id"] == "abc12345"
        assert result[1]["id"] == "def67890"

    def test_empty_output(self):
        result = _parse_ssh_output("", "/path/")
        assert result == []

    def test_whitespace_only(self):
        result = _parse_ssh_output("   \n  \n  ", "/path/")
        assert result == []

    def test_corrupted_json_skipped(self):
        meta = {"id": "abc12345", "name": "valid", "status": "completed"}
        output = (
            f"{_SSH_SEPARATOR}\n{{invalid json}}\n"
            f"{_SSH_SEPARATOR}\n{json.dumps(meta)}\n"
        )

        result = _parse_ssh_output(output, "/path/")
        assert len(result) == 1
        assert result[0]["id"] == "abc12345"

    def test_metadata_without_id_skipped(self):
        meta_no_id = {"name": "no-id", "status": "completed"}
        meta_ok = {"id": "abc12345", "name": "ok", "status": "completed"}
        output = (
            f"{_SSH_SEPARATOR}\n{json.dumps(meta_no_id)}\n"
            f"{_SSH_SEPARATOR}\n{json.dumps(meta_ok)}\n"
        )

        result = _parse_ssh_output(output, "/path/")
        assert len(result) == 1
        assert result[0]["id"] == "abc12345"


class TestParseS3MetadataDir:
    """Tests for _parse_s3_metadata_dir()."""

    def test_single_experiment(self, tmp_path):
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()
        metadata = {"name": "test", "status": "completed"}
        (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        result = _parse_s3_metadata_dir(tmp_path)
        assert len(result) == 1
        assert result[0]["id"] == "abc12345"
        assert result[0]["name"] == "test"

    def test_multiple_experiments(self, tmp_path):
        for exp_id in ["abc12345", "def67890"]:
            exp_dir = tmp_path / exp_id
            exp_dir.mkdir()
            metadata = {"name": exp_id, "status": "completed"}
            (exp_dir / "metadata.json").write_text(json.dumps(metadata))

        result = _parse_s3_metadata_dir(tmp_path)
        assert len(result) == 2

    def test_invalid_id_length_skipped(self, tmp_path):
        # Valid 8-char ID
        valid_dir = tmp_path / "abc12345"
        valid_dir.mkdir()
        (valid_dir / "metadata.json").write_text(json.dumps({"name": "ok"}))

        # Invalid ID length
        invalid_dir = tmp_path / "short"
        invalid_dir.mkdir()
        (invalid_dir / "metadata.json").write_text(json.dumps({"name": "bad"}))

        result = _parse_s3_metadata_dir(tmp_path)
        assert len(result) == 1
        assert result[0]["id"] == "abc12345"

    def test_corrupted_json_skipped(self, tmp_path):
        exp_dir = tmp_path / "abc12345"
        exp_dir.mkdir()
        (exp_dir / "metadata.json").write_text("{invalid")

        result = _parse_s3_metadata_dir(tmp_path)
        assert len(result) == 0

    def test_empty_dir(self, tmp_path):
        result = _parse_s3_metadata_dir(tmp_path)
        assert result == []


class TestFetchRemoteMetadataSSH:
    """Tests for SSH metadata fetching."""

    @patch("yanex.sync.remote_metadata.subprocess.run")
    @patch("yanex.sync.remote_metadata.check_tool")
    def test_success(self, _mock_check, mock_run):
        metadata = {"id": "abc12345", "name": "test", "status": "completed"}
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"{_SSH_SEPARATOR}\n{json.dumps(metadata)}\n",
            stderr="",
        )

        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        result = fetch_remote_metadata(target)

        assert len(result) == 1
        assert result[0]["id"] == "abc12345"

    @patch("yanex.sync.remote_metadata.subprocess.run")
    @patch("yanex.sync.remote_metadata.check_tool")
    def test_ssh_with_user(self, _mock_check, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        target = SyncTarget(
            protocol=SyncProtocol.SSH, host="h", user="u", path="/data/"
        )
        fetch_remote_metadata(target)

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "u@h"

    @patch("yanex.sync.remote_metadata.subprocess.run")
    @patch("yanex.sync.remote_metadata.check_tool")
    def test_connection_failure(self, _mock_check, mock_run):
        mock_run.return_value = MagicMock(
            returncode=255, stdout="", stderr="Connection refused"
        )

        target = SyncTarget(protocol=SyncProtocol.SSH, host="bad-host", path="/")
        with pytest.raises(click.ClickException, match="Connection refused"):
            fetch_remote_metadata(target)

    @patch("yanex.sync.remote_metadata.subprocess.run")
    @patch("yanex.sync.remote_metadata.check_tool")
    def test_timeout(self, _mock_check, mock_run):
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired(cmd="ssh", timeout=30)

        target = SyncTarget(protocol=SyncProtocol.SSH, host="slow-host", path="/")
        with pytest.raises(click.ClickException, match="timed out"):
            fetch_remote_metadata(target)

    @patch("yanex.sync.remote_metadata.subprocess.run")
    @patch("yanex.sync.remote_metadata.check_tool")
    def test_empty_remote(self, _mock_check, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        target = SyncTarget(protocol=SyncProtocol.SSH, host="empty-host", path="/")
        result = fetch_remote_metadata(target)
        assert result == []
