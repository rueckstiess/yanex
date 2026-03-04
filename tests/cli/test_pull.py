"""Tests for yanex pull CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestPullCommand:
    """Tests for the pull command."""

    def setup_method(self):
        self.runner = create_cli_runner()

    def test_pull_help(self):
        result = self.runner.invoke(cli, ["pull", "--help"])
        assert result.exit_code == 0
        assert "Pull experiments from a remote target" in result.output
        assert "TARGET" in result.output
        assert "--yes" in result.output

    def test_pull_requires_target(self):
        result = self.runner.invoke(cli, ["pull"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    @patch("yanex.cli.commands.pull.fetch_remote_metadata")
    @patch("yanex.cli.commands.pull.parse_target")
    def test_pull_empty_remote(self, mock_parse, mock_fetch):
        from yanex.sync.target import SyncProtocol, SyncTarget

        mock_parse.return_value = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        mock_fetch.return_value = []

        result = self.runner.invoke(cli, ["pull", "sky-dev", "--yes"])
        assert result.exit_code == 0
        assert "No experiments found on remote" in result.output

    @patch("yanex.cli.commands.pull.ExperimentManager")
    @patch("yanex.cli.commands.pull.sync_experiments_pull")
    @patch("yanex.cli.commands.pull.fetch_remote_metadata")
    @patch("yanex.cli.commands.pull.parse_target")
    def test_pull_all_experiments(self, mock_parse, mock_fetch, mock_sync, mock_mgr):
        from yanex.sync.target import SyncProtocol, SyncTarget

        mock_parse.return_value = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        mock_fetch.return_value = [
            {"id": "abc12345", "name": "exp1", "status": "completed"},
            {"id": "def67890", "name": "exp2", "status": "completed"},
        ]
        mock_mgr.return_value = MagicMock(
            storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
        )
        mock_sync.return_value = MagicMock(all_succeeded=True, success_count=1)

        result = self.runner.invoke(cli, ["pull", "sky-dev", "--yes"])
        assert result.exit_code == 0
        assert "Successfully pulled" in result.output

    @patch("yanex.cli.commands.pull.ExperimentManager")
    @patch("yanex.cli.commands.pull.sync_experiments_pull")
    @patch("yanex.cli.commands.pull.filter_experiment_dicts")
    @patch("yanex.cli.commands.pull.fetch_remote_metadata")
    @patch("yanex.cli.commands.pull.parse_target")
    def test_pull_with_name_filter(
        self, mock_parse, mock_fetch, mock_filter, mock_sync, mock_mgr
    ):
        from yanex.sync.target import SyncProtocol, SyncTarget

        mock_parse.return_value = SyncTarget(
            protocol=SyncProtocol.SSH, host="sky-dev", path="~/.yanex/experiments/"
        )
        mock_fetch.return_value = [
            {"id": "abc12345", "name": "train-v1", "status": "completed"},
            {"id": "def67890", "name": "eval-v1", "status": "completed"},
        ]
        mock_filter.return_value = [
            {"id": "abc12345", "name": "train-v1", "status": "completed"},
        ]
        mock_mgr.return_value = MagicMock(
            storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
        )
        mock_sync.return_value = MagicMock(all_succeeded=True, success_count=1)

        result = self.runner.invoke(cli, ["pull", "sky-dev", "-n", "train*", "--yes"])
        assert result.exit_code == 0

        # Verify filter was applied
        mock_filter.assert_called_once()
        call_kwargs = mock_filter.call_args[1]
        assert call_kwargs["name"] == "train*"

    @patch("yanex.cli.commands.pull.fetch_remote_metadata")
    @patch("yanex.cli.commands.pull.filter_experiment_dicts")
    @patch("yanex.cli.commands.pull.parse_target")
    def test_pull_no_matches_after_filter(self, mock_parse, mock_filter, mock_fetch):
        from yanex.sync.target import SyncProtocol, SyncTarget

        mock_parse.return_value = SyncTarget(
            protocol=SyncProtocol.SSH, host="h", path="/"
        )
        mock_fetch.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"},
        ]
        mock_filter.return_value = []

        result = self.runner.invoke(
            cli, ["pull", "sky-dev", "-n", "nonexistent*", "--yes"]
        )
        assert result.exit_code == 0
        assert "No experiments on remote match" in result.output

    @patch("yanex.cli.commands.pull.fetch_remote_metadata")
    @patch("yanex.cli.commands.pull.parse_target")
    def test_pull_with_confirmation_denied(self, mock_parse, mock_fetch):
        from yanex.sync.target import SyncProtocol, SyncTarget

        mock_parse.return_value = SyncTarget(
            protocol=SyncProtocol.SSH, host="h", path="/"
        )
        mock_fetch.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"},
        ]

        result = self.runner.invoke(cli, ["pull", "sky-dev"], input="n\n")
        assert "Pull cancelled" in result.output

    @patch("yanex.cli.commands.pull.ExperimentManager")
    @patch("yanex.cli.commands.pull.sync_experiments_pull")
    @patch("yanex.cli.commands.pull.fetch_remote_metadata")
    @patch("yanex.cli.commands.pull.parse_target")
    def test_pull_sync_failure(self, mock_parse, mock_fetch, mock_sync, mock_mgr):
        from yanex.sync.target import SyncProtocol, SyncTarget

        mock_parse.return_value = SyncTarget(
            protocol=SyncProtocol.SSH, host="h", path="/"
        )
        mock_fetch.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"},
        ]
        mock_mgr.return_value = MagicMock(
            storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
        )
        mock_sync.return_value = MagicMock(
            all_succeeded=False,
            success_count=0,
            failed_count=1,
            errors=["Permission denied"],
        )

        result = self.runner.invoke(cli, ["pull", "sky-dev", "--yes"])
        assert result.exit_code == 1
        assert "Permission denied" in result.output

    @patch("yanex.cli.commands.pull.ExperimentManager")
    @patch("yanex.cli.commands.pull.sync_experiments_pull")
    @patch("yanex.cli.commands.pull.fetch_remote_metadata")
    @patch("yanex.cli.commands.pull.parse_target")
    def test_pull_s3_target(self, mock_parse, mock_fetch, mock_sync, mock_mgr):
        from yanex.sync.target import SyncProtocol, SyncTarget

        mock_parse.return_value = SyncTarget(
            protocol=SyncProtocol.S3, bucket="my-bucket", prefix="experiments"
        )
        mock_fetch.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"},
        ]
        mock_mgr.return_value = MagicMock(
            storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
        )
        mock_sync.return_value = MagicMock(all_succeeded=True, success_count=1)

        result = self.runner.invoke(
            cli, ["pull", "s3://my-bucket/experiments", "--yes"]
        )
        assert result.exit_code == 0
