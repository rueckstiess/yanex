"""Tests for yanex push CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestPushCommand:
    """Tests for the push command."""

    def setup_method(self):
        self.runner = create_cli_runner()

    def test_push_help(self):
        result = self.runner.invoke(cli, ["push", "--help"])
        assert result.exit_code == 0
        assert "Push experiments to a remote target" in result.output
        assert "TARGET" in result.output
        assert "--yes" in result.output

    def test_push_requires_target(self):
        result = self.runner.invoke(cli, ["push"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    @patch("yanex.sync.transport.shutil.which", return_value=None)
    @patch("yanex.cli.commands.push.find_experiments_by_filters")
    @patch("yanex.cli.commands.push.ExperimentFilter")
    def test_push_rsync_not_installed(self, mock_filter_cls, mock_find, _mock_which):
        mock_find.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"}
        ]
        mock_filter_cls.return_value = MagicMock(
            manager=MagicMock(
                storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
            )
        )

        result = self.runner.invoke(cli, ["push", "sky-dev", "--yes"])
        assert result.exit_code != 0
        assert "rsync" in result.output.lower()

    @patch("yanex.cli.commands.push.ExperimentFilter")
    @patch("yanex.cli.commands.push.find_experiments_by_filters")
    def test_push_no_experiments(self, mock_find, mock_filter_cls):
        mock_find.return_value = []

        result = self.runner.invoke(cli, ["push", "sky-dev", "--yes"])
        assert result.exit_code == 0
        assert "No experiments found" in result.output

    @patch("yanex.cli.commands.push.sync_experiments_push")
    @patch("yanex.cli.commands.push.ExperimentFilter")
    @patch("yanex.cli.commands.push.find_experiments_by_filters")
    def test_push_with_confirmation(self, mock_find, mock_filter_cls, mock_sync):
        """Without --yes, push should prompt for confirmation."""
        mock_find.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"}
        ]
        mock_filter_cls.return_value = MagicMock(
            manager=MagicMock(
                storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
            )
        )
        mock_sync.return_value = MagicMock(all_succeeded=True, success_count=1)

        # Deny confirmation
        result = self.runner.invoke(cli, ["push", "sky-dev"], input="n\n")
        assert "Push cancelled" in result.output

    @patch("yanex.cli.commands.push.sync_experiments_push")
    @patch("yanex.cli.commands.push.ExperimentFilter")
    @patch("yanex.cli.commands.push.find_experiments_by_filters")
    def test_push_with_yes_flag(self, mock_find, mock_filter_cls, mock_sync):
        mock_find.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"}
        ]
        mock_filter_cls.return_value = MagicMock(
            manager=MagicMock(
                storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
            )
        )
        mock_sync.return_value = MagicMock(all_succeeded=True, success_count=1)

        result = self.runner.invoke(cli, ["push", "sky-dev", "--yes"])
        assert result.exit_code == 0
        assert "Successfully pushed 1 experiment(s)" in result.output

    @patch("yanex.cli.commands.push.sync_experiments_push")
    @patch("yanex.cli.commands.push.ExperimentFilter")
    @patch("yanex.cli.commands.push.find_experiments_by_filters")
    def test_push_s3_target(self, mock_find, mock_filter_cls, mock_sync):
        mock_find.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"}
        ]
        mock_filter_cls.return_value = MagicMock(
            manager=MagicMock(
                storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
            )
        )
        mock_sync.return_value = MagicMock(all_succeeded=True, success_count=1)

        result = self.runner.invoke(
            cli, ["push", "s3://my-bucket/experiments", "--yes"]
        )
        assert result.exit_code == 0

        # Verify the target was parsed as S3
        call_args = mock_sync.call_args
        target = call_args[1].get("target") or call_args[0][1]
        from yanex.sync.target import SyncProtocol

        assert target.protocol == SyncProtocol.S3

    @patch("yanex.cli.commands.push.sync_experiments_push")
    @patch("yanex.cli.commands.push.ExperimentFilter")
    @patch("yanex.cli.commands.push.find_experiments_by_filters")
    def test_push_with_filters(self, mock_find, mock_filter_cls, mock_sync):
        mock_find.return_value = [
            {"id": "abc12345", "name": "train-v1", "status": "completed"}
        ]
        mock_filter_cls.return_value = MagicMock(
            manager=MagicMock(
                storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
            )
        )
        mock_sync.return_value = MagicMock(all_succeeded=True, success_count=1)

        result = self.runner.invoke(
            cli,
            ["push", "sky-dev", "-n", "train*", "-s", "completed", "--yes"],
        )
        assert result.exit_code == 0

        # Verify filters were passed to find_experiments_by_filters
        call_kwargs = mock_find.call_args[1]
        assert call_kwargs["name"] == "train*"
        assert call_kwargs["status"] == "completed"

    @patch("yanex.cli.commands.push.sync_experiments_push")
    @patch("yanex.cli.commands.push.ExperimentFilter")
    @patch("yanex.cli.commands.push.find_experiments_by_filters")
    def test_push_sync_failure_exits_1(self, mock_find, mock_filter_cls, mock_sync):
        mock_find.return_value = [
            {"id": "abc12345", "name": "test", "status": "completed"}
        ]
        mock_filter_cls.return_value = MagicMock(
            manager=MagicMock(
                storage=MagicMock(experiments_dir=Path("/tmp/experiments"))
            )
        )
        mock_sync.return_value = MagicMock(
            all_succeeded=False,
            success_count=0,
            failed_count=1,
            errors=["Connection refused"],
        )

        result = self.runner.invoke(cli, ["push", "sky-dev", "--yes"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output
