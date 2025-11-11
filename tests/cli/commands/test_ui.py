"""
Tests for UI command functionality.

This module tests the UI server command that launches the web interface.
"""

from unittest.mock import Mock, patch

from yanex.cli.main import cli


def _create_path_mocks(build_exists=True):
    """Helper to create path mocks for UI command tests."""
    mock_out_dir = Mock()
    mock_out_dir.exists.return_value = build_exists

    mock_web_dir = Mock()
    mock_web_dir.__truediv__ = Mock(return_value=mock_out_dir)

    mock_grandparent = Mock()
    mock_grandparent.__truediv__ = Mock(return_value=mock_web_dir)

    mock_file = Mock()
    mock_file.parent.parent.parent = mock_grandparent

    return mock_file


class TestUICommandHelp:
    """Test UI command help and documentation."""

    def test_ui_help_output(self, cli_runner):
        """Test that UI command shows help information."""
        result = cli_runner.invoke(cli, ["ui", "--help"])
        assert result.exit_code == 0
        assert "Start the yanex web UI server" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output
        assert "--no-browser" in result.output

    def test_ui_help_shows_examples(self, cli_runner):
        """Test that help includes usage examples."""
        result = cli_runner.invoke(cli, ["ui", "--help"])
        assert result.exit_code == 0
        assert "Examples:" in result.output or "yanex ui" in result.output


class TestUICommandValidation:
    """Test UI command validation and error handling."""

    def test_ui_missing_build_directory_error(self, cli_runner):
        """Test error when web UI build directory is missing."""
        with patch("yanex.cli.commands.ui.Path") as mock_path_cls:
            mock_path_cls.return_value = _create_path_mocks(build_exists=False)

            result = cli_runner.invoke(cli, ["ui"])

            # Should abort with error message
            assert result.exit_code == 1
            assert "Web UI build not found" in result.output


class TestUIServerOptions:
    """Test UI server configuration options."""

    def test_ui_default_host_and_port(self, cli_runner):
        """Test UI starts with default host and port."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread"),
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["ui", "--no-browser"])

            assert result.exit_code == 0
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 8000
            assert call_kwargs["reload"] is False

    def test_ui_custom_host_and_port(self, cli_runner):
        """Test UI starts with custom host and port."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread"),
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(
                cli, ["ui", "--host", "0.0.0.0", "--port", "8080", "--no-browser"]
            )

            assert result.exit_code == 0
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080

    def test_ui_reload_flag(self, cli_runner):
        """Test UI starts with reload enabled."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread"),
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["ui", "--reload", "--no-browser"])

            assert result.exit_code == 0
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["reload"] is True

    def test_ui_verbose_output(self, cli_runner):
        """Test UI shows verbose output when --verbose flag is set."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread"),
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["--verbose", "ui", "--no-browser"])

            assert result.exit_code == 0
            assert "Starting yanex web UI server..." in result.output
            assert "http://127.0.0.1:8000" in result.output
            assert "Auto-reload:" in result.output


class TestUIBrowserLaunching:
    """Test browser launching behavior."""

    def test_ui_opens_browser_by_default(self, cli_runner):
        """Test that browser thread is created by default."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread") as mock_thread_cls,
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["ui"])

            assert result.exit_code == 0
            # Thread should be created for browser opening
            mock_thread_cls.assert_called_once()
            call_kwargs = mock_thread_cls.call_args[1]
            assert call_kwargs["daemon"] is True

    def test_ui_no_browser_flag(self, cli_runner):
        """Test that --no-browser flag prevents browser thread creation."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread") as mock_thread_cls,
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["ui", "--no-browser"])

            assert result.exit_code == 0
            # Thread should NOT be created when --no-browser is set
            mock_thread_cls.assert_not_called()


class TestUIServerLifecycle:
    """Test UI server startup and lifecycle."""

    def test_ui_starts_uvicorn_server(self, cli_runner):
        """Test that uvicorn server is started with correct parameters."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread"),
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["ui", "--no-browser"])

            assert result.exit_code == 0
            mock_uvicorn.run.assert_called_once()

            # Verify uvicorn is called with correct parameters
            call_kwargs = mock_uvicorn.run.call_args[1]
            assert "host" in call_kwargs
            assert "port" in call_kwargs
            assert "reload" in call_kwargs
            assert call_kwargs["log_level"] == "info"

    def test_ui_keyboard_interrupt_handled(self, cli_runner):
        """Test that KeyboardInterrupt is handled gracefully."""
        mock_uvicorn = Mock()
        mock_uvicorn.run.side_effect = KeyboardInterrupt()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread"),
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["ui", "--no-browser"])

            # Should handle gracefully
            assert result.exit_code == 0
            assert "Server stopped" in result.output

    def test_ui_shows_startup_messages(self, cli_runner):
        """Test that startup messages are displayed."""
        mock_uvicorn = Mock()
        mock_app = Mock()

        with (
            patch.dict(
                "sys.modules", {"uvicorn": mock_uvicorn, "yanex.web.app": mock_app}
            ),
            patch("yanex.cli.commands.ui.Path") as mock_path_cls,
            patch("yanex.cli.commands.ui.threading.Thread"),
        ):
            mock_path_cls.return_value = _create_path_mocks()

            result = cli_runner.invoke(cli, ["ui", "--no-browser"])

            assert result.exit_code == 0
            assert "Starting server at http://127.0.0.1:8000" in result.output
            assert "Press Ctrl+C to stop the server" in result.output
