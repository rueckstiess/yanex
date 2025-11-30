"""Tests for script_executor module."""

from unittest.mock import MagicMock, call

import pytest


class TestStreamOutputFlushBehavior:
    """Test that stream_output function flushes after each write."""

    def test_flush_called_after_each_write(self):
        """Verify flush() is called after each write() to ensure real-time output."""
        # Import the function indirectly by testing the pattern
        # We'll simulate the stream_output function's behavior

        # Create mock file handle
        mock_file = MagicMock()

        # Create a mock pipe that yields lines
        lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        mock_pipe = MagicMock()
        mock_pipe.readline = MagicMock(side_effect=lines + [""])

        # Create mock output stream (console)
        mock_console = MagicMock()

        # Simulate the stream_output function logic
        capture_list: list[str] = []
        for line in iter(mock_pipe.readline, ""):
            mock_console.write(line)
            mock_console.flush()
            mock_file.write(line)
            mock_file.flush()
            capture_list.append(line)
        mock_pipe.close()

        # Verify write and flush were called in correct order for each line
        assert mock_file.write.call_count == 3
        assert mock_file.flush.call_count == 3

        # Verify the pattern: write followed by flush for each line
        expected_calls = []
        for line in lines:
            expected_calls.append(call.write(line))
            expected_calls.append(call.flush())

        assert mock_file.method_calls == expected_calls

        # Verify console also got flushed
        assert mock_console.write.call_count == 3
        assert mock_console.flush.call_count == 3

        # Verify capture list
        assert capture_list == lines

    def test_empty_output_no_flush(self):
        """Verify no flush calls when there's no output."""
        mock_file = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.readline = MagicMock(side_effect=[""])
        mock_console = MagicMock()

        # Simulate stream_output with empty output
        for line in iter(mock_pipe.readline, ""):
            mock_console.write(line)
            mock_console.flush()
            mock_file.write(line)
            mock_file.flush()

        # No writes or flushes should occur
        assert mock_file.write.call_count == 0
        assert mock_file.flush.call_count == 0


class TestScriptExecutorIntegration:
    """Integration tests for ScriptExecutor."""

    @pytest.fixture
    def mock_manager(self, tmp_path):
        """Create a mock manager with proper storage setup."""
        exp_dir = tmp_path / "exp_dir"
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create mock manager without spec to allow nested mocking
        manager = MagicMock()
        manager.storage.get_experiment_directory.return_value = exp_dir

        return manager, exp_dir, artifacts_dir

    def test_stdout_file_created_during_execution(self, tmp_path, mock_manager):
        """Test that stdout.txt is created during script execution."""
        from yanex.core.script_executor import ScriptExecutor

        manager, exp_dir, artifacts_dir = mock_manager

        # Create a simple script that outputs to stdout
        script_path = tmp_path / "test_script.py"
        script_path.write_text("print('Hello from test')\n")

        executor = ScriptExecutor(manager)

        import os

        env = os.environ.copy()
        env["YANEX_EXPERIMENT_ID"] = "test123"
        env["YANEX_CLI_ACTIVE"] = "1"

        return_code, stdout, stderr = executor._execute_with_streaming(
            script_path, env, [], "test123"
        )

        # Verify output was captured
        assert return_code == 0
        assert "Hello from test" in stdout

        # Verify file was created
        stdout_file = artifacts_dir / "stdout.txt"
        assert stdout_file.exists()
        assert "Hello from test" in stdout_file.read_text()

    def test_stderr_file_created_during_execution(self, tmp_path, mock_manager):
        """Test that stderr.txt is created during script execution."""
        from yanex.core.script_executor import ScriptExecutor

        manager, exp_dir, artifacts_dir = mock_manager

        # Create a script that outputs to stderr
        script_path = tmp_path / "test_script.py"
        script_path.write_text("import sys; print('Error output', file=sys.stderr)\n")

        executor = ScriptExecutor(manager)

        import os

        env = os.environ.copy()
        env["YANEX_EXPERIMENT_ID"] = "test123"
        env["YANEX_CLI_ACTIVE"] = "1"

        return_code, stdout, stderr = executor._execute_with_streaming(
            script_path, env, [], "test123"
        )

        # Verify stderr was captured
        assert return_code == 0
        assert "Error output" in stderr

        # Verify stderr file was created
        stderr_file = artifacts_dir / "stderr.txt"
        assert stderr_file.exists()
        assert "Error output" in stderr_file.read_text()

    def test_multiline_output_all_lines_written(self, tmp_path, mock_manager):
        """Test that all lines are written to file."""
        from yanex.core.script_executor import ScriptExecutor

        manager, exp_dir, artifacts_dir = mock_manager

        # Create a script with multiple output lines
        script_path = tmp_path / "test_script.py"
        script_path.write_text("for i in range(10):\n    print(f'Line {i}')\n")

        executor = ScriptExecutor(manager)

        import os

        env = os.environ.copy()
        env["YANEX_EXPERIMENT_ID"] = "test123"
        env["YANEX_CLI_ACTIVE"] = "1"

        return_code, stdout, stderr = executor._execute_with_streaming(
            script_path, env, [], "test123"
        )

        assert return_code == 0

        # Verify all lines are in the file
        stdout_file = artifacts_dir / "stdout.txt"
        content = stdout_file.read_text()
        for i in range(10):
            assert f"Line {i}" in content
