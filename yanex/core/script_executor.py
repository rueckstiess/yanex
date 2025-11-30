"""Script execution module for Yanex experiments."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from yanex.core.manager import ExperimentManager


class ScriptExecutor:
    """Handles script execution for Yanex experiments with unified logic."""

    def __init__(self, manager: ExperimentManager):
        """Initialize the script executor.

        Args:
            manager: The experiment manager instance to use for operations.
        """
        self.manager = manager
        self.console = Console()  # Use stdout with colors for yanex messages

    def execute_script(
        self,
        experiment_id: str,
        script_path: Path,
        config: dict[str, Any],
        verbose: bool = False,
        script_args: list[str] | None = None,
    ) -> None:
        """Execute a script for an experiment with proper output handling.

        Args:
            experiment_id: The ID of the experiment to execute.
            script_path: Path to the script to execute.
            config: Configuration parameters for the script.
            verbose: Whether to show verbose output.
            script_args: Arguments to pass through to the script via sys.argv.

        Raises:
            click.Abort: If the script execution fails.
            KeyboardInterrupt: If execution is interrupted by user.
        """
        if script_args is None:
            script_args = []

        # Print experiment ID immediately so user can track it
        exp_dir = self.manager.storage.get_experiment_directory(experiment_id)
        self.console.print(f"[green]✓ Experiment created: {experiment_id}[/]")
        self.console.print(f"[dim]  Directory: {exp_dir}[/]")

        try:
            # Prepare environment for subprocess
            env = self._prepare_environment(experiment_id, config)

            if verbose:
                self.console.print(f"[dim]Starting script execution: {script_path}[/]")

            # Execute script with real-time output streaming to both console and disk
            return_code, stdout_text, stderr_text = self._execute_with_streaming(
                script_path, env, script_args, experiment_id
            )

            # Output is already saved to disk in real-time, no need for separate save

            # Handle experiment result based on exit code
            self._handle_execution_result(
                experiment_id, return_code, stderr_text, verbose
            )

        except KeyboardInterrupt:
            self.manager.cancel_experiment(
                experiment_id, "Interrupted by user (Ctrl+C)"
            )
            self.console.print(
                f"[bright_red]✗ Experiment cancelled: {experiment_id}[/]"
            )
            raise

        except click.Abort:
            # click.Abort is intentional (from _handle_execution_result)
            # Error message already stored in metadata, just re-raise
            raise

        except Exception as e:
            self.manager.fail_experiment(experiment_id, f"Unexpected error: {str(e)}")
            self.console.print(f"[red]✗ Experiment failed: {experiment_id}[/]")
            self.console.print(f"[red]Error: {e}[/]")
            raise click.Abort() from e

    def _prepare_environment(
        self, experiment_id: str, config: dict[str, Any]
    ) -> dict[str, str]:
        """Prepare environment variables for script execution.

        Args:
            experiment_id: The experiment ID to set in environment.
            config: Configuration parameters to add as environment variables.

        Returns:
            Dictionary of environment variables.
        """
        env = os.environ.copy()
        env["YANEX_EXPERIMENT_ID"] = experiment_id
        env["YANEX_CLI_ACTIVE"] = "1"  # Mark as CLI context

        # Add parameters as environment variables
        for key, value in config.items():
            env[f"YANEX_PARAM_{key}"] = (
                json.dumps(value) if not isinstance(value, str) else value
            )

        # Add CLI arguments to environment (for yanex.get_cli_args())
        try:
            metadata = self.manager.get_experiment_metadata(experiment_id)
            cli_args = metadata.get("cli_args", {})
            if cli_args:
                env["YANEX_CLI_ARGS"] = json.dumps(cli_args)
        except Exception:
            # If we can't get CLI args, continue without them
            pass

        return env

    def _execute_with_streaming(
        self,
        script_path: Path,
        env: dict[str, str],
        script_args: list[str],
        experiment_id: str,
    ) -> tuple[int, str, str]:
        """Execute script with real-time output streaming to console and disk.

        Args:
            script_path: Path to the script to execute.
            env: Environment variables for the subprocess.
            script_args: Arguments to pass through to the script via sys.argv.
            experiment_id: The experiment ID for saving output artifacts.

        Returns:
            Tuple of (return_code, stdout_text, stderr_text).
        """
        stdout_capture: list[str] = []
        stderr_capture: list[str] = []

        # Get artifact paths for real-time streaming
        exp_dir = self.manager.storage.get_experiment_directory(experiment_id)
        artifacts_dir = exp_dir / "artifacts"
        stdout_path = artifacts_dir / "stdout.txt"
        stderr_path = artifacts_dir / "stderr.txt"

        # Build command: python -u script.py [script_args...]
        cmd = [sys.executable, "-u", str(script_path.resolve())] + script_args

        def stream_output(
            pipe: Any,
            capture_list: list[str],
            output_stream: Any,
            file_handle: Any,
        ) -> None:
            """Stream output line by line to console and disk."""
            for line in iter(pipe.readline, ""):
                # Display to console in real-time
                output_stream.write(line)
                output_stream.flush()
                # Write to disk in real-time
                file_handle.write(line)
                file_handle.flush()
                # Capture for error handling
                capture_list.append(line)
            pipe.close()

        # Open files for real-time writing before starting process
        with (
            open(stdout_path, "w", encoding="utf-8") as stdout_file,
            open(stderr_path, "w", encoding="utf-8") as stderr_file,
        ):
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd(),
            )

            # Start threads for stdout and stderr streaming
            stdout_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, stdout_capture, sys.stdout, stdout_file),
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(process.stderr, stderr_capture, sys.stderr, stderr_file),
            )

            stdout_thread.start()
            stderr_thread.start()

            try:
                # Wait for process completion
                return_code = process.wait()

                # Wait for output threads to finish
                stdout_thread.join()
                stderr_thread.join()

                return return_code, "".join(stdout_capture), "".join(stderr_capture)

            except KeyboardInterrupt:
                # Terminate the process and wait for threads
                process.terminate()
                process.wait()
                stdout_thread.join()
                stderr_thread.join()
                raise

    def _save_output_artifacts(
        self, experiment_id: str, stdout_text: str, stderr_text: str
    ) -> None:
        """Save captured output as experiment artifacts.

        Args:
            experiment_id: The experiment ID to save artifacts for.
            stdout_text: Captured stdout content.
            stderr_text: Captured stderr content.
        """
        if stdout_text:
            self.manager.storage.save_text_artifact(
                experiment_id, "stdout.txt", stdout_text
            )
        if stderr_text:
            self.manager.storage.save_text_artifact(
                experiment_id, "stderr.txt", stderr_text
            )

    def _handle_execution_result(
        self,
        experiment_id: str,
        return_code: int,
        stderr_text: str,
        verbose: bool = False,
    ) -> None:
        """Handle the result of script execution.

        Args:
            experiment_id: The experiment ID.
            return_code: The script's exit code.
            stderr_text: Captured stderr content.
            verbose: Whether to show verbose output.

        Raises:
            click.Abort: If the script execution failed.
        """
        exp_dir = self.manager.storage.get_experiment_directory(experiment_id)

        if return_code == 0:
            self.manager.complete_experiment(experiment_id)
            self.console.print(
                f"[green]✓ Experiment completed successfully: {experiment_id}[/]"
            )
            self.console.print(f"[dim]  Directory: {exp_dir}[/]")
        else:
            error_msg = f"Script exited with code {return_code}"
            if stderr_text:
                error_msg += f": {stderr_text.strip()}"

            self.manager.fail_experiment(experiment_id, error_msg)
            self.console.print(f"[red]✗ Experiment failed: {experiment_id}[/]")
            self.console.print(f"[dim]  Directory: {exp_dir}[/]")
            self.console.print(f"[red]Error: {error_msg}[/]")
            raise click.Abort()
