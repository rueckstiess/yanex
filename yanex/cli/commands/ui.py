"""
UI command implementation for yanex CLI.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import click

from ..error_handling import CLIErrorHandler


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the server to (default: 127.0.0.1)",
)
@click.option(
    "--api-port",
    default=8000,
    help="Port to bind the API server to (default: 8000)",
)
@click.option(
    "--frontend-port",
    default=3000,
    help="Port to bind the frontend server to (default: 3000)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--skip-build",
    is_flag=True,
    help="Skip building the frontend (use existing build)",
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def ui(
    ctx: click.Context,
    host: str,
    api_port: int,
    frontend_port: int,
    reload: bool,
    skip_build: bool,
) -> None:
    """
    Start the yanex web UI with both frontend and backend servers.

    This command starts:
    - FastAPI backend server on port 8000 (API)
    - Next.js frontend server on port 3000 (UI)

    The frontend will automatically connect to the backend API.

    Examples:

      # Start both servers with default ports
      yanex ui

      # Start with custom ports
      yanex ui --api-port 8001 --frontend-port 3001

      # Start with auto-reload for development
      yanex ui --reload

      # Skip building frontend (use existing build)
      yanex ui --skip-build
    """
    verbose = ctx.obj.get("verbose", False)

    if verbose:
        click.echo("Starting yanex web UI...")
        click.echo(f"  API Server: http://{host}:{api_port}")
        click.echo(f"  Frontend: http://{host}:{frontend_port}")
        click.echo(f"  Auto-reload: {reload}")
        click.echo(f"  Skip build: {skip_build}")

    # Get the web directory path
    web_dir = Path(__file__).parent.parent.parent / "web"

    # Build the frontend if needed
    if not skip_build:
        click.echo("Building frontend...")
        try:
            # Check if Node.js is available
            try:
                subprocess.run(["node", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                click.echo("Error: Node.js is not installed or not in PATH", err=True)
                click.echo("Please install Node.js from https://nodejs.org/", err=True)
                raise click.Abort()

            # Check if npm is available
            try:
                subprocess.run(["npm", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                click.echo("Error: npm is not installed or not in PATH", err=True)
                click.echo("Please install npm (comes with Node.js)", err=True)
                raise click.Abort()

            # Change to web directory and install dependencies
            os.chdir(web_dir)

            # Install dependencies if needed
            if not (web_dir / "node_modules").exists():
                click.echo("Installing frontend dependencies...")
                result = subprocess.run(
                    ["npm", "install"], check=True, capture_output=True, text=True
                )
                if verbose:
                    click.echo(result.stdout)

            # Build the frontend
            click.echo("Building Next.js frontend...")
            result = subprocess.run(
                ["npm", "run", "build"], check=True, capture_output=True, text=True
            )
            if verbose:
                click.echo(result.stdout)

            click.echo("Frontend built successfully!")

        except subprocess.CalledProcessError as e:
            click.echo(f"Error building frontend: {e}", err=True)
            if e.stdout:
                click.echo(f"stdout: {e.stdout}", err=True)
            if e.stderr:
                click.echo(f"stderr: {e.stderr}", err=True)
            raise click.Abort()
        except Exception as e:
            click.echo(f"Unexpected error: {e}", err=True)
            raise click.Abort()

    # Start both servers
    try:
        click.echo(f"Starting API server at http://{host}:{api_port}")
        click.echo(f"Starting frontend server at http://{host}:{frontend_port}")
        click.echo("Press Ctrl+C to stop both servers")

        # Start FastAPI backend in a subprocess
        api_process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"import uvicorn; from yanex.web.app import app; uvicorn.run(app, host='{host}', port={api_port}, reload={reload})",
            ]
        )

        # Start Next.js frontend in a subprocess
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(frontend_port)], cwd=web_dir
        )

        webbrowser.open(f"http://localhost:{frontend_port}")

        # Wait for both processes
        try:
            api_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            click.echo("\nStopping servers...")
            api_process.terminate()
            frontend_process.terminate()

            # Wait a bit for graceful shutdown
            try:
                api_process.wait(timeout=5)
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                click.echo("Force killing processes...")
                api_process.kill()
                frontend_process.kill()

            click.echo("👋 Servers stopped")

    except Exception as e:
        click.echo(f"Error: Failed to start servers: {e}", err=True)
        # Clean up any running processes
        if "api_process" in locals():
            api_process.terminate()
        if "frontend_process" in locals():
            frontend_process.terminate()
        raise click.Abort()
