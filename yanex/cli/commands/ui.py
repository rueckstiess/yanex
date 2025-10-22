"""
UI command implementation for yanex CLI.
"""

import threading
import time
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
    "--port",
    default=8000,
    help="Port to bind the server to (default: 8000)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development (requires source files)",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't automatically open browser",
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def ui(
    ctx: click.Context,
    host: str,
    port: int,
    reload: bool,
    no_browser: bool,
) -> None:
    """
    Start the yanex web UI server.

    This command starts a FastAPI server that serves both the web UI
    and the API endpoints on a single port.

    The web UI must be built before running this command. If you're
    developing the UI, see yanex/web/README.md for development setup.

    Examples:

      # Start server with default settings
      yanex ui

      # Start with custom port
      yanex ui --port 8080

      # Start without opening browser
      yanex ui --no-browser

      # Development mode with auto-reload (requires source files)
      yanex ui --reload
    """
    verbose = ctx.obj.get("verbose", False)

    # Check if build exists
    web_dir = Path(__file__).parent.parent.parent / "web"
    out_dir = web_dir / "out"

    if not out_dir.exists():
        click.echo("⚠️  Web UI build not found.", err=True)
        click.echo(f"Expected build directory: {out_dir}", err=True)
        click.echo(
            "\nThe web UI needs to be built during package installation.",
            err=True,
        )
        click.echo(
            "If you're developing, run: cd yanex/web && npm run build",
            err=True,
        )
        raise click.Abort()

    if verbose:
        click.echo("Starting yanex web UI server...")
        click.echo(f"  URL: http://{host}:{port}")
        click.echo(f"  Auto-reload: {reload}")

    click.echo(f"🚀 Starting server at http://{host}:{port}")

    if not no_browser:
        # Open browser after short delay
        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    click.echo("Press Ctrl+C to stop the server")

    # Start uvicorn server (blocks until stopped)
    try:
        import uvicorn

        from yanex.web.app import app

        uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
