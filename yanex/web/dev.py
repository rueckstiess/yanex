#!/usr/bin/env python3
"""
Development server that runs both FastAPI backend and Next.js frontend.
"""

import argparse
import subprocess
import time
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the process."""
    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run both FastAPI backend and Next.js frontend development servers"
    )
    parser.add_argument(
        "--backend-port",
        "-b",
        type=int,
        default=8000,
        help="Port for the FastAPI backend (default: 8000)",
    )
    parser.add_argument(
        "--frontend-port",
        "-f",
        type=int,
        default=3000,
        help="Port for the Next.js frontend (default: 3000)",
    )
    args = parser.parse_args()

    web_dir = Path(__file__).parent

    print("Starting yanex web UI development server...")
    print("This will start both the FastAPI backend and Next.js frontend.")
    print("Press Ctrl+C to stop both servers.")
    print()

    # Start FastAPI backend
    print(f"Starting FastAPI backend on http://localhost:{args.backend_port}...")
    backend_cmd = f"python -m uvicorn yanex.web.app:app --host 127.0.0.1 --port {args.backend_port} --reload"
    backend_process = run_command(backend_cmd, cwd=web_dir.parent.parent)

    # Wait a moment for backend to start
    time.sleep(2)

    # Start Next.js frontend
    print(f"Starting Next.js frontend on http://localhost:{args.frontend_port}...")
    frontend_cmd = f"npm run dev -- -p {args.frontend_port}"
    frontend_process = run_command(frontend_cmd, cwd=web_dir)

    try:
        # Monitor both processes
        while True:
            # Check if backend is still running
            if backend_process.poll() is not None:
                print("Backend process stopped unexpectedly")
                break

            # Check if frontend is still running
            if frontend_process.poll() is not None:
                print("Frontend process stopped unexpectedly")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down servers...")

        # Terminate both processes
        backend_process.terminate()
        frontend_process.terminate()

        # Wait for them to terminate
        backend_process.wait()
        frontend_process.wait()

        print("Servers stopped.")


if __name__ == "__main__":
    main()
