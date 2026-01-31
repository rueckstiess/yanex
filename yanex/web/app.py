"""
FastAPI application for yanex web UI.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router as api_router

# Create FastAPI app
app = FastAPI(
    title="Yanex Web UI",
    description="Web interface for yanex experiment tracking",
    version="0.6.0a3",
)

# Include API routes first (highest priority)
app.include_router(api_router, prefix="/api")

# Get web directory from environment or use default
web_dir = Path(os.environ.get("YANEX_WEB_BUILD_PATH", Path(__file__).parent))
out_dir = web_dir / "out"

# Serve static assets from out/_next (Next.js static export)
next_dir = out_dir / "_next"
if next_dir.is_dir():
    app.mount("/_next", StaticFiles(directory=next_dir), name="next_assets")


# Serve the Next.js SPA for all non-API, non-static routes
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    Serve the Next.js SPA.

    For the static export, we need to:
    1. Check if a specific HTML file exists for this route
    2. Otherwise fall back to index.html for client-side routing
    """
    # Check if it's a direct file request
    file_path = out_dir / full_path
    if file_path.is_file():
        return FileResponse(file_path)

    # Try to find a matching HTML file for this path
    # Next.js static export creates path/index.html for each route
    if full_path and not full_path.endswith(".html"):
        # Try with trailing slash
        potential_file = out_dir / full_path / "index.html"
        if potential_file.is_file():
            return FileResponse(potential_file)

        # Try exact .html file
        potential_file = out_dir / f"{full_path}.html"
        if potential_file.is_file():
            return FileResponse(potential_file)

    # Handle dynamic routes
    # For /experiment/[id] routes, serve the experiment template
    if full_path.startswith("experiment/") and "/" in full_path[10:]:
        experiment_template = out_dir / "experiment" / "[id]" / "index.html"
        if experiment_template.is_file():
            return FileResponse(experiment_template)

    # Default to root index.html for client-side routing
    index_html_path = out_dir / "index.html"
    if index_html_path.is_file():
        return FileResponse(index_html_path)
    else:
        return {
            "error": "Web UI not built",
            "message": (
                "The web UI has not been built. "
                "Build it with: cd yanex/web && npm run build"
            ),
        }
