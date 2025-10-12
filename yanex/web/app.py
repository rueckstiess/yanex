"""
FastAPI application for yanex web UI.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router as api_router

# Create FastAPI app
app = FastAPI(
    title="Yanex Web UI",
    description="Web interface for yanex experiment tracking",
    version="0.4.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Get web directory from environment or use default
web_dir = Path(os.environ.get("YANEX_WEB_BUILD_PATH", Path(__file__).parent))
next_build_dir = web_dir / ".next"
next_static_dir = next_build_dir / "static"
next_server_dir = next_build_dir / "server"
next_public_dir = web_dir / "public"

# Serve static assets from .next/static
if next_static_dir.is_dir():
    app.mount(
        "/_next/static", StaticFiles(directory=next_static_dir), name="next_static"
    )

# Serve public assets
if next_public_dir.is_dir():
    app.mount("/public", StaticFiles(directory=next_public_dir), name="public_static")


# Serve the Next.js app for all non-API routes
@app.get("/{path:path}")
async def serve_react_app(path: str):
    """Serve the React app for all non-API routes."""
    # Check if we have a built Next.js app
    if next_build_dir.exists():
        # Try to serve the built Next.js files
        index_html_path = next_server_dir / "app" / "index.html"  # For app router
        if not index_html_path.is_file():
            index_html_path = (
                next_build_dir / "server" / "pages" / "index.html"
            )  # For pages router

        if index_html_path.is_file():
            return FileResponse(index_html_path)

        # Fallback for client-side routing in development or if index.html is not found
        fallback_path = web_dir / "index.html"
        if fallback_path.is_file():
            return FileResponse(fallback_path)
        else:
            return {
                "message": "Next.js app not built. Run 'yanex ui' to build and start the server."
            }
    else:
        return {
            "message": "Next.js app not built. Run 'yanex ui' to build and start the server."
        }
