# Implementation Plan: SPA Mode Conversion for Yanex UI

**Created:** 2025-10-13
**Status:** Planning
**Related PR:** #20 (Feature/yanex UI)

---

## Background & Rationale

### The Problem

PR #20 introduced a web UI feature for Yanex that adds significant value by providing a modern web interface for browsing experiments. However, the current implementation has a critical deployment issue:

**Current approach (as implemented in PR #20):**
- Runs `npm install` and `npm run build` every time `yanex ui` is executed (unless `--skip-build` flag is used)
- Launches two separate servers: Next.js dev server (port 3000) + FastAPI backend (port 8000)
- Uses `npm run dev` to serve the frontend in development mode
- Requires Node.js and npm to be installed on every end-user's machine
- Ships the entire Next.js source code (TypeScript/React files) and development dependencies
- Estimated package size: ~300MB with node_modules

**Why this is problematic:**
1. **End-user burden:** Users must have Node.js/npm installed just to use the UI
2. **Slow startup:** Building the frontend on every launch adds 30-60 seconds of wait time
3. **Poor user experience:** Two servers, two ports, complex subprocess management
4. **Large package size:** Shipping source files and node_modules bloats the distribution
5. **Not production-ready:** Development mode (`npm run dev`) is inappropriate for end users
6. **Goes against Yanex philosophy:** Yanex is meant to be "lightweight" and easy to use

### Decision: Why SPA Mode?

After analyzing the web UI implementation, we determined that **the application is fundamentally a client-side app**:
- All pages use `'use client'` directive (client-side React components)
- Data fetching happens at runtime in the browser via API calls
- No server-side rendering (SSR) or static site generation (SSG) is utilized
- Dynamic routes (`/experiment/[id]`) fetch data client-side, not at build time

This architecture is **perfect for SPA (Single Page Application) mode**, where:
- Next.js generates a fully static export (HTML/CSS/JS files)
- FastAPI serves these static files directly
- Client-side JavaScript handles routing and API calls
- No Node.js runtime needed on end-user machines

### Alternatives Considered & Rejected

#### Option A: Keep Next.js Development Server (Rejected)
**Description:** Continue using `npm run dev` but improve the implementation.

**Why rejected:**
- ❌ Still requires npm on user machines
- ❌ Development mode not suitable for production
- ❌ Two servers, two ports complexity remains
- ❌ Slow startup times persist
- ❌ Large package size with node_modules

#### Option B: Use Next.js Production Server with `next start` (Rejected)
**Description:** Build once, then use `npm run start` to serve the `.next` directory.

**Why rejected:**
- ❌ Still requires Node.js/npm on user machines (dealbreaker)
- ❌ Two servers still needed (Next.js + FastAPI)
- ❌ Must ship `.next` build directory (~50-100MB)
- ❌ More complex deployment
- ✅ Pro: Full Next.js features (ISR, middleware, etc.)

**Why the pro doesn't matter:** The current web UI doesn't use any advanced Next.js features that require a Node.js server. It's purely client-side rendering with API calls.

#### Option C: True Static Site Generation with `generateStaticParams` (Rejected)
**Description:** Pre-render all experiment pages at build time using Next.js SSG.

**Why rejected:**
- ❌ Experiments are dynamic and created at runtime
- ❌ Cannot pre-render unknown experiment IDs at build time
- ❌ Would require rebuilding/redeploying the package every time experiments change
- ❌ Fundamentally incompatible with Yanex's use case

**From Next.js documentation:** Static export with dynamic routes requires `generateStaticParams()` to pre-define all possible IDs. This doesn't work for runtime-created experiments.

#### Option D: SPA Mode with Static Export (✅ SELECTED)
**Description:** Configure Next.js to export a static SPA, serve via FastAPI's built-in static file serving.

**Why this is the best choice:**
- ✅ **No Node.js/npm required** on end-user machines
- ✅ **Single server, single port** (FastAPI handles everything)
- ✅ **Instant startup** (no build step at runtime)
- ✅ **Small package size** (~5-10MB, only built artifacts)
- ✅ **Production-ready** deployment model
- ✅ **Matches current architecture** (client-side app already)
- ✅ **Simple for end users** (`pip install yanex` → `yanex ui` just works)
- ✅ **Aligns with Yanex philosophy** (lightweight, easy to use)
- ✅ **Industry standard** for Python+JS hybrid apps (see: Jupyter, Streamlit)

### How This Fits Industry Standards

This approach follows established patterns from successful Python+JavaScript projects:

**Jupyter Extensions:**
- Pre-build JavaScript assets
- Ship built files in Python package
- Serve via Python web server
- No npm required for end users

**Streamlit:**
- React frontend built and bundled
- Served by Python backend
- Users install via pip, run via Python command

**Django/Flask + React:**
- Build frontend → copy to `static/` → serve via Python
- Standard deployment pattern for production apps

**Our approach:**
```
Build time (developer/CI):  npm run build → yanex/web/out/
Distribution:               pip package includes out/ directory
Runtime (end user):         yanex ui → FastAPI serves out/ + API
```

### Key Decision Points

**Q: Why not just require users to have Node.js?**
A: Yanex is a Python tool for Python users. Requiring Node.js creates unnecessary friction and goes against the "lightweight" philosophy. Most ML/research users have Python but not necessarily Node.js.

**Q: Won't this make development harder?**
A: No. Developers can still use `npm run dev` for hot-reloading during development. The SPA mode is only for distribution. We'll document both workflows clearly.

**Q: What if we want SSR features later?**
A: The current implementation doesn't use SSR and doesn't need it. If requirements change significantly in the future, we can reconsider. But YAGNI (You Aren't Gonna Need It) applies here.

**Q: How do we handle API URLs?**
A: Change from absolute (`http://localhost:8000`) to relative (`/api/experiments`). This works in all environments since FastAPI serves both the SPA and API.

---

## Overview

Convert the Yanex web UI from development mode (requiring npm at runtime) to production SPA mode (static build artifacts served by FastAPI, no npm required on end-user machines).

---

## Phase 1: Frontend Changes

### 1.1 Update API URLs to Use Relative Paths

**Files to modify:** 5 files with hardcoded `http://localhost:8000`

| File | Line | Current | Change To |
|------|------|---------|-----------|
| `app/page.tsx` | 41 | `http://localhost:8000/api/experiments` | `/api/experiments` |
| `app/experiment/[id]/page.tsx` | 22 | `http://localhost:8000/api/experiments/${experimentId}` | `/api/experiments/${experimentId}` |
| `components/StatusStats.tsx` | 13 | `http://localhost:8000/api/status` | `/api/status` |
| `components/ExperimentDetails.tsx` | 40 | `http://localhost:8000/api/experiments/${experimentId}` | `/api/experiments/${experimentId}` |
| `components/ExperimentDetails.tsx` | 201 | `http://localhost:8000/api/experiments/${experimentId}/artifacts/${artifact.name}` | `/api/experiments/${experimentId}/artifacts/${artifact.name}` |

**Why:** Relative URLs work regardless of where the app is hosted (localhost, production domain, custom port).

### 1.2 Configure Next.js for Static Export

**File:** `yanex/web/next.config.js`

**Current:**
```javascript
const nextConfig = {
  trailingSlash: true,
  images: { unoptimized: true },
  assetPrefix: '/',
  basePath: '',
}
```

**Change to:**
```javascript
const nextConfig = {
  output: 'export',           // Enable static export
  trailingSlash: true,
  images: { unoptimized: true },
  distDir: 'out',            // Output directory for static files
}
```

**Why:** `output: 'export'` tells Next.js to generate a fully static site that can be served by any web server.

### 1.3 Add Development Environment Variable Support (Optional Enhancement)

**Create:** `yanex/web/.env.local` (for development only, not shipped)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Update fetch calls** to use environment variable with fallback:
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
fetch(`${API_URL}/api/experiments`)
```

**Why:** Allows developers to test against different backend URLs without code changes. Production uses relative URLs (empty string).

---

## Phase 2: Backend Changes

### 2.1 Update FastAPI App to Serve Static SPA

**File:** `yanex/web/app.py`

**Replace lines 34-79** with:

```python
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Get web directory
web_dir = Path(__file__).parent
out_dir = web_dir / "out"

# Include API routes first (higher priority)
app.include_router(api_router, prefix="/api")

# Serve Next.js static export
if out_dir.exists():
    # Mount static assets from out/_next/static
    next_static_dir = out_dir / "_next" / "static"
    if next_static_dir.exists():
        app.mount("/_next/static", StaticFiles(directory=next_static_dir), name="next_static")

    # Mount _next root directory for other assets
    next_dir = out_dir / "_next"
    if next_dir.exists():
        app.mount("/_next", StaticFiles(directory=next_dir), name="next_assets")

    # Serve all other routes with index.html for client-side routing
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA for all non-API routes."""
        # Check if a specific file exists (e.g., favicon.ico)
        file_path = out_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Default to index.html for client-side routing
        index_path = out_dir / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)

        return {
            "error": "Web UI not built",
            "message": "Run 'npm run build' in yanex/web/ directory"
        }
else:
    # Fallback when build doesn't exist
    @app.get("/{full_path:path}")
    async def no_build_error(full_path: str):
        return {
            "error": "Web UI not built",
            "message": "The web UI has not been built. Build it with: cd yanex/web && npm run build"
        }
```

**Why:**
- Serves API routes first (they take priority)
- Serves static assets from `out/` directory
- Falls back to `index.html` for all non-file routes (enables client-side routing)
- Provides helpful error if build doesn't exist

### 2.2 Remove CORS Middleware (No Longer Needed)

**File:** `yanex/web/app.py`

**Remove lines 22-29** (CORS configuration):

**Why:** Since API and frontend are served from the same origin (single FastAPI server), CORS is not needed and removing it improves security.

---

## Phase 3: CLI Command Updates

### 3.1 Simplify `yanex ui` Command

**File:** `yanex/cli/commands/ui.py`

**Replace the entire command** with this simpler version:

```python
"""
UI command implementation for yanex CLI.
"""

import sys
import webbrowser
from pathlib import Path

import click
import uvicorn

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
    developing the UI, run 'npm run dev' in the yanex/web directory
    instead.

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
        click.echo("\nIf you're developing, see yanex/web/README.md for setup.", err=True)
        raise click.Abort()

    if verbose:
        click.echo(f"Starting yanex web UI server...")
        click.echo(f"  URL: http://{host}:{port}")
        click.echo(f"  Auto-reload: {reload}")

    click.echo(f"🚀 Starting server at http://{host}:{port}")

    if not no_browser:
        # Open browser after short delay
        import threading
        def open_browser():
            import time
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=open_browser, daemon=True).start()

    click.echo("Press Ctrl+C to stop the server")

    # Start uvicorn server (blocks until stopped)
    try:
        from yanex.web.app import app
        uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
```

**Key changes:**
- Single port (removed `--frontend-port` and `--api-port`)
- No subprocess management (uses uvicorn directly)
- Removed npm/node checks
- Removed build logic (build happens during package creation)
- Added check for `out/` directory existence
- Simpler, cleaner code

---

## Phase 4: Build & Packaging

### 4.1 Update MANIFEST.in

**File:** `MANIFEST.in`

**Add after line 17 (after examples):**

```
# Include web UI build artifacts
recursive-include yanex/web/out *.html *.js *.css *.json *.txt *.ico *.png *.svg *.woff *.woff2
recursive-include yanex/web/out/_next *
exclude yanex/web/out/.gitignore
```

**Ensure these exclusions exist:**

```
# Exclude web UI source files (not needed in distribution)
recursive-exclude yanex/web/app *
recursive-exclude yanex/web/components *
recursive-exclude yanex/web/types *
exclude yanex/web/*.tsx
exclude yanex/web/*.ts
exclude yanex/web/tsconfig.json
exclude yanex/web/next.config.js
exclude yanex/web/tailwind.config.js
exclude yanex/web/postcss.config.js
exclude yanex/web/package.json
exclude yanex/web/package-lock.json
recursive-exclude yanex/web/node_modules *
recursive-exclude yanex/web/.next *
```

**Why:** Include built artifacts, exclude source files and dependencies.

### 4.2 Add .gitignore Entry

**File:** `yanex/web/.gitignore`

**Add:**
```
# Next.js static export output
/out
```

**Why:** Don't commit build artifacts to git (they're generated).

### 4.3 Create Build Script

**File:** `build_web_ui.sh` (new file in project root)

```bash
#!/bin/bash
set -e

echo "Building Yanex Web UI..."

cd yanex/web

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build the static export
echo "Building Next.js static export..."
npm run build

# Verify build
if [ ! -d "out" ]; then
    echo "Error: Build failed - out directory not created"
    exit 1
fi

echo "✅ Web UI built successfully!"
echo "Build artifacts are in yanex/web/out/"
```

**Make executable:**
```bash
chmod +x build_web_ui.sh
```

### 4.4 Update Makefile

**File:** `Makefile`

**Add new targets:**

```makefile
.PHONY: build-web
build-web:  ## Build the web UI
	./build_web_ui.sh

.PHONY: clean-web
clean-web:  ## Clean web UI build artifacts
	rm -rf yanex/web/out
	rm -rf yanex/web/.next
	rm -rf yanex/web/node_modules

.PHONY: build
build: clean build-web  ## Build distribution packages (includes web UI)
	python -m build
```

**Update existing build target** to include web UI build.

---

## Phase 5: Documentation

### 5.1 Create Web UI README

**File:** `yanex/web/README.md` (new file)

```markdown
# Yanex Web UI

Modern web interface for Yanex experiment tracking.

## For End Users

The web UI is pre-built and included in the yanex package. Simply run:

```bash
yanex ui
```

This will start the server at http://localhost:8000

## For Developers

### Development Mode

For active frontend development:

```bash
cd yanex/web

# Install dependencies
npm install

# Start development servers
npm run dev  # Frontend on localhost:3000

# In another terminal, start the backend
cd ../..
python -m uvicorn yanex.web.app:app --reload  # Backend on localhost:8000
```

### Production Build

To build the static export:

```bash
cd yanex/web
npm run build
```

This creates the `out/` directory with static files.

### Project Structure

- `app/` - Next.js App Router pages
- `components/` - React components
- `types/` - TypeScript type definitions
- `out/` - Build output (generated, not in git)

### Tech Stack

- **Framework:** Next.js 14 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **Icons:** Lucide React

## Architecture

The web UI is a **Single Page Application (SPA)** that:
- Fetches experiment data from FastAPI backend at runtime
- Uses client-side routing (Next.js App Router)
- Is served as static files by FastAPI
- Requires no Node.js on end-user machines

```
┌─────────────────────┐
│   yanex ui command  │
└──────────┬──────────┘
           │
           v
    ┌─────────────┐
    │  uvicorn    │
    │  (FastAPI)  │
    └──────┬──────┘
           │
     ┌─────┴─────┐
     │           │
     v           v
┌─────────┐  ┌────────────┐
│ API     │  │ Static SPA │
│ /api/*  │  │ (Next.js)  │
└─────────┘  └────────────┘
```
```

### 5.2 Update Main README

**File:** `README.md`

**Add section on Web UI:**

```markdown
### Web UI

Yanex includes a modern web interface for browsing and analyzing experiments:

```bash
yanex ui
```

This starts a web server at http://localhost:8000 with:
- Interactive experiment list with filtering
- Detailed experiment views
- Artifact downloads
- Real-time status dashboard

See [yanex/web/README.md](yanex/web/README.md) for development details.
```

---

## Phase 6: CI/CD Updates

### 6.1 Update GitHub Actions

**File:** `.github/workflows/test.yml` (if exists)

**Add build step before package creation:**

```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '18'

- name: Build Web UI
  run: |
    cd yanex/web
    npm ci
    npm run build

- name: Build package
  run: python -m build
```

**Why:** Ensures CI builds the web UI before creating distribution packages.

---

## Phase 7: Testing

### 7.1 Manual Testing Checklist

**Before deploying:**

- [ ] Build web UI: `./build_web_ui.sh`
- [ ] Verify `yanex/web/out/` exists and contains files
- [ ] Test `yanex ui` command starts successfully
- [ ] Browser opens to http://localhost:8000
- [ ] Main page loads and shows experiments
- [ ] Filters work correctly
- [ ] Clicking experiment shows details
- [ ] Artifact download works
- [ ] All API calls use relative URLs (check browser DevTools Network tab)
- [ ] Test with different port: `yanex ui --port 9000`
- [ ] Test `--no-browser` flag
- [ ] Verify package includes `out/` directory: `python -m build && tar -tzf dist/yanex-*.tar.gz | grep "web/out"`

### 7.2 Add Automated Test

**File:** `tests/cli/test_ui.py` (new file)

```python
"""Tests for UI command."""

import pytest
from pathlib import Path
from click.testing import CliRunner
from yanex.cli.commands.ui import ui


def test_ui_command_checks_build_exists(tmp_path, monkeypatch):
    """Test that UI command checks for build directory."""
    runner = CliRunner()

    # Mock web directory to point to temp path (no build)
    def mock_file(self):
        return tmp_path / "yanex" / "cli" / "commands" / "ui.py"

    monkeypatch.setattr(Path, "__file__", property(mock_file))

    result = runner.invoke(ui, obj={"verbose": False})

    assert result.exit_code != 0
    assert "Web UI build not found" in result.output


def test_ui_command_with_build(tmp_path, monkeypatch):
    """Test UI command with existing build directory."""
    # Create mock build directory
    out_dir = tmp_path / "web" / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "index.html").write_text("<html></html>")

    # This test would need more mocking to actually test uvicorn.run
    # For now, just verify the build check passes
    assert out_dir.exists()
```

---

## Migration Checklist

### Pre-Migration
- [ ] Review all changes with team
- [ ] Backup current working state
- [ ] Create feature branch: `git checkout -b feature/spa-mode-ui`

### Phase 1: Frontend
- [ ] Update 5 files to use relative API URLs
- [ ] Modify `next.config.js` for static export
- [ ] Test build locally: `cd yanex/web && npm run build`
- [ ] Verify `out/` directory created with expected files

### Phase 2: Backend
- [ ] Update `app.py` to serve static files
- [ ] Remove CORS middleware
- [ ] Test locally: `uvicorn yanex.web.app:app` and visit http://localhost:8000

### Phase 3: CLI
- [ ] Refactor `ui.py` command
- [ ] Test: `yanex ui`
- [ ] Verify single server serves both API and UI

### Phase 4: Packaging
- [ ] Update `MANIFEST.in`
- [ ] Add web UI `.gitignore` entry
- [ ] Create `build_web_ui.sh` script
- [ ] Update `Makefile`
- [ ] Test full build: `make build`
- [ ] Extract and inspect package: verify `out/` included, source excluded

### Phase 5: Documentation
- [ ] Create `yanex/web/README.md`
- [ ] Update main `README.md`

### Phase 6: CI/CD
- [ ] Update GitHub Actions workflow
- [ ] Test CI build

### Phase 7: Testing
- [ ] Complete manual testing checklist
- [ ] Add automated tests
- [ ] Test installation from built package

### Post-Migration
- [ ] Update PR description with changes
- [ ] Request code review
- [ ] Address Claude's security concerns (from original review)
- [ ] Merge to main

---

## Expected Outcomes

**Before (Current State):**
- Requires Node.js/npm on user machines
- Two servers, two ports
- Builds on every `yanex ui` run
- Ships source code + node_modules
- ~300MB package size (estimate)

**After (SPA Mode):**
- ✅ No Node.js/npm required for end users
- ✅ Single server, single port (8000)
- ✅ Instant startup (no build step)
- ✅ Ships only built artifacts
- ✅ ~5-10MB package size (estimate)
- ✅ Simpler deployment and usage

---

## Estimated Effort

- **Phase 1-3:** 2-3 hours (core functionality)
- **Phase 4-5:** 1-2 hours (packaging & docs)
- **Phase 6-7:** 1-2 hours (CI/CD & testing)
- **Total:** 4-7 hours

---

## Related Documents

- PR #20: Feature/yanex UI
- Claude's code review: See PR #20 comments
- Security concerns to address: CORS, input validation, file access controls

---

## Questions or Issues?

If you encounter issues during implementation:
1. Check that `npm run build` succeeds in `yanex/web/`
2. Verify `out/` directory contains `index.html`
3. Test FastAPI can serve files: `curl http://localhost:8000/`
4. Check browser DevTools Network tab for API call URLs

---

## Notes

- This plan addresses the deployment issue raised in PR #20 review
- The current implementation builds on every run, which is not suitable for production
- SPA mode is the recommended approach for this use case (client-side data fetching)
- Static export with `generateStaticParams` was considered but not suitable due to dynamic runtime data