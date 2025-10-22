# Yanex Web UI

Modern web interface for Yanex experiment tracking, built as a Single Page Application (SPA) using Next.js Pages Router.

## Architecture Overview

The web UI is a **pure client-side SPA** that:
- Renders entirely in the browser (no server-side rendering)
- Fetches experiment data from FastAPI backend at runtime via REST API
- Uses Next.js Pages Router for client-side navigation
- Exports to static HTML/CSS/JS files served by FastAPI
- Requires **no Node.js on end-user machines** - only during development/build

```
┌─────────────────┐
│  yanex ui       │  Single command
└────────┬────────┘
         │
         v
  ┌─────────────┐
  │  FastAPI    │  Single server, single port (8000)
  │  (uvicorn)  │
  └──────┬──────┘
         │
    ┌────┴─────┐
    │          │
    v          v
┌────────┐  ┌──────────────┐
│ API    │  │ Static SPA   │
│ /api/* │  │ (Next.js)    │
└────────┘  └──────────────┘
```

## For End Users

The web UI is **pre-built** and included in the yanex package. Simply run:

```bash
yanex ui
```

This starts the server at http://localhost:8000 with both the UI and API available.

## For Developers

### Quick Start

```bash
# Install dependencies
npm install

# Start development servers
npm run dev      # Frontend dev server on localhost:3000

# In another terminal, start the backend
cd ../..
python -m uvicorn yanex.web.app:app --reload  # Backend on localhost:8000
```

**Note:** In development mode, the frontend (port 3000) and backend (port 8000) run separately. The Next.js dev server automatically proxies API requests from `/api/*` to `http://localhost:8000/api/*`.

### Production Build

To build the static export (required before using `yanex ui`):

```bash
# From project root
./build_web_ui.sh

# Or using make
make build-web

# Or manually
cd yanex/web
npm run build
```

This creates the `out/` directory with static files ready to be served by FastAPI.

### Project Structure

```
yanex/web/
├── pages/              # Next.js Pages Router
│   ├── _app.tsx       # Custom App component (layout, global styles)
│   ├── _document.tsx  # Custom Document (HTML structure)
│   ├── index.tsx      # Home page (experiment list)
│   └── experiment/
│       └── [id].tsx   # Dynamic experiment detail page
├── components/        # Reusable React components
│   ├── ExperimentDetails.tsx
│   ├── ExperimentFilters.tsx
│   ├── ExperimentList.tsx
│   ├── Navbar.tsx
│   └── StatusStats.tsx
├── types/            # TypeScript type definitions
│   └── experiment.ts
├── styles/           # Global styles
│   └── globals.css   # Tailwind CSS + custom styles
├── out/              # Build output (generated, not in git)
└── next.config.js    # Next.js configuration
```

### Tech Stack

- **Framework:** Next.js 14 (Pages Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **Date Utilities:** date-fns
- **Icons:** Lucide React
- **Build Tool:** Next.js static export

### Development Workflow

#### 1. Making Changes

Edit files in `pages/`, `components/`, or `styles/`:

```bash
# Development server with hot reload
npm run dev
```

Visit http://localhost:3000 to see changes in real-time.

#### 2. Testing with Backend

The frontend needs the FastAPI backend for data:

```bash
# Terminal 1: Frontend
cd yanex/web
npm run dev

# Terminal 2: Backend
cd ../..
python -m uvicorn yanex.web.app:app --reload
```

Frontend (port 3000) will proxy `/api/*` requests to backend (port 8000).

#### 3. Building for Production

```bash
# Clean previous build
rm -rf yanex/web/out

# Build static export
cd yanex/web
npm run build

# Test production build
cd ../..
yanex ui --no-browser
curl http://localhost:8000/
```

### Key Differences: Development vs Production

| Aspect | Development | Production |
|--------|-------------|------------|
| Command | `npm run dev` | `yanex ui` |
| Ports | 3000 (frontend) + 8000 (backend) | 8000 (single server) |
| Hot Reload | ✅ Yes | ❌ No |
| Build Required | ❌ No | ✅ Yes |
| Node.js Required | ✅ Yes | ❌ No |

## API Integration

All API calls use **relative URLs** to work in both dev and prod:

```typescript
// ✅ Correct - works everywhere
const response = await fetch('/api/experiments')

// ❌ Wrong - hardcoded URL breaks in production
const response = await fetch('http://localhost:8000/api/experiments')
```

### Available API Endpoints

- `GET /api/status` - Server status and experiment counts
- `GET /api/experiments` - List experiments (with filters)
- `GET /api/experiments/{id}` - Get experiment details
- `GET /api/experiments/{id}/artifacts/{name}` - Download artifact

See `yanex/web/api.py` for full API documentation.

## Routing

### Client-Side Navigation

Use Next.js `Link` component for navigation:

```typescript
import Link from 'next/link'

<Link href="/experiment/abc123">View Details</Link>
```

This provides instant page transitions without full page reloads.

### Dynamic Routes

The `[id]` in `pages/experiment/[id].tsx` creates a dynamic route. Access the parameter with `useRouter`:

```typescript
import { useRouter } from 'next/router'

function ExperimentPage() {
  const router = useRouter()
  const { id } = router.query  // Get experiment ID from URL

  // Use id to fetch data...
}
```

## Styling

### Tailwind CSS

The UI uses Tailwind CSS utility classes:

```typescript
<button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
  Click me
</button>
```

### Custom Classes

Pre-defined classes in `styles/globals.css`:

```css
.card        /* White card with shadow and border */
.btn         /* Base button styles */
.btn-primary /* Primary blue button */
.btn-secondary /* Gray secondary button */
.status-badge /* Status badge base */
.status-{status} /* Status-specific colors */
```

Usage:

```typescript
<div className="card p-6">
  <button className="btn btn-primary">Save</button>
  <span className="status-badge status-completed">Completed</span>
</div>
```

## Common Tasks

### Adding a New Page

1. Create file in `pages/`:
   ```typescript
   // pages/mypage.tsx
   export default function MyPage() {
     return <div>My new page</div>
   }
   ```

2. Add navigation link:
   ```typescript
   <Link href="/mypage">Go to My Page</Link>
   ```

3. Rebuild:
   ```bash
   npm run build
   ```

### Adding a New Component

1. Create file in `components/`:
   ```typescript
   // components/MyComponent.tsx
   export function MyComponent() {
     return <div>Hello</div>
   }
   ```

2. Import and use:
   ```typescript
   import { MyComponent } from '@/components/MyComponent'

   <MyComponent />
   ```

### Adding an API Endpoint

API endpoints are in the FastAPI backend (`yanex/web/api.py`), not in Next.js:

1. Edit `yanex/web/api.py`:
   ```python
   @router.get("/my-endpoint")
   def my_endpoint():
       return {"message": "Hello"}
   ```

2. Call from frontend:
   ```typescript
   const response = await fetch('/api/my-endpoint')
   const data = await response.json()
   ```

### Updating Dependencies

```bash
# Update all dependencies
npm update

# Update specific package
npm install next@latest

# Check for outdated packages
npm outdated
```

## Troubleshooting

### Build fails with "Module not found"

**Problem:** TypeScript can't find a module.

**Solution:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Changes not appearing in dev mode

**Problem:** Hot reload not working.

**Solution:**
1. Stop dev server (Ctrl+C)
2. Clear Next.js cache: `rm -rf .next`
3. Restart: `npm run dev`

### Production build works but `yanex ui` shows errors

**Problem:** Backend can't find the `out/` directory.

**Solution:**
```bash
# Ensure build exists
ls yanex/web/out/index.html

# Rebuild if needed
./build_web_ui.sh
```

### API calls fail with 404 in development

**Problem:** Frontend can't reach API endpoints at `/api/*`.

**Solution:**
1. **Ensure backend is running** on port 8000:
   ```bash
   python -m uvicorn yanex.web.app:app --reload
   ```

2. **Check the proxy is configured** - `next.config.js` should have rewrites for development:
   ```javascript
   async rewrites() {
     if (process.env.NODE_ENV === 'development') {
       return [{ source: '/api/:path*', destination: 'http://localhost:8000/api/:path*' }]
     }
   }
   ```

3. **Restart dev server** after config changes:
   ```bash
   # Stop dev server (Ctrl+C) and restart
   npm run dev
   ```

4. **Verify backend is accessible**:
   ```bash
   curl http://localhost:8000/api/status
   ```

### Experiments not showing in development

**Problem:** Page loads but no experiments appear.

**Solution:**
This usually means the API proxy isn't working. See "API calls fail with 404" above.

**Check browser console:**
1. Open DevTools (F12)
2. Look for failed network requests to `/api/experiments`
3. If showing 404, the proxy isn't working
4. If showing CORS errors, backend isn't running

### Styling not applied

**Problem:** Tailwind classes not working.

**Solution:**
1. Check `tailwind.config.js` content paths
2. Restart dev server
3. Rebuild: `npm run build`

## Performance Tips

### Code Splitting

Next.js automatically code-splits by page. Keep pages focused to minimize bundle size.

### Dynamic Imports

For large components, use dynamic imports:

```typescript
import dynamic from 'next/dynamic'

const HeavyComponent = dynamic(() => import('@/components/HeavyComponent'), {
  loading: () => <p>Loading...</p>
})
```

### Image Optimization

Use Next.js Image component (already configured with `unoptimized: true` for static export):

```typescript
import Image from 'next/image'

<Image src="/logo.png" alt="Logo" width={200} height={100} />
```

## Testing

Currently, the web UI doesn't have automated tests. To add them:

```bash
# Install testing dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom jest

# Add test script to package.json
"scripts": {
  "test": "jest"
}
```

## Resources

### Next.js Documentation
- [Pages Router](https://nextjs.org/docs/pages)
- [Static Export](https://nextjs.org/docs/pages/guides/static-exports)
- [API Routes](https://nextjs.org/docs/pages/building-your-application/routing/api-routes)

### Tailwind CSS
- [Documentation](https://tailwindcss.com/docs)
- [Utility Classes](https://tailwindcss.com/docs/utility-first)

### React
- [React Hooks](https://react.dev/reference/react)
- [useRouter](https://nextjs.org/docs/pages/api-reference/functions/use-router)

### Project References
- Implementation plan: `../../planning/spa-mode-implementation.md`
- Backend API: `../api.py`
- CLI command: `../cli/commands/ui.py`

## Contributing

When making changes to the web UI:

1. **Test in development** (`npm run dev`)
2. **Build for production** (`npm run build`)
3. **Test production build** (`yanex ui`)
4. **Run linting** (if added)
5. **Update this README** if adding new features
6. **Commit changes** with descriptive message

For questions or issues, refer to the main project documentation or the implementation plan in `planning/spa-mode-implementation.md`.
