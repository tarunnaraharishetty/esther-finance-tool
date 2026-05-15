# esther web

Read-only web mirror of the Esther dashboard. Subscribes to the
backend's `GET /api/stream` SSE feed and renders the snapshot as a
dense panel grid.

## Prerequisites

- Node 22+ / npm 10+
- The Python backend serving on `http://127.0.0.1:8000`:

```bash
# from the repo root
esther serve --mock                  # or with real Alpaca creds (no --mock)
```

## Run

```bash
cd web
npm install
npm run dev
```

Vite dev server runs on http://localhost:5173 and proxies `/api/*`
to the backend on `:8000`. Open the URL and look for the green
"live" pill in the header — that's your SSE connection.

## Build for production

```bash
npm run build
```

Outputs static assets to `web/dist/`. Drop them behind any static
host (or have the FastAPI app serve them — same-origin removes the
need for CORS).

## Scope notes

- Read-only. No mutation surface. Add-symbol / remove-symbol /
  view-flip would require new authenticated POST endpoints in the
  Python API, which is Phase 3 work.
- Stays thin on dependencies — react, react-dom, vite, typescript.
  No CSS frameworks, no state library; React's `useState` is enough
  for the snapshot tree.
- Layout is information-dense by design. Esther's positioning is
  trader workstation, not consumer app. The TUI keybindings and the
  spreadsheet feel are the brand.
