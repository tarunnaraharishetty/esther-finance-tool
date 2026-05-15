"""HTTP/JSON face of the Esther controller.

This is the read-only API mirror of the Textual dashboard. The
``DashboardController`` produces a :class:`DashboardSnapshot` each tick;
:func:`create_app` exposes that snapshot as JSON over HTTP so a
future web / mobile / external client can render the same data the
TUI does without re-implementing the intelligence pipeline.

Phase 0 of the frontend/web roadmap. Localhost-only, no auth, no
streaming. Auth + SSE / WebSocket streaming arrive in later phases.
"""

from __future__ import annotations

from src.api.app import create_app

__all__ = ["create_app"]
