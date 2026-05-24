"""Per-user watchlist CRUD endpoints.

Three routes, all auth-required:

* ``GET    /api/watchlist`` — current user's watchlist.
* ``POST   /api/watchlist/symbols`` — append a symbol.
* ``DELETE /api/watchlist/symbols/{symbol}`` — drop a symbol.

Coupling to the snapshot pipeline
---------------------------------
The dashboard controller maintains a global "universe" of tracked
symbols (the rows in every snapshot). When a user adds a symbol that
the controller isn't already tracking, we call ``controller.add_symbol``
so the next snapshot tick produces a row for it. We do **not** prune
the controller's universe on delete — other users may still watch
that symbol, and a per-symbol refcount is overkill for v1. The
universe grows monotonically; future cleanup can run a sweep that
drops symbols no user watches.

Why client-side filtering, not server-side
------------------------------------------
The snapshot wire shape currently surfaces ``rows`` for every symbol
the controller tracks. Filtering server-side would require either
(a) per-user snapshot pipelines (expensive — N parallel scoring
loops) or (b) cloning the snapshot per-request and filtering rows
(adds latency to every request). For v1 the frontend filters the
shared snapshot down to ``user.watchlist`` symbols. Tradeoff: the
wire payload is "everyone's symbols" — fine at our scale.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api.auth import require_current_user
from src.data.user_store import User
from src.data.watchlist_store import (
    SymbolAlreadyWatched,
    WatchlistStore,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


# Cap how many symbols a single user can watch. Cheap protection
# against pathological inputs (someone scripting POST in a loop). 200
# is well above any reasonable trader's manual watchlist.
_MAX_SYMBOLS_PER_USER = 200

# Symbol-shape guardrail. Tickers are uppercase alphanumeric plus
# ``.`` (BRK.B) and ``-`` (BF-A). Tight enough to reject obvious
# garbage; loose enough not to gatekeep legitimate exchange tickers.
_SYMBOL_MAX_LEN = 12


# Module-level dependency alias. FastAPI introspects route signatures
# via ``get_type_hints()``, which can't see aliases declared inside a
# function body — that's why this lives at the top of the module.
CurrentUser = Annotated[User, Depends(require_current_user)]


class AddSymbolRequest(BaseModel):
    """Body for ``POST /api/watchlist/symbols``."""

    symbol: str = Field(min_length=1, max_length=_SYMBOL_MAX_LEN)

    @field_validator("symbol")
    @classmethod
    def _symbol_shape(cls, v: str) -> str:
        """Shape check. UPPERCASE alphanumeric plus ``.`` / ``-``.

        We don't validate against a known-symbol universe — users
        may genuinely want to track obscure tickers, and the analyzer
        gracefully reports "no data" for unknown ones.
        """
        normalized = v.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        for ch in normalized:
            if not (ch.isalnum() or ch in ".-"):
                raise ValueError(
                    "symbol may only contain letters, digits, '.', or '-'"
                )
        return normalized


def register_watchlist_routes(
    app: FastAPI,
    *,
    watchlist_store: WatchlistStore,
    on_symbol_added: Callable[[str], None] | None = None,
) -> None:
    """Attach the ``/api/watchlist`` routes.

    Args:
        app: FastAPI app to extend.
        watchlist_store: Bound :class:`WatchlistStore`.
        on_symbol_added: Optional hook invoked after a user adds a
            symbol. Wired by ``app.py`` to ``controller.add_symbol``
            so the snapshot pipeline starts tracking the new symbol.
            Hook failures are logged but don't fail the API call.
    """
    # Stash so other routes / tests can resolve without an import.
    app.state.watchlist_store = watchlist_store

    router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

    @router.get("")
    async def list_watchlist(user: CurrentUser) -> dict[str, Any]:
        """Return the current user's watchlist in display order."""
        entries = watchlist_store.list_for(user.id)
        return {
            "symbols": [e.symbol for e in entries],
            "entries": [e.to_wire() for e in entries],
        }

    @router.post("/symbols", status_code=201)
    async def add_symbol(
        body: AddSymbolRequest,
        user: CurrentUser,
    ) -> dict[str, Any]:
        """Append a symbol. 201 on success, 409 if already present,
        413 if the user is at the per-user symbol cap."""
        existing = watchlist_store.list_for(user.id)
        if len(existing) >= _MAX_SYMBOLS_PER_USER:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Watchlist is full ({_MAX_SYMBOLS_PER_USER} symbols). "
                    "Remove one before adding more."
                ),
            )
        try:
            entry = watchlist_store.add(user.id, body.symbol)
        except SymbolAlreadyWatched:
            raise HTTPException(
                status_code=409,
                detail=f"{body.symbol} is already on your watchlist.",
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        # Best-effort: ask the controller to start tracking the
        # symbol so the next snapshot tick produces a row. We don't
        # block the response on this — even if the hook fails the
        # symbol is on the user's list and shows up after the next
        # add elsewhere wakes the controller.
        if on_symbol_added is not None:
            try:
                on_symbol_added(entry.symbol)
            except Exception as exc:
                log.warning(
                    "watchlist.add.controller_hook_failed",
                    symbol=entry.symbol,
                    user_id=user.id,
                    error=str(exc),
                )

        log.info(
            "watchlist.add.ok",
            user_id=user.id,
            symbol=entry.symbol,
        )
        return {"entry": entry.to_wire()}

    @router.delete("/symbols/{symbol}", status_code=200)
    async def remove_symbol(
        symbol: str,
        user: CurrentUser,
    ) -> dict[str, Any]:
        """Drop a symbol. 200 with ``{removed: bool}``.

        Idempotent: removing a symbol not on the list returns
        ``{removed: false}`` rather than 404 — the desired end state
        (symbol absent) holds either way, and the frontend doesn't
        need to special-case it.
        """
        normalized = symbol.strip().upper()
        if not normalized:
            raise HTTPException(
                status_code=400, detail="symbol must not be empty"
            )
        removed = watchlist_store.remove(user.id, normalized)
        if removed:
            log.info(
                "watchlist.remove.ok",
                user_id=user.id,
                symbol=normalized,
            )
        return {"removed": removed, "symbol": normalized}

    app.include_router(router)


__all__ = ["register_watchlist_routes"]
