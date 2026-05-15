"""Tests for the FastAPI snapshot endpoint.

Wires :class:`MockDashboardController` into :func:`create_app` and
hits it with FastAPI's bundled :class:`TestClient`. No network and
no real Alpaca / Anthropic credentials required.

Phase 0 contract being verified:
- /api/health is cheap, never touches fetch_snapshot.
- /api/snapshot returns a JSON tree shaped like ``DashboardSnapshot``
  with enums serialized to their .value strings, datetimes as ISO 8601,
  and a stable top-level field set.
- The same controller instance produces a monotonically-rising tick
  count across requests (the controller is shared, not rebuilt per
  call).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController


def _build_client(watchlist: list[str] | None = None) -> tuple[TestClient, MockDashboardController]:
    """Return a TestClient bound to a fresh mock controller.

    Each test gets its own controller so tick counters and history
    trackers don't leak between cases.
    """
    # Explicit None check — falsy `[]` is a valid input that the empty-
    # watchlist test relies on.
    if watchlist is None:
        watchlist = ["AAPL", "MSFT", "NVDA"]
    controller = MockDashboardController(watchlist=watchlist, seed=7)
    app = create_app(controller)
    return TestClient(app), controller


def test_health_endpoint_is_cheap_and_does_not_fetch() -> None:
    """Health probe should not advance the controller's tick counter.

    Hitting /api/health twice in a row must leave the next snapshot
    starting from tick 1 (not 3) — proving the endpoint never called
    ``fetch_snapshot`` behind the scenes.
    """
    client, _ = _build_client()
    for _ in range(2):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["watchlist_size"] == 3

    snap = client.get("/api/snapshot").json()
    assert snap["tick"] == 1  # first fetch_snapshot call


def test_snapshot_endpoint_returns_dataclass_shape() -> None:
    """The snapshot JSON should mirror DashboardSnapshot's field set."""
    client, _ = _build_client()
    res = client.get("/api/snapshot")
    assert res.status_code == 200
    body: dict[str, Any] = res.json()

    expected_fields = {
        "tick",
        "rows",
        "events",
        "alerts",
        "recent_alerts",
        "signal_history",
        "opp_history",
        "ranked_opportunities",
        "pulse",
        "pulse_history",
        "pulse_evolution",
        "intraday_signal_history",
        "intraday_opp_history",
        "intraday_ranked_opportunities",
        "intraday_pulse",
        "intraday_pulse_history",
        "intraday_pulse_evolution",
        "session_store_status",
        "timestamp",
    }
    assert expected_fields.issubset(body.keys()), (
        f"missing fields: {expected_fields - body.keys()}"
    )

    # Row shape — one entry per watchlist symbol with enums string-encoded.
    assert len(body["rows"]) == 3
    row = body["rows"][0]
    assert row["symbol"] == "AAPL"
    assert isinstance(row["action"], str)
    assert row["action"] in {"buy", "sell", "hold"}
    assert isinstance(row["tier"], str)
    assert isinstance(row["confidence"], float)
    # Datetimes are ISO-8601 strings, not raw datetime objects.
    assert isinstance(row["timestamp"], str)
    assert "T" in row["timestamp"]


def test_snapshot_endpoint_advances_tick_across_requests() -> None:
    """Each call to /api/snapshot advances the controller's tick.

    The endpoint must hit the *same* controller instance the factory
    closed over — re-instantiating per request would reset tick to 1
    every time and break adaptive-refresh semantics for a streaming
    client built on top of this endpoint.
    """
    client, _ = _build_client()
    ticks = [client.get("/api/snapshot").json()["tick"] for _ in range(3)]
    assert ticks == [1, 2, 3]


def test_snapshot_carries_pre_computed_ranked_opportunities() -> None:
    """``ranked_opportunities`` is the API contract for "what's hot
    this tick" — frontend clients render the panel from this field
    without re-running ``rank_opportunities`` themselves.

    Each entry should carry the symbol, tier, composite_score, and
    the seven driver fields the TUI uses in its drilldown. We assert
    the structural shape; the ordering and threshold semantics are
    covered by the opportunities-ranker tests, not here.
    """
    client, _ = _build_client(watchlist=["AAPL", "MSFT", "NVDA", "TSLA", "SPY"])
    body = client.get("/api/snapshot").json()

    assert "ranked_opportunities" in body
    assert "intraday_ranked_opportunities" in body
    ranked = body["ranked_opportunities"]
    assert isinstance(ranked, list)
    # Mock controller produces enough motion that *some* symbol clears
    # the threshold; defensive against quiet ticks where it's empty.
    for opp in ranked:
        assert {
            "symbol",
            "tier",
            "composite_score",
            "profile",
            "rationale",
            "technical_alignment",
            "sentiment_alignment",
            "confidence_acceleration",
            "momentum_persistence",
            "unusual_activity",
            "reversal_strength",
            "signal_quality_score",
        }.issubset(opp.keys())
        assert isinstance(opp["composite_score"], float)
        # Tier should be a string enum value (post-jsonable_encoder).
        assert isinstance(opp["tier"], str)

    # Daily-only sessions (intraday feature off) leave the intraday
    # ranked field empty rather than absent.
    assert body["intraday_ranked_opportunities"] == []


def test_strongest_symbols_flows_through_to_api_when_promoter_fires() -> None:
    """Full controller -> API -> JSON path proof for STRONG tiers.

    The mock controller never produces STRONG_BUY / STRONG_SELL in
    realistic mock runs (the 5-tier promoter's gates are strict by
    design). We stub a controller that emits two STRONG rows + one
    BUY + one HOLD, hit ``/api/snapshot``, and assert the
    ``pulse.strongest_symbols`` field carries through with the exact
    ``[symbol, tier-display]`` pair shape the frontend expects.

    Verifies the contract the PulseCard STRONG-chip rendering depends
    on. Anti-spam invariant lives on the Python side: ``strongest_symbols``
    is empty when no row qualifies; this test proves the *positive*
    case round-trips correctly so the React panel knows when to surface
    chips at all.
    """
    from datetime import UTC, datetime

    from src.api import create_app
    from src.dashboard.controller import BaseController
    from src.dashboard.state import DashboardSnapshot, RecommendationRow
    from src.intelligence.pulse import compute_pulse
    from src.strategy.base import RecommendationTier, SignalAction
    from src.strategy.recommendation import RecommendationEngine

    now = datetime.now(UTC)

    def _row(sym: str, action: SignalAction, conf: float, tier: RecommendationTier) -> RecommendationRow:
        return RecommendationRow(
            symbol=sym,
            action=action,
            confidence=conf,
            combined_score=conf,
            technical_score=conf,
            sentiment_score=conf,
            rsi=0.5,
            macd=0.5,
            bollinger=0.5,
            last_price=100.0,
            num_news_articles=3,
            reasoning="",
            timestamp=now,
            tier=tier,
            signal_quality="high",
            stability="stable",
        )

    class _StubController(BaseController):
        """Minimal BaseController that emits a fixed snapshot.

        Avoids the mock controller's synthetic OHLCV pipeline so we
        can force STRONG tiers deterministically. compute_pulse is
        invoked the same way the real controllers do via
        _record_pulse, so the strongest_symbols computation is the
        production code path.
        """

        async def fetch_snapshot(self) -> DashboardSnapshot:
            rows = [
                _row("NVDA", SignalAction.BUY, 0.92, RecommendationTier.STRONG_BUY),
                _row("AAPL", SignalAction.BUY, 0.55, RecommendationTier.BUY),
                _row("TSLA", SignalAction.SELL, 0.88, RecommendationTier.STRONG_SELL),
                _row("SPY", SignalAction.HOLD, 0.15, RecommendationTier.HOLD),
            ]
            snap = DashboardSnapshot(tick=1, rows=rows, timestamp=now)
            snap.pulse = compute_pulse(snap)
            return snap

    controller = _StubController(watchlist=["NVDA", "AAPL", "TSLA", "SPY"], engine=RecommendationEngine())
    app = create_app(controller)
    with TestClient(app) as client:
        body = client.get("/api/snapshot").json()

    # 1. The field exists and is non-empty — promoter fired.
    pulse = body["pulse"]
    strong = pulse["strongest_symbols"]
    assert len(strong) == 2, f"expected 2 STRONG rows, got {strong}"

    # 2. Each entry is a [symbol, tier-display] pair (encoded as a list
    #    because tuples become lists through jsonable_encoder). The
    #    tightened TS type `Array<[string, string]>` consumes exactly
    #    this shape — a misuse like indexing [2] would be a compile
    #    error client-side.
    for entry in strong:
        assert isinstance(entry, list) and len(entry) == 2
        assert isinstance(entry[0], str) and isinstance(entry[1], str)

    # 3. The two STRONG symbols come through — NVDA + TSLA, with the
    #    tier-display strings that the React component splits on
    #    ("BUY" vs "SELL") to pick a chip color.
    by_symbol = dict(strong)
    assert by_symbol["NVDA"] == "STRONG BUY"
    assert by_symbol["TSLA"] == "STRONG SELL"
    # 4. The non-STRONG rows (AAPL BUY, SPY HOLD) are correctly filtered
    #    out by _compute_strongest_symbols.
    assert "AAPL" not in by_symbol
    assert "SPY" not in by_symbol


def test_snapshot_payload_is_json_roundtrippable() -> None:
    """The encoded payload must survive a JSON round-trip cleanly.

    FastAPI applies jsonable_encoder server-side; the raw bytes need to
    be parseable by any vanilla client (a future web frontend, curl,
    etc.) without surprises. We re-serialize after parsing to catch
    cases where a field encodes to something json.dumps can't handle.
    """
    client, _ = _build_client()
    res = client.get("/api/snapshot")
    assert res.status_code == 200
    body = res.json()
    # Round-trip — should not raise.
    json.dumps(body)


def test_app_exposes_openapi_metadata() -> None:
    """FastAPI auto-generates /openapi.json; we lean on this for future
    typed clients. Smoke-test that the schema includes both endpoints.
    """
    client, _ = _build_client()
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    assert "/api/health" in paths
    assert "/api/snapshot" in paths


def test_snapshot_endpoint_reflects_runtime_watchlist_mutation() -> None:
    """The API reads from a live controller instance, so a watchlist
    mutation between requests must be visible on the next snapshot
    without re-instantiating the app.

    This proves a future streaming web client can see add/remove
    actions take effect on the same connection. We only exercise
    removal here — the mock controller pre-builds its synthetic
    profiles from the initial watchlist, so add-after-construction
    is a controller limitation, not an API one.
    """
    client, ctrl = _build_client(watchlist=["AAPL", "MSFT", "NVDA"])
    first = client.get("/api/snapshot").json()
    assert {row["symbol"] for row in first["rows"]} == {"AAPL", "MSFT", "NVDA"}

    assert ctrl.remove_symbol("MSFT") is True
    second = client.get("/api/snapshot").json()
    assert {row["symbol"] for row in second["rows"]} == {"AAPL", "NVDA"}
