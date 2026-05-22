"""Targeted integration test for the validation wire shape on /api/research.

Doesn't exhaustively test the research endpoint — that's already
covered by the validator's own unit tests + the existing
endpoint-shape assertions elsewhere. This module pins one invariant:
``/api/research/{symbol}`` always emits a ``validation`` field with
the documented shape, even when the validator found nothing to drop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.dashboard.controller import MockDashboardController


def _build_client(tmp_path: Path) -> TestClient:
    controller = MockDashboardController(watchlist=["AAPL"], seed=7)
    app = create_app(
        controller,
        fundamentals_cache_dir=tmp_path / "fc",
        analyzer_cache_dir=tmp_path / "ac",
        research_cache_dir=tmp_path / "rc",
    )
    return TestClient(app)


def test_research_response_carries_validation_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wire contract must always include ``validation`` with the
    documented shape.

    The frontend keys off ``thesis.validation.drop_count`` for the
    badge; the field's *absence* would cause a tsc compile error in
    strict mode and a render crash at runtime. Pinning its presence
    here prevents a regression that would surface only as a white
    screen in the browser.

    Note we do *not* assert ``drop_count == 0`` — the template
    generator can legitimately emit numbers the corpus doesn't fully
    anchor (e.g., precision rounding off from the row's raw RSI). The
    contract is "the validator runs and reports", not "the template
    never trips it".
    """
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    res = client.get("/api/research/AAPL")
    assert res.status_code == 200, res.text

    body = res.json()
    assert "validation" in body, "validation field missing from research response"
    # Trust score rides on the same response. Frontend keys off
    # ``thesis.trust_score.grade`` for the badge; an absent field
    # would surface as a white-screen render crash in strict mode.
    assert "trust_score" in body, "trust_score field missing from research response"
    trust = body["trust_score"]
    assert set(trust.keys()) == {"score", "grade", "components"}
    assert trust["grade"] in {"A+", "A", "B", "C", "D", "F"}
    assert 0.0 <= trust["score"] <= 100.0
    validation = body["validation"]
    # Shape: drop_count + dropped_claims array.
    assert set(validation.keys()) == {"drop_count", "dropped_claims"}
    assert isinstance(validation["drop_count"], int)
    assert validation["drop_count"] >= 0
    assert isinstance(validation["dropped_claims"], list)
    assert len(validation["dropped_claims"]) == validation["drop_count"]
    # Each dropped claim carries the documented per-entry shape.
    for c in validation["dropped_claims"]:
        assert set(c.keys()) == {
            "section",
            "sentence",
            "unsupported_tokens",
        }


def test_dropped_claims_record_section_and_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the validator drops a claim it must record the *path* to
    the offending field — operators can't audit drops without it."""
    monkeypatch.chdir(tmp_path)
    client = _build_client(tmp_path)
    body = client.get("/api/research/AAPL").json()
    for claim in body["validation"]["dropped_claims"]:
        # Section paths look like "technical_analysis.body" or
        # "bull_thesis[0].detail" — neither should be empty.
        assert claim["section"], "empty section path on dropped claim"
        assert claim["sentence"], "empty sentence on dropped claim"
        # At least one unsupported token must have caused the drop.
        assert len(claim["unsupported_tokens"]) >= 1
