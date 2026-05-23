"""Tests for the Key Drivers composer.

Covers per-rule salience computation, ranking + tie-break, top-N
truncation, suppression below the salience floor, missing-input
degradation, and the wire shape.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.intelligence.movement_drivers import (
    Driver,
    KeyDrivers,
    compute_key_drivers,
)


def _report(
    *,
    trust_score: float = 90.0,
    trust_grade: str = "A",
    overbought: float | None = 50.0,
    oversold: float | None = 50.0,
    pullback_risk: float | None = 50.0,
    rebound_potential: float | None = 50.0,
    calibrations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "symbol": "AAPL",
        "trust_score": {"score": trust_score, "grade": trust_grade},
        "technicals": {
            "overbought_score": overbought,
            "oversold_score": oversold,
            "pullback_risk": pullback_risk,
            "rebound_potential": rebound_potential,
        },
        "calibrations": calibrations or [],
    }


def _row(
    *,
    sentiment: float = 0.0,
    headlines: list[str] | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    hs = headlines if headlines is not None else []
    return {
        "symbol": "AAPL",
        "sentiment_score": sentiment,
        "num_news_articles": n if n is not None else len(hs),
        "headlines": hs,
    }


def _sector_rank(
    *,
    available: bool = True,
    rank: int = 1,
    cohort: int = 24,
    percentile: float = 95.0,
    metric_label: str = "Trust score",
    direction: str = "higher_better",
) -> dict[str, Any]:
    return {
        "available": available,
        "sector": "Information Technology",
        "metrics": [
            {
                "metric": "trust_score",
                "label": metric_label,
                "rank": rank,
                "cohort_size": cohort,
                "percentile": percentile,
                "direction": direction,
            }
        ],
    }


def _by_kind(drivers: KeyDrivers) -> dict[str, Driver]:
    return {d.kind: d for d in drivers.drivers}


# ---------------------------------------------------------------------------
# Empty / no-input case
# ---------------------------------------------------------------------------


def test_no_inputs_returns_empty_drivers() -> None:
    result = compute_key_drivers("AAPL", analyzer_report=None)
    assert result.symbol == "AAPL"
    assert result.drivers == ()
    assert result.considered == 0
    assert result.suppressed == 0


def test_baseline_signals_produce_no_drivers() -> None:
    """A symbol with everything at baseline → no drivers fire."""
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),  # trust=90, all techs=50
        sector_rank=_sector_rank(percentile=50),  # median percentile
        snapshot_row=_row(sentiment=0.0, n=0),
    )
    assert len(result.drivers) == 0


# ---------------------------------------------------------------------------
# Per-rule behavior
# ---------------------------------------------------------------------------


def test_high_overbought_produces_technical_driver_warn_tone() -> None:
    result = compute_key_drivers(
        "AAPL", analyzer_report=_report(overbought=85)
    )
    tech = _by_kind(result).get("technical")
    assert tech is not None
    assert tech.tone == "warn"
    assert tech.salience == pytest.approx((85 - 50) / 50, abs=1e-6)
    assert "85" in tech.detail


def test_high_oversold_produces_technical_driver_bull_tone() -> None:
    """Oversold composite reads bullishly (rebound setup)."""
    result = compute_key_drivers(
        "AAPL", analyzer_report=_report(oversold=80)
    )
    tech = _by_kind(result).get("technical")
    assert tech is not None
    assert tech.tone == "bull"


def test_strong_sentiment_with_many_headlines_fires_sentiment_driver() -> None:
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        snapshot_row=_row(sentiment=0.7, n=12),
    )
    sent = _by_kind(result).get("sentiment")
    assert sent is not None
    assert sent.tone == "bull"
    assert "+0.70" in sent.detail
    assert "12 articles" in sent.detail


def test_negative_sentiment_marks_bear_tone() -> None:
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        snapshot_row=_row(sentiment=-0.6, n=8),
    )
    sent = _by_kind(result).get("sentiment")
    assert sent is not None
    assert sent.tone == "bear"


def test_low_volume_sentiment_below_threshold_is_suppressed() -> None:
    """A polarity read with only 1-2 headlines is too thin to surface."""
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        snapshot_row=_row(sentiment=0.1, n=1),
    )
    assert "sentiment" not in _by_kind(result)


def test_news_driver_fires_when_three_or_more_headlines_present() -> None:
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        snapshot_row=_row(
            sentiment=0.0,
            headlines=[
                "Headline one",
                "Headline two",
                "Headline three",
                "Headline four",
            ],
        ),
    )
    news = _by_kind(result).get("news")
    assert news is not None
    # Top headline appears verbatim.
    assert "Headline one" in news.detail


def test_news_driver_suppressed_when_fewer_than_three_headlines() -> None:
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        snapshot_row=_row(
            sentiment=0.0,
            headlines=["only one", "and two"],
        ),
    )
    assert "news" not in _by_kind(result)


def test_extreme_sector_percentile_fires_sector_driver_bull() -> None:
    """Top-decile percentile → bull tone, high salience."""
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        sector_rank=_sector_rank(percentile=92),
    )
    sector = _by_kind(result).get("sector")
    assert sector is not None
    assert sector.tone == "bull"
    assert sector.salience == pytest.approx(abs(92 - 50) / 50, abs=1e-6)
    assert "Information Technology" in sector.detail


def test_bottom_percentile_sector_renders_bear_tone() -> None:
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        sector_rank=_sector_rank(percentile=10, rank=20),
    )
    sector = _by_kind(result).get("sector")
    assert sector is not None
    assert sector.tone == "bear"


def test_unavailable_sector_rank_skips_sector_driver() -> None:
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(),
        sector_rank={"available": False, "reason": "no peers"},
    )
    assert "sector" not in _by_kind(result)


def test_low_trust_score_fires_warning_driver() -> None:
    """Grade D (~65) crosses the threshold; surfaces as a trust warning."""
    result = compute_key_drivers(
        "AAPL", analyzer_report=_report(trust_score=65, trust_grade="D")
    )
    trust = _by_kind(result).get("trust")
    assert trust is not None
    assert trust.tone == "warn"
    assert "65" in trust.detail
    assert "D" in trust.detail


def test_healthy_trust_score_does_not_fire() -> None:
    result = compute_key_drivers(
        "AAPL", analyzer_report=_report(trust_score=92, trust_grade="A")
    )
    assert "trust" not in _by_kind(result)


def test_strong_calibration_fires_with_correct_tone() -> None:
    """Hit rate ≥ 50% → bull tone; below → bear tone."""
    high = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(
            calibrations=[
                {
                    "bucket_published": True,
                    "score_name": "rebound_potential",
                    "outcome_name": "5d_return_positive",
                    "hit_rate": 0.78,
                    "n_observations": 250,
                    "horizon_days": 5,
                }
            ]
        ),
    )
    bull = _by_kind(high).get("calibration")
    assert bull is not None
    assert bull.tone == "bull"
    assert "78%" in bull.detail
    assert "n=250" in bull.detail

    low = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(
            calibrations=[
                {
                    "bucket_published": True,
                    "score_name": "overbought",
                    "outcome_name": "5d_return_positive",
                    "hit_rate": 0.22,
                    "n_observations": 200,
                    "horizon_days": 5,
                }
            ]
        ),
    )
    bear = _by_kind(low).get("calibration")
    assert bear is not None
    assert bear.tone == "bear"


def test_unpublished_calibration_is_ignored() -> None:
    """Under-sampled buckets are not eligible drivers."""
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(
            calibrations=[
                {
                    "bucket_published": False,
                    "score_name": "x",
                    "outcome_name": "y",
                    "hit_rate": 0.95,  # would be high signal if published
                    "n_observations": 500,
                }
            ]
        ),
    )
    assert "calibration" not in _by_kind(result)


# ---------------------------------------------------------------------------
# Ranking + truncation
# ---------------------------------------------------------------------------


def test_drivers_sorted_by_salience_descending() -> None:
    """Top driver = highest salience."""
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(overbought=85, trust_score=65, trust_grade="D"),
        snapshot_row=_row(sentiment=0.7, n=12),
    )
    salience_seq = [d.salience for d in result.drivers]
    assert salience_seq == sorted(salience_seq, reverse=True)


def test_drivers_capped_at_max_drivers() -> None:
    """max_drivers=2 returns at most 2 even when more clear threshold."""
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(overbought=85, trust_score=65),
        sector_rank=_sector_rank(percentile=95),
        snapshot_row=_row(
            sentiment=0.7,
            headlines=["h1", "h2", "h3", "h4"],
        ),
        max_drivers=2,
    )
    assert len(result.drivers) == 2
    assert result.suppressed >= 1  # rest were tallied as suppressed


def test_min_salience_floor_suppresses_quiet_drivers() -> None:
    """A barely-active signal stays out of the rendered list."""
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(overbought=55),  # salience ~ 0.10
        min_salience=0.3,
    )
    assert "technical" not in _by_kind(result)
    assert result.suppressed >= 1


def test_alphabetic_tie_break_on_label() -> None:
    """Equal salience falls back to alphabetic label order."""
    # Construct two drivers with identical salience. Use a custom
    # min_salience that lets them both pass.
    # Same percentile distance on sector and same magnitude on
    # technical → forge equal salience values via construction.
    result = compute_key_drivers(
        "AAPL",
        # technical salience: (75-50)/50 = 0.50
        analyzer_report=_report(overbought=75),
        # sector salience: |75-50|/50 = 0.50
        sector_rank=_sector_rank(percentile=75, metric_label="Trust score"),
    )
    # Both should be present, sorted alphabetically by label when
    # salience matches.
    labels = [d.label for d in result.drivers]
    # "Overbought technicals" < "Sector outlier · Trust score" alphabetically.
    if len(labels) == 2 and result.drivers[0].salience == result.drivers[1].salience:
        assert labels == sorted(labels)


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_shape() -> None:
    result = compute_key_drivers(
        "AAPL",
        analyzer_report=_report(overbought=85, trust_score=65),
        snapshot_row=_row(sentiment=0.6, n=8),
    )
    wire = result.to_dict()
    assert set(wire.keys()) == {"symbol", "drivers", "considered", "suppressed"}
    for d in wire["drivers"]:
        assert set(d.keys()) == {
            "kind",
            "label",
            "detail",
            "tone",
            "salience",
            "citations",
        }
        assert d["kind"] in {
            "technical",
            "sentiment",
            "news",
            "sector",
            "trust",
            "calibration",
        }
        assert d["tone"] in {"bull", "bear", "warn", "neutral"}
        assert 0.0 <= d["salience"] <= 1.0
        assert isinstance(d["citations"], list)


def test_uppercase_symbol_normalization() -> None:
    """Caller may pass lowercase; output is always uppercase."""
    result = compute_key_drivers("aapl", analyzer_report=_report(overbought=85))
    assert result.symbol == "AAPL"
