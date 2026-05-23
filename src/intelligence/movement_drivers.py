"""Key Drivers — ranked signals that explain a symbol's current read.

The platform produces a lot of signals (trust grade, sector rank,
technicals, sentiment, calibration, valuation) and asking a trader
to assemble those into a single story every time is wasteful. This
module composes the available signals into a ranked tuple of "what's
most notable about this symbol right now."

What this is, what it isn't
---------------------------
This is **not** "why is the price moving" — without intraday signal
correlation we can't honestly claim causation. It's "what's notable
about this symbol's current posture." Every driver is grounded in a
specific data point we already have; the UI labels them ``Key
Drivers`` (not ``Why Moving``) to match the contract.

Driver kinds
------------
Six kinds, each computed from a different signal stream:

* ``technical``    — extreme sub-scores (overbought/oversold/pullback/rebound)
* ``sentiment``    — news-weighted sentiment polarity
* ``news``         — recent headline volume + first headline verbatim
* ``sector``       — sector-cohort percentile outlier
* ``trust``        — low trust composite (warning)
* ``calibration``  — strong historical hit-rate

Each driver carries a ``salience`` in ``[0, 1]`` describing how far
from the unremarkable baseline this signal is. Drivers below
``min_salience`` (default 0.3) are suppressed; the panel shouldn't
crowd the UI with low-signal items. The top ``max_drivers`` (default
5) by salience are returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

DriverKind = Literal[
    "technical",
    "sentiment",
    "news",
    "sector",
    "trust",
    "calibration",
]

DriverTone = Literal["bull", "bear", "warn", "neutral"]


_DEFAULT_MIN_SALIENCE = 0.3
_DEFAULT_MAX_DRIVERS = 5

# Threshold below which a technical sub-score is considered baseline.
# Sub-scores in [0, 100]; we measure salience as distance from 50 → if
# everything sits in [30, 70] the technicals are too quiet to surface.
_TECHNICAL_BASELINE = 50.0

# Trust score above this is considered "fine" → no trust driver fires.
_TRUST_OK_FLOOR = 80.0


@dataclass(frozen=True)
class Driver:
    """One ranked driver.

    Attributes:
        kind: Source bucket (``technical``/``sentiment``/...). Drives
            icon + accent color in the UI.
        label: Short noun phrase, ≤ 5 words. Renders as the card
            heading.
        detail: One sentence grounded in the source data — must cite
            the actual value that earned the salience score.
        tone: Visual tone (``bull``/``bear``/``warn``/``neutral``).
            Independent of salience; e.g. a high sentiment score is
            high salience AND bullish tone, but a low trust score is
            high salience AND warning tone.
        salience: ``[0, 1]`` — distance from baseline. The ranker
            sorts on this descending. Suppressed below ``min_salience``.
        citations: List of source descriptor strings ("technicals",
            "sector", etc.). Display hint only; not the strict
            citation contract the LLM validator enforces.
    """

    kind: DriverKind
    label: str
    detail: str
    tone: DriverTone
    salience: float
    citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "tone": self.tone,
            "salience": round(self.salience, 3),
            "citations": list(self.citations),
        }


@dataclass(frozen=True)
class KeyDrivers:
    """The ranked driver bundle for one symbol."""

    symbol: str
    drivers: tuple[Driver, ...]
    # Number of drivers computed before truncation. ``len(drivers)``
    # is the rendered count; ``considered`` is total above threshold.
    considered: int
    # Number of computed drivers below the salience threshold
    # (suppressed) — useful so the UI can say "3 low-signal drivers
    # suppressed" if a deeper inspector is ever added.
    suppressed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "drivers": [d.to_dict() for d in self.drivers],
            "considered": self.considered,
            "suppressed": self.suppressed,
        }


# ---------------------------------------------------------------------------
# Public composer
# ---------------------------------------------------------------------------


def compute_key_drivers(
    symbol: str,
    *,
    analyzer_report: dict[str, Any] | None,
    sector_rank: dict[str, Any] | None = None,
    snapshot_row: dict[str, Any] | None = None,
    min_salience: float = _DEFAULT_MIN_SALIENCE,
    max_drivers: int = _DEFAULT_MAX_DRIVERS,
) -> KeyDrivers:
    """Compose the ranked drivers for ``symbol``.

    Each input source may be ``None`` — the driver for that source
    simply doesn't fire. Returns at least an empty :class:`KeyDrivers`
    so the caller can always render.

    ``min_salience`` is the floor below which a driver is suppressed.
    ``max_drivers`` caps the rendered count; the rest are tallied in
    ``suppressed`` (along with sub-threshold ones).
    """
    candidates: list[Driver] = []

    for builder in (
        _technical_driver,
        _sentiment_driver,
        _news_driver,
        _sector_driver,
        _trust_driver,
        _calibration_driver,
    ):
        driver = builder(
            analyzer_report=analyzer_report,
            sector_rank=sector_rank,
            snapshot_row=snapshot_row,
        )
        if driver is not None:
            candidates.append(driver)

    above = [d for d in candidates if d.salience >= min_salience]
    suppressed = len(candidates) - len(above)
    # Sort by salience desc, alphabetic by label for stable ties.
    above.sort(key=lambda d: (-d.salience, d.label))
    truncated = tuple(above[:max_drivers])
    extra_suppressed = max(0, len(above) - max_drivers)
    return KeyDrivers(
        symbol=symbol.upper(),
        drivers=truncated,
        considered=len(above),
        suppressed=suppressed + extra_suppressed,
    )


# ---------------------------------------------------------------------------
# Per-kind builders
# ---------------------------------------------------------------------------


def _technical_driver(
    *,
    analyzer_report: dict[str, Any] | None,
    sector_rank: dict[str, Any] | None,
    snapshot_row: dict[str, Any] | None,
) -> Driver | None:
    """Pick the most extreme of the four technical sub-scores."""
    if analyzer_report is None:
        return None
    technicals = analyzer_report.get("technicals") or {}
    # Declarative spec: per-field label + tone. Tones are typed at the
    # spec level so the literal narrowing reaches the Driver constructor.
    specs: tuple[tuple[str, str, DriverTone], ...] = (
        ("overbought_score", "Overbought technicals", "warn"),
        ("oversold_score", "Oversold setup", "bull"),
        ("pullback_risk", "Elevated pullback risk", "warn"),
        ("rebound_potential", "Rebound potential", "bull"),
    )
    best: tuple[str, float, DriverTone, float] | None = None
    for field, label, tone in specs:
        value = technicals.get(field)
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        salience = max(0.0, (v - _TECHNICAL_BASELINE) / _TECHNICAL_BASELINE)
        if best is None or salience > best[3]:
            best = (label, v, tone, salience)
    if best is None or best[3] <= 0:
        return None
    label, value, tone, salience = best
    return Driver(
        kind="technical",
        label=label,
        detail=(
            f"{label.lower()} composite reads {value:.0f}/100 — distance from "
            "neutral makes this the loudest technical signal."
        ),
        tone=tone,
        salience=min(1.0, salience),
        citations=("technicals",),
    )


def _sentiment_driver(
    *,
    analyzer_report: dict[str, Any] | None,
    sector_rank: dict[str, Any] | None,
    snapshot_row: dict[str, Any] | None,
) -> Driver | None:
    """News-weighted sentiment polarity. Needs row data (sentiment + count)."""
    if snapshot_row is None:
        return None
    sentiment = snapshot_row.get("sentiment_score")
    headlines = snapshot_row.get("num_news_articles")
    if sentiment is None or headlines is None:
        return None
    try:
        s = float(sentiment)
        n = int(headlines)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    # Volume-weighted polarity: |s| * log(1+n) normalized to [0, 1]
    # at roughly s=0.5 across 8+ articles.
    salience = min(1.0, abs(s) * math.log(1 + n) / 2.0)
    if salience < 0.05:
        return None
    tone: DriverTone = "bull" if s > 0 else "bear" if s < 0 else "neutral"
    direction = "bullish" if s > 0 else "bearish" if s < 0 else "mixed"
    return Driver(
        kind="sentiment",
        label=f"{direction.title()} news sentiment",
        detail=(
            f"FinBERT polarity {s:+.2f} across {n} article"
            f"{'s' if n != 1 else ''} — volume-weighted polarity stands out."
        ),
        tone=tone,
        salience=salience,
        citations=("sentiment", "news"),
    )


def _news_driver(
    *,
    analyzer_report: dict[str, Any] | None,
    sector_rank: dict[str, Any] | None,
    snapshot_row: dict[str, Any] | None,
) -> Driver | None:
    """High headline volume → news-coverage driver. Quote the top headline."""
    if snapshot_row is None:
        return None
    headlines_raw = snapshot_row.get("headlines") or []
    if not isinstance(headlines_raw, list):
        return None
    headlines = [h for h in headlines_raw if isinstance(h, str) and h.strip()]
    if len(headlines) < 3:
        return None
    # Salience scales with coverage volume; 10+ headlines = 1.0.
    salience = min(1.0, len(headlines) / 10.0)
    top = headlines[0].strip()
    return Driver(
        kind="news",
        label="Active news cycle",
        detail=(
            f"{len(headlines)} headlines in the current window. Latest: "
            f"“{top}”"
        ),
        tone="neutral",
        salience=salience,
        citations=("news",),
    )


def _sector_driver(
    *,
    analyzer_report: dict[str, Any] | None,
    sector_rank: dict[str, Any] | None,
    snapshot_row: dict[str, Any] | None,
) -> Driver | None:
    """Most extreme sector-cohort percentile across the ranked metrics."""
    if sector_rank is None or not sector_rank.get("available"):
        return None
    metrics = sector_rank.get("metrics") or []
    best_metric: dict[str, Any] | None = None
    best_salience = 0.0
    for m in metrics:
        pct = m.get("percentile")
        if pct is None:
            continue
        try:
            p = float(pct)
        except (TypeError, ValueError):
            continue
        # Outlier salience: high at top (p→100) or bottom (p→0); zero
        # at median. Both ends of the cohort are "notable."
        salience = abs(p - 50.0) / 50.0
        if salience > best_salience:
            best_salience = salience
            best_metric = m
    if best_metric is None or best_salience < 0.05:
        return None
    label = str(best_metric.get("label") or "Sector position")
    rank = best_metric.get("rank")
    cohort = best_metric.get("cohort_size") or 0
    pct = best_metric.get("percentile")
    sector_name = sector_rank.get("sector") or "sector"
    direction = best_metric.get("direction") or "higher_better"
    # Tone follows: top decile = bull, bottom decile = bear; in between = neutral.
    tone: DriverTone
    if pct is not None and pct >= 80.0:
        tone = "bull"
    elif pct is not None and pct <= 20.0:
        tone = "bear"
    else:
        tone = "neutral"
    direction_phrase = (
        "higher is better" if direction == "higher_better" else "lower is better"
    )
    pct_str = f"{pct:.0f}th percentile" if pct is not None else "outlier"
    rank_str = f"#{rank}/{cohort}" if rank else "ranked"
    return Driver(
        kind="sector",
        label=f"Sector outlier · {label}",
        detail=(
            f"{rank_str} in {sector_name} on {label.lower()} "
            f"({pct_str}; {direction_phrase})."
        ),
        tone=tone,
        salience=best_salience,
        citations=("sector",),
    )


def _trust_driver(
    *,
    analyzer_report: dict[str, Any] | None,
    sector_rank: dict[str, Any] | None,
    snapshot_row: dict[str, Any] | None,
) -> Driver | None:
    """Low Trust Score → warning driver. High trust is not a driver
    (it's the baseline; the panel highlights what stands out)."""
    if analyzer_report is None:
        return None
    trust = analyzer_report.get("trust_score") or {}
    score = trust.get("score")
    grade = trust.get("grade")
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s >= _TRUST_OK_FLOOR:
        return None
    # Salience grows as trust drops. Scaled so grade D (~65) clears
    # the default 0.3 threshold; grade F (≤60) crosses 0.5; collapse
    # to 1.0 around score=30. Steeper than (floor-s)/floor would be.
    salience = max(0.0, (_TRUST_OK_FLOOR - s) / 50.0)
    return Driver(
        kind="trust",
        label="Degraded report trust",
        detail=(
            f"Trust composite at {s:.0f}/100 (grade {grade or '—'}) — "
            "lean on the per-component breakdown before acting."
        ),
        tone="warn",
        salience=min(1.0, salience),
        citations=("trust_score",),
    )


def _calibration_driver(
    *,
    analyzer_report: dict[str, Any] | None,
    sector_rank: dict[str, Any] | None,
    snapshot_row: dict[str, Any] | None,
) -> Driver | None:
    """Strongest published calibration reading (distance from 50% hit rate)."""
    if analyzer_report is None:
        return None
    readings = analyzer_report.get("calibrations") or []
    best: dict[str, Any] | None = None
    best_salience = 0.0
    for r in readings:
        if not r.get("bucket_published"):
            continue
        rate = r.get("hit_rate")
        n = r.get("n_observations")
        if rate is None or n is None:
            continue
        try:
            rate_f = float(rate)
            n_int = int(n)
        except (TypeError, ValueError):
            continue
        if n_int <= 0:
            continue
        # Distance from 50% weighted by observation count.
        distance = abs(rate_f - 0.5) * 2.0  # [0, 1]
        # log-scale dampen for low-obs buckets so 5 observations
        # don't dominate; 100+ obs is essentially full weight.
        weight = math.log(1 + n_int) / math.log(101)
        salience = min(1.0, distance * weight)
        if salience > best_salience:
            best_salience = salience
            best = r
    if best is None or best_salience < 0.05:
        return None
    rate_pct = float(best["hit_rate"]) * 100.0
    n = int(best["n_observations"])
    score_name = str(best.get("score_name") or "signal")
    outcome = str(best.get("outcome_name") or "outcome")
    horizon = best.get("horizon_days")
    direction = "positive" if rate_pct >= 50.0 else "downward"
    tone: DriverTone = "bull" if rate_pct >= 50.0 else "bear"
    horizon_str = f" over {horizon}d" if horizon else ""
    return Driver(
        kind="calibration",
        label="Historical signal calibration",
        detail=(
            f"Similar {score_name} setups resolved {direction} "
            f"{rate_pct:.0f}% of the time{horizon_str} "
            f"(n={n}, outcome: {outcome})."
        ),
        tone=tone,
        salience=best_salience,
        citations=("calibration",),
    )


__all__ = [
    "Driver",
    "DriverKind",
    "DriverTone",
    "KeyDrivers",
    "compute_key_drivers",
]
