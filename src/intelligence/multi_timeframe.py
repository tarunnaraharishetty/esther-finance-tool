"""Multi-timeframe trend reading derived from a single OHLCV window.

Answers the discretionary trader's "is this short-window move consistent
with the longer trend?" question. Three horizons (short / medium / long)
classified by sign and magnitude of pct change, plus a watchlist-stable
``alignment`` label so the dashboard can color the table cell at a glance.

Why derive from one window instead of fetching three Alpaca resolutions:
the controller already builds a daily-bar dataframe per symbol per tick.
Adding 5m + 1h fetches would triple API calls + rate-limit pressure for
no gain to a daily-cadence decision-support tool. The three horizons are
slices of the same df: short = last 5 bars, medium = 20, long = 60.

Pure data. No state, no LLM. ``compute_mtf_view`` returns ``None`` when
the df is shorter than the short window (5 bars); otherwise computes
what it can per horizon. Render code branches on ``None`` to omit the
column / section entirely.

No prediction language — every value is a signed comparison of close
prices already in the df.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# Window sizes in bars. With daily bars these are ~1 week, ~1 month,
# ~3 months. Anchored to the long window: a df with fewer than `_LONG`
# bars still produces a view — the long horizon just uses what's available.
_SHORT_BARS = 5
_MEDIUM_BARS = 20
_LONG_BARS = 60

# Neutral thresholds per horizon (absolute pct). Below this the trend
# is "neutral" — too small a move on that horizon to call directional.
# Larger windows need larger moves to count as directional (a 1% drift
# over 60 bars is not a trend; over 5 bars it is).
_NEUTRAL_PCT_SHORT = 1.5
_NEUTRAL_PCT_MEDIUM = 3.0
_NEUTRAL_PCT_LONG = 5.0

# Saturation thresholds — the abs pct at which strength reaches 1.0.
# Anything beyond saturates. Tuned so a "clear" multi-week trend on
# liquid equities reads as strong rather than maxed out by every spike.
_STRENGTH_SATURATE_SHORT = 5.0
_STRENGTH_SATURATE_MEDIUM = 10.0
_STRENGTH_SATURATE_LONG = 20.0


@dataclass(frozen=True)
class TimeframeTrend:
    """Trend reading for one horizon.

    Stable schema so the render code never branches on values it doesn't
    expect. ``direction`` ∈ {"bullish", "bearish", "neutral"}, ``strength``
    ∈ [0, 1].
    """

    horizon: str  # "short" | "medium" | "long"
    direction: str  # "bullish" | "bearish" | "neutral"
    strength: float  # [0, 1]
    delta_pct: float  # signed pct change over the window
    bars_used: int  # actual bars in the slice; may be < window if df is short


@dataclass(frozen=True)
class MultiTimeframeView:
    """Three-horizon trend summary for one symbol.

    ``alignment`` is the trader's headline read:
      * ``aligned_bullish`` — all three horizons bullish
      * ``aligned_bearish`` — all three horizons bearish
      * ``short_reversal`` — short opposes long (regardless of medium)
      * ``mixed`` — partial alignment, neutrals, anything else

    Includes ``alignment_strength`` ∈ [0, 1] so the dashboard can dim
    weak readings even when the alignment label is directional.
    """

    symbol: str
    short: TimeframeTrend
    medium: TimeframeTrend
    long: TimeframeTrend
    alignment: str
    alignment_strength: float

    @property
    def is_aligned(self) -> bool:
        return self.alignment in ("aligned_bullish", "aligned_bearish")

    @property
    def is_conflicting(self) -> bool:
        return self.alignment == "short_reversal"


def compute_mtf_view(df: pd.DataFrame, symbol: str) -> MultiTimeframeView | None:
    """Derive a :class:`MultiTimeframeView` from an OHLCV dataframe.

    Returns ``None`` when the df has fewer than the short window (5 bars)
    or no ``close`` column — the row is unrenderable. Otherwise computes
    each horizon, using ``min(window, len(df))`` for short dfs.
    """
    if df is None or len(df) < _SHORT_BARS or "close" not in df.columns:
        return None

    short = _trend_for(df, "short", _SHORT_BARS, _NEUTRAL_PCT_SHORT, _STRENGTH_SATURATE_SHORT)
    medium = _trend_for(df, "medium", _MEDIUM_BARS, _NEUTRAL_PCT_MEDIUM, _STRENGTH_SATURATE_MEDIUM)
    long = _trend_for(df, "long", _LONG_BARS, _NEUTRAL_PCT_LONG, _STRENGTH_SATURATE_LONG)

    alignment = _classify_alignment(short, medium, long)
    alignment_strength = _classify_alignment_strength(short, medium, long, alignment)

    return MultiTimeframeView(
        symbol=symbol,
        short=short,
        medium=medium,
        long=long,
        alignment=alignment,
        alignment_strength=alignment_strength,
    )


def _trend_for(
    df: pd.DataFrame,
    horizon: str,
    window: int,
    neutral_pct: float,
    saturate_pct: float,
) -> TimeframeTrend:
    """Compute the trend reading for one horizon.

    Pct change is measured between the close at the start of the window
    and the latest close. Window is clipped to available bars — short
    dfs still produce a (lower-strength) reading rather than None.
    """
    bars_used = min(window, len(df))
    closes = df["close"].iloc[-bars_used:]
    start = float(closes.iloc[0])
    end = float(closes.iloc[-1])
    delta_pct = ((end / start) - 1.0) * 100.0 if start != 0.0 else 0.0

    abs_delta = abs(delta_pct)
    if abs_delta < neutral_pct:
        direction = "neutral"
    elif delta_pct > 0:
        direction = "bullish"
    else:
        direction = "bearish"

    strength = min(1.0, abs_delta / saturate_pct) if saturate_pct > 0 else 0.0

    return TimeframeTrend(
        horizon=horizon,
        direction=direction,
        strength=strength,
        delta_pct=delta_pct,
        bars_used=bars_used,
    )


def _classify_alignment(
    short: TimeframeTrend, medium: TimeframeTrend, long: TimeframeTrend
) -> str:
    """Pattern-match the three direction labels.

    Aligned variants require ALL three horizons in the same non-neutral
    direction — a single neutral demotes to mixed. Short-reversal fires
    whenever short and long have opposite non-neutral directions, even
    if medium is neutral or aligned with one side (it's still the
    headline conflict for a trader).
    """
    s, m, lo = short.direction, medium.direction, long.direction
    if s == "bullish" and m == "bullish" and lo == "bullish":
        return "aligned_bullish"
    if s == "bearish" and m == "bearish" and lo == "bearish":
        return "aligned_bearish"
    if (s == "bullish" and lo == "bearish") or (s == "bearish" and lo == "bullish"):
        return "short_reversal"
    return "mixed"


def _classify_alignment_strength(
    short: TimeframeTrend,
    medium: TimeframeTrend,
    long: TimeframeTrend,
    alignment: str,
) -> float:
    """Strength of the alignment label.

    For aligned variants: mean of the three horizon strengths. For
    short_reversal: mean of short + long strengths (medium ignored —
    it can be neutral or either side). For mixed: average of whatever
    horizons are non-neutral, or 0.0 when all three are neutral.
    """
    if alignment in ("aligned_bullish", "aligned_bearish"):
        return (short.strength + medium.strength + long.strength) / 3.0
    if alignment == "short_reversal":
        return (short.strength + long.strength) / 2.0
    directional = [
        t.strength for t in (short, medium, long) if t.direction != "neutral"
    ]
    if not directional:
        return 0.0
    return sum(directional) / len(directional)


__all__ = ["MultiTimeframeView", "TimeframeTrend", "compute_mtf_view"]
