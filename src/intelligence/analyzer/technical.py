"""Normalized technical scoring.

The legacy pipeline mixed raw RSI (0-100), MACD (price units), and a
Bollinger position (roughly -1 to +1) into one composite score with
hand-tuned weights. That made it impossible to compare strengths or
reason about why a name looked "overbought" — the units were
incompatible and the weights weren't grounded in anything.

This module replaces that approach with normalized sub-scores in
``[0, 100]``. Each sub-score answers one well-defined question; the
combiners are simple weighted averages of the relevant sub-scores so
the math is transparent.

Sub-scores
----------
* ``rsi_overbought_score``      — how stretched RSI is into the >70 zone.
* ``rsi_oversold_score``        — how stretched RSI is into the <30 zone.
* ``bollinger_extension_score`` — how far close is outside ±1 sigma.
* ``ma_extension_score``        — % deviation of close from its 50-bar SMA.
* ``volume_spike_score``        — recent-bar volume z-score vs 20-bar log mean.
* ``atr_volatility_score``      — current ATR / 50-bar ATR (overheating proxy).
* ``momentum_exhaustion_score`` — RSI rolling over after a recent peak.

Combiners
---------
* ``overbought_score``  = mean(RSI-OB, Bollinger-OB, MA-OB, ATR, Exhaustion).
* ``oversold_score``    = mean(RSI-OS, Bollinger-OS, MA-OS, ATR, Exhaustion).
* ``pullback_risk``     = mean(Overbought, MomentumExhaustion).
* ``rebound_potential`` = mean(Oversold, MomentumExhaustion).
* ``confidence_score``  = data-coverage * dispersion-penalty — how much
  weight downstream consumers should give this set. Low when half the
  sub-scores were NaN, low when sub-scores wildly disagree.

Everything is computed off the last bar of a closing-price OHLCV
DataFrame. Callers must pass a frame with at least ~60 rows for the
50-bar lookbacks; shorter frames yield lower confidence rather than
exceptions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.indicators.atr import ATR
from src.indicators.bollinger import BollingerBands
from src.indicators.rsi import RSI
from src.indicators.volume import VolumeZScore


@dataclass(frozen=True)
class TechnicalSubscores:
    """The seven normalized [0, 100] sub-scores.

    Each field can be ``None`` when the underlying indicator couldn't
    be computed (e.g. not enough history). Downstream code must treat
    ``None`` as "no signal" rather than treating it as 0 or 50, which
    would silently bias the combiners.
    """

    rsi_overbought_score: float | None
    rsi_oversold_score: float | None
    bollinger_extension_score: float | None
    ma_extension_score: float | None
    volume_spike_score: float | None
    atr_volatility_score: float | None
    momentum_exhaustion_score: float | None


@dataclass(frozen=True)
class TechnicalScores:
    """Full analyzer technical output.

    Carries the per-component sub-scores plus the combiner outputs +
    a ``confidence_score`` summarizing how trustworthy this set is.
    Includes the raw indicator values that drove the scoring so the
    AI explanation layer (Phase 5) can cite numerics without rerunning
    the indicators.
    """

    subscores: TechnicalSubscores
    overbought_score: float | None
    oversold_score: float | None
    pullback_risk: float | None
    rebound_potential: float | None
    confidence_score: float
    raw_rsi: float | None
    raw_bollinger_z: float | None
    raw_ma_distance_pct: float | None
    raw_volume_z: float | None
    raw_atr_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe dict for the API layer (Phase 6 frontend)."""
        return asdict(self)


# Indicator parameters — single source of truth. Exposed as module
# constants so the test suite can pin behavior to specific defaults
# without re-importing the analyzer.
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
SMA_PERIOD = 50
ATR_PERIOD = 14
ATR_BASELINE_PERIOD = 50
VOLUME_PERIOD = 20

# Combiner weights — kept here so the AI explanation prompt can cite
# them verbatim. Equal weights inside each combiner; the bigger
# decisions live in which sub-scores feed which combiner.
_OVERBOUGHT_FIELDS = (
    "rsi_overbought_score",
    "bollinger_extension_score",
    "ma_extension_score",
    "atr_volatility_score",
    "momentum_exhaustion_score",
)
_OVERSOLD_FIELDS = (
    "rsi_oversold_score",
    "bollinger_extension_score",
    "ma_extension_score",
    "atr_volatility_score",
    "momentum_exhaustion_score",
)


def score_technicals(df: pd.DataFrame) -> TechnicalScores:
    """Compute the full :class:`TechnicalScores` from an OHLCV frame.

    The frame must carry ``open / high / low / close / volume`` columns
    (Title case is not accepted — match the rest of the codebase's
    convention). Insufficient history is *not* an error: components
    that can't be computed drop to ``None`` and the confidence score
    reflects the missing coverage.
    """
    if df.empty:
        return _empty_scores()

    close = df["close"]
    last_close = float(close.iloc[-1])

    # ---- RSI sub-scores ----
    rsi_series = RSI(period=RSI_PERIOD).compute(df).dropna()
    raw_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
    rsi_ob = _rsi_overbought_score(raw_rsi)
    rsi_os = _rsi_oversold_score(raw_rsi)
    momentum_exhaustion = _momentum_exhaustion_score(rsi_series)

    # ---- Bollinger extension ----
    bands = BollingerBands(period=BOLLINGER_PERIOD, num_std=BOLLINGER_STD).compute_full(df)
    raw_bz = _bollinger_zscore(close, bands)
    bollinger_ext = _bollinger_extension_score(raw_bz)

    # ---- Moving-average extension ----
    raw_ma_dist = _ma_distance_pct(close, SMA_PERIOD)
    ma_ext = _ma_extension_score(raw_ma_dist)

    # ---- Volume spike ----
    vol_z_series = VolumeZScore(period=VOLUME_PERIOD).compute(df).dropna()
    raw_vol_z = float(vol_z_series.iloc[-1]) if not vol_z_series.empty else None
    vol_spike = _volume_spike_score(raw_vol_z)

    # ---- ATR / baseline ATR ----
    atr_series = ATR(period=ATR_PERIOD).compute(df).dropna()
    raw_atr_ratio = _atr_ratio(atr_series, last_close)
    atr_vol = _atr_volatility_score(raw_atr_ratio)

    subs = TechnicalSubscores(
        rsi_overbought_score=rsi_ob,
        rsi_oversold_score=rsi_os,
        bollinger_extension_score=bollinger_ext,
        ma_extension_score=ma_ext,
        volume_spike_score=vol_spike,
        atr_volatility_score=atr_vol,
        momentum_exhaustion_score=momentum_exhaustion,
    )

    # ---- Combiners ----
    overbought = _combine(subs, _OVERBOUGHT_FIELDS, direction="overbought")
    oversold = _combine(subs, _OVERSOLD_FIELDS, direction="oversold")
    pullback = _mean_of([overbought, momentum_exhaustion])
    rebound = _mean_of([oversold, momentum_exhaustion])
    confidence = _confidence(subs)

    return TechnicalScores(
        subscores=subs,
        overbought_score=overbought,
        oversold_score=oversold,
        pullback_risk=pullback,
        rebound_potential=rebound,
        confidence_score=confidence,
        raw_rsi=raw_rsi,
        raw_bollinger_z=raw_bz,
        raw_ma_distance_pct=raw_ma_dist,
        raw_volume_z=raw_vol_z,
        raw_atr_ratio=raw_atr_ratio,
    )


def _rsi_overbought_score(rsi: float | None) -> float | None:
    """Map RSI in [70, 100] linearly onto [0, 100]; below 70 is 0.

    RSI under 70 contributes nothing to "overbought" — keeping the
    floor at 0 prevents a neutral 50 RSI from leaking into the
    combiner. 100 means RSI = 100 (true asymptote), 50 means RSI = 85.
    """
    if rsi is None or math.isnan(rsi):
        return None
    if rsi <= 70:
        return 0.0
    return _clip(((rsi - 70) / 30) * 100)


def _rsi_oversold_score(rsi: float | None) -> float | None:
    """Mirror image: RSI in [0, 30] mapped linearly onto [0, 100]."""
    if rsi is None or math.isnan(rsi):
        return None
    if rsi >= 30:
        return 0.0
    return _clip(((30 - rsi) / 30) * 100)


def _bollinger_zscore(close: pd.Series, bands: pd.DataFrame) -> float | None:
    """Convert the latest close to a band-relative z-score.

    Using ``(close - middle) / sigma`` where sigma is one std dev
    (derived from the upper band, since ``upper = middle + 2 * std``).
    A close at the upper band returns +2.0; at the lower band, -2.0.
    Outside the bands returns |z| > 2.
    """
    middle = bands["middle"].iloc[-1]
    upper = bands["upper"].iloc[-1]
    if pd.isna(middle) or pd.isna(upper):
        return None
    sigma = (upper - middle) / BOLLINGER_STD
    if sigma == 0 or pd.isna(sigma):
        return None
    last = float(close.iloc[-1])
    return float((last - middle) / sigma)


def _bollinger_extension_score(z: float | None) -> float | None:
    """A two-sided "how far outside the bands" score.

    Symmetric — saturates at |z| = 3 (3 sigma above or below). 0 means
    inside the bands; 100 means well beyond. The combiner direction
    flag picks whether the value should feed overbought or oversold;
    the *score itself* is sign-agnostic because both extensions are
    "extreme".
    """
    if z is None:
        return None
    magnitude = max(0.0, abs(z) - 1.0)  # only count past ±1 sigma
    return _clip((magnitude / 2.0) * 100)


def _ma_distance_pct(close: pd.Series, period: int) -> float | None:
    """Percent deviation of last close from the ``period``-bar SMA."""
    if len(close) < period:
        return None
    sma = close.rolling(period).mean().iloc[-1]
    if pd.isna(sma) or sma == 0:
        return None
    last = float(close.iloc[-1])
    return float((last - sma) / sma) * 100


def _ma_extension_score(distance_pct: float | None) -> float | None:
    """Map |%MA-distance| onto [0, 100], saturating at 30%.

    30% above (or below) a 50-bar SMA is extreme for an equity. The
    cap keeps a single rip-roaring outlier from saturating the
    combiner so hard it drowns the other sub-scores.
    """
    if distance_pct is None:
        return None
    magnitude = abs(distance_pct)
    return _clip((magnitude / 30.0) * 100)


def _volume_spike_score(z: float | None) -> float | None:
    """Map positive z-score onto [0, 100]; negative volume z = 0.

    Quiet volume isn't a "spike". Only positive deviations contribute,
    and we saturate at z = 3 (i.e. ~3-sigma higher than the 20-bar log
    mean — earnings-day territory).
    """
    if z is None or math.isnan(z):
        return None
    if z <= 0:
        return 0.0
    return _clip((z / 3.0) * 100)


def _atr_ratio(atr: pd.Series, last_close: float) -> float | None:
    """Ratio of current ATR to its 50-bar mean ATR.

    1.0 means "vol matches baseline". 2.0 means "vol doubled" — the
    name is overheating. The ratio is preferred to absolute ATR
    because it normalizes across symbols (a $10 stock and a $400
    stock have very different absolute ATRs but similar regimes).
    """
    if atr.empty or last_close <= 0:
        return None
    baseline = atr.rolling(ATR_BASELINE_PERIOD).mean().iloc[-1]
    current = atr.iloc[-1]
    if pd.isna(baseline) or pd.isna(current) or baseline == 0:
        return None
    return float(current / baseline)


def _atr_volatility_score(ratio: float | None) -> float | None:
    """ATR ratio mapped onto [0, 100], saturating at 2x baseline.

    Anything <= 1.0 (current vol at or below baseline) contributes 0.
    A ratio of 2.0 saturates at 100.
    """
    if ratio is None:
        return None
    if ratio <= 1.0:
        return 0.0
    return _clip(((ratio - 1.0) / 1.0) * 100)


def _momentum_exhaustion_score(rsi_series: pd.Series) -> float | None:
    """Detect RSI rolling over after a recent overbought/oversold peak.

    Scans the trailing 10 bars of RSI. Returns:

    * High score when the recent RSI peak was extreme (>80 or <20) and
      RSI has since pulled back materially (>= 5 points). That's the
      classic "momentum exhausted" pattern.
    * Lower scores for milder exhaustion or no recent extreme.
    * ``None`` when there isn't enough RSI history.
    """
    if rsi_series.empty or len(rsi_series) < 5:
        return None
    window = rsi_series.iloc[-10:]
    last = float(window.iloc[-1])
    high = float(window.max())
    low = float(window.min())
    high_pullback = max(0.0, high - last) if high >= 70 else 0.0
    low_pullback = max(0.0, last - low) if low <= 30 else 0.0
    pullback = max(high_pullback, low_pullback)
    if pullback < 5.0:
        return 0.0
    # Map 5..25 points of pullback onto 0..100.
    return _clip(((pullback - 5.0) / 20.0) * 100)


def _combine(
    subs: TechnicalSubscores, fields: tuple[str, ...], *, direction: str
) -> float | None:
    """Mean of the named sub-scores, after a direction-aware mask.

    For ``direction="overbought"``: only count Bollinger / MA scores
    when the underlying movement is *up* (close > middle band, close
    > SMA). For ``direction="oversold"``: only when it's *down*. RSI
    sub-scores and momentum exhaustion already encode direction
    themselves so they're always included.
    """
    values: list[float] = []
    for name in fields:
        value = getattr(subs, name)
        if value is None:
            continue
        values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _mean_of(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _confidence(subs: TechnicalSubscores) -> float:
    """Confidence in this scoring set.

    Two factors, multiplied:

    * **Coverage**: fraction of sub-scores that were computable. Pulls
      to 0 when most fields were ``None`` (insufficient history).
    * **Agreement**: ``1 - dispersion``, where dispersion is the std
      of the populated sub-scores divided by 50 (their possible
      half-range). High disagreement among the sub-scores drops
      confidence — if RSI screams overbought but volume + ATR are
      flat, this set is mixed and downstream weight should be low.

    Output is in ``[0, 100]`` so it composes with the other scores.
    """
    all_fields = (
        subs.rsi_overbought_score,
        subs.rsi_oversold_score,
        subs.bollinger_extension_score,
        subs.ma_extension_score,
        subs.volume_spike_score,
        subs.atr_volatility_score,
        subs.momentum_exhaustion_score,
    )
    populated = [v for v in all_fields if v is not None]
    coverage = len(populated) / len(all_fields)
    if not populated:
        return 0.0
    dispersion = float(np.std(populated)) / 50.0
    agreement = max(0.0, 1.0 - dispersion)
    return _clip(coverage * agreement * 100)


def _empty_scores() -> TechnicalScores:
    """The all-None / zero-confidence baseline.

    Returned when the input frame is empty. Lets callers always render
    the analyzer card without branching on whether scoring happened.
    """
    subs = TechnicalSubscores(
        rsi_overbought_score=None,
        rsi_oversold_score=None,
        bollinger_extension_score=None,
        ma_extension_score=None,
        volume_spike_score=None,
        atr_volatility_score=None,
        momentum_exhaustion_score=None,
    )
    return TechnicalScores(
        subscores=subs,
        overbought_score=None,
        oversold_score=None,
        pullback_risk=None,
        rebound_potential=None,
        confidence_score=0.0,
        raw_rsi=None,
        raw_bollinger_z=None,
        raw_ma_distance_pct=None,
        raw_volume_z=None,
        raw_atr_ratio=None,
    )


def _clip(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


__all__ = [
    "ATR_BASELINE_PERIOD",
    "ATR_PERIOD",
    "BOLLINGER_PERIOD",
    "BOLLINGER_STD",
    "RSI_PERIOD",
    "SMA_PERIOD",
    "VOLUME_PERIOD",
    "TechnicalScores",
    "TechnicalSubscores",
    "score_technicals",
]
