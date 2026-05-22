"""Analyzer-side calibration lookup.

Bridges :class:`~src.intelligence.analyzer.technical.TechnicalScores`
to :class:`~src.intelligence.calibration.CalibrationTable`. The
calibration storage layer is generic — it doesn't know about "pullback
risk" or "rebound potential" — so this module owns the analyzer-
specific score↔outcome pairings.

Why the pairings are hardcoded
------------------------------
The pairings encode a *semantic claim* about what each score means:
"a high pullback_risk should predict a negative next-N-day return".
Moving them to config would let an operator silently change the
meaning of a published probability, which is exactly the kind of
trust-eroding ambiguity the platform avoids. Adding a new pairing
should be a code change reviewed against the underlying score's
definition.

What we deliberately don't calibrate
------------------------------------
``momentum_exhaustion_score`` is direction-agnostic ("RSI rolled
over after a recent extreme") — the binary outcome question
("did the move reverse?") needs a starting-position reference, which
the scoring layer doesn't track. Adding it requires defining that
contract first.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.intelligence.analyzer.technical import TechnicalScores
from src.intelligence.calibration import CalibrationBucket, CalibrationTable

# Each entry pairs a score with the binary outcome it claims to
# predict. The semantic claim:
# * ``overbought_score``  high → next return more likely negative
# * ``oversold_score``    high → next return more likely positive
# * ``pullback_risk``     high → next return more likely negative
# * ``rebound_potential`` high → next return more likely positive
#
# Both ``return_negative`` and ``return_positive`` are evaluated
# against the same observation horizon (the operator's choice of
# ``calibration_horizon_days``, default 5).
PAIRINGS: tuple[tuple[str, str], ...] = (
    ("overbought_score", "return_negative"),
    ("oversold_score", "return_positive"),
    ("pullback_risk", "return_negative"),
    ("rebound_potential", "return_positive"),
)


@dataclass(frozen=True)
class CalibrationReading:
    """One calibrated probability lookup for the analyzer report.

    Even when the bucket is under-sampled (``bucket_published=False``),
    the reading is still emitted so the UI can render "calibration
    pending" with the actual score value. Absent fields would force
    the UI to disambiguate "we didn't compute this" from "we don't
    have enough history yet" — same render, very different meaning.
    """

    score_name: str
    score_value: float
    outcome_name: str
    horizon_days: int
    bucket_published: bool
    bucket: CalibrationBucket | None


def lookup_readings(
    technicals: TechnicalScores | None,
    table: CalibrationTable | None,
    *,
    horizon_days: int,
    min_observations: int,
) -> tuple[CalibrationReading, ...]:
    """Produce one reading per pairing whose score is computable.

    Returns an empty tuple when ``technicals`` or ``table`` is None
    — the caller treats absence as "no calibration available", not
    as a failure. A score field that's ``None`` on ``TechnicalScores``
    (e.g., insufficient bar history) drops out silently.
    """
    if technicals is None or table is None:
        return ()
    readings: list[CalibrationReading] = []
    for score_name, outcome_name in PAIRINGS:
        value = getattr(technicals, score_name, None)
        if value is None:
            continue
        bucket = table.lookup(
            score_name,
            float(value),
            horizon_days=horizon_days,
            outcome_name=outcome_name,
            min_observations=min_observations,
        )
        readings.append(
            CalibrationReading(
                score_name=score_name,
                score_value=float(value),
                outcome_name=outcome_name,
                horizon_days=horizon_days,
                bucket_published=bucket is not None,
                bucket=bucket,
            )
        )
    return tuple(readings)


__all__ = [
    "PAIRINGS",
    "CalibrationReading",
    "lookup_readings",
]
