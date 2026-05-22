"""Report Trust Score — composite quality grade for analyzer + research reports.

The trust score is Esther's north-star quality metric. Every other
subsystem in the platform (freshness envelopes, provider reconciliation,
calibration, scenarios, claim validation) exists to move it. Surfacing
it as a single 0-100 + letter grade gives traders one number to anchor
on and gives the codebase one number to optimize against.

Composition (data-first weighting):

* ``freshness``              — 0.25 — fundamentals freshness tier.
* ``provider_confidence``    — 0.25 — chain confidence after reconciliation.
* ``analyzer_confidence``    — 0.20 — technical coverage × agreement.
* ``calibration_coverage``   — 0.15 — fraction of analyzer scores with a
  published historical-outcome bucket.
* ``scenario_availability``  — 0.10 — probabilistic scenario layer attached.
* ``validation_cleanliness`` — 0.05 — LLM claim validator drop count.

Components that don't apply to a given report type (e.g. validation on
the analyzer endpoint, calibration on the research endpoint) are
marked ``n/a`` and excluded from the weighted sum; the remaining
weights are renormalized to 1.0 so the score remains in [0, 100].
Components that *should* have data but didn't (e.g. fundamentals chain
exhausted) are marked ``missing`` and excluded the same way — but the
component carries a human-readable ``detail`` so the UI can show why.

Grades:

* ``A+`` ≥ 95
* ``A``  ≥ 90
* ``B``  ≥ 80
* ``C``  ≥ 70
* ``D``  ≥ 60
* ``F``  < 60

The grade thresholds are deliberately strict: a "B" report is good
enough to act on; an "A" report is institutional-grade. Anything below
"C" should not be presented without a degraded-mode warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

COMPONENT_FRESHNESS = "freshness"
COMPONENT_PROVIDER = "provider_confidence"
COMPONENT_ANALYZER = "analyzer_confidence"
COMPONENT_CALIBRATION = "calibration_coverage"
COMPONENT_SCENARIO = "scenario_availability"
COMPONENT_VALIDATION = "validation_cleanliness"


WEIGHTS: dict[str, float] = {
    COMPONENT_FRESHNESS: 0.25,
    COMPONENT_PROVIDER: 0.25,
    COMPONENT_ANALYZER: 0.20,
    COMPONENT_CALIBRATION: 0.15,
    COMPONENT_SCENARIO: 0.10,
    COMPONENT_VALIDATION: 0.05,
}

# Freshness tier → component value in [0, 100]. The "fresh" floor is set
# below 100 so a fresh report still has headroom: a perfect score
# requires fresh data AND a fully-attached scoring stack.
_FRESHNESS_SCORE: dict[str, float] = {
    "fresh": 100.0,
    "aging": 75.0,
    "stale": 40.0,
    "expired": 10.0,
}


ComponentStatus = Literal["ok", "warn", "missing", "n/a"]


@dataclass(frozen=True)
class TrustComponent:
    """One weighted ingredient of the trust score.

    ``value`` is the component's own score in [0, 100] when status is
    ``ok``/``warn`` and ``None`` otherwise. ``contribution`` is the
    weighted contribution to the final composite *after* renormalizing
    over present components — the sum of contributions across all
    components with a non-None value equals ``score`` exactly.
    ``detail`` is a one-line human-readable explanation safe to render
    in a tooltip.
    """

    name: str
    weight: float
    value: float | None
    contribution: float | None
    status: ComponentStatus
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weight": round(self.weight, 4),
            "value": None if self.value is None else round(self.value, 1),
            "contribution": (
                None if self.contribution is None else round(self.contribution, 2)
            ),
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TrustScore:
    """Composite trust grade for a single report."""

    score: float
    grade: str
    components: tuple[TrustComponent, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 1),
            "grade": self.grade,
            "components": [c.to_dict() for c in self.components],
        }


def _grade(score: float) -> str:
    if score >= 95.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _na(name: str, detail: str) -> TrustComponent:
    return TrustComponent(
        name=name,
        weight=WEIGHTS[name],
        value=None,
        contribution=None,
        status="n/a",
        detail=detail,
    )


def _missing(name: str, detail: str) -> TrustComponent:
    return TrustComponent(
        name=name,
        weight=WEIGHTS[name],
        value=None,
        contribution=None,
        status="missing",
        detail=detail,
    )


def _ok_or_warn(
    name: str,
    *,
    value: float,
    threshold: float,
    detail: str,
) -> TrustComponent:
    status: ComponentStatus = "ok" if value >= threshold else "warn"
    return TrustComponent(
        name=name,
        weight=WEIGHTS[name],
        value=value,
        contribution=0.0,  # filled in by ``_build`` after renormalization.
        status=status,
        detail=detail,
    )


def _freshness_component(freshness: str | None) -> TrustComponent:
    if freshness is None:
        return _missing(
            COMPONENT_FRESHNESS,
            "No fundamentals freshness envelope available.",
        )
    pct = _FRESHNESS_SCORE.get(freshness)
    if pct is None:
        return _missing(
            COMPONENT_FRESHNESS,
            f"Unknown freshness tier '{freshness}'.",
        )
    return _ok_or_warn(
        COMPONENT_FRESHNESS,
        value=pct,
        threshold=75.0,
        detail=f"Fundamentals data tier: {freshness}.",
    )


def _provider_component(pc: float | None) -> TrustComponent:
    if pc is None:
        return _missing(
            COMPONENT_PROVIDER,
            "Provider confidence unavailable — fundamentals chain did not return.",
        )
    clamped = max(0.0, min(1.0, pc))
    return _ok_or_warn(
        COMPONENT_PROVIDER,
        value=clamped * 100.0,
        threshold=70.0,
        detail=f"Provider chain confidence after reconciliation: {clamped:.2f}.",
    )


def _analyzer_component(c: float | None) -> TrustComponent:
    if c is None:
        return _missing(
            COMPONENT_ANALYZER,
            "Technical scoring unavailable — no bars returned for this symbol.",
        )
    clamped = max(0.0, min(1.0, c))
    return _ok_or_warn(
        COMPONENT_ANALYZER,
        value=clamped * 100.0,
        threshold=60.0,
        detail=(
            f"Technical sub-score coverage × agreement: {clamped:.2f}."
        ),
    )


def _calibration_component(coverage: float | None) -> TrustComponent:
    if coverage is None:
        return _missing(
            COMPONENT_CALIBRATION,
            "Calibration store not configured for this report.",
        )
    clamped = max(0.0, min(1.0, coverage))
    return _ok_or_warn(
        COMPONENT_CALIBRATION,
        value=clamped * 100.0,
        threshold=50.0,
        detail=(
            f"{clamped:.0%} of analyzer scores have a published historical-outcome "
            "bucket."
        ),
    )


def _scenario_component(available: bool | None) -> TrustComponent:
    if available is None:
        return _missing(
            COMPONENT_SCENARIO,
            "Scenario layer unavailable — bars or last price missing.",
        )
    if available:
        return _ok_or_warn(
            COMPONENT_SCENARIO,
            value=100.0,
            threshold=100.0,
            detail="Probabilistic scenario layer attached to this report.",
        )
    return _ok_or_warn(
        COMPONENT_SCENARIO,
        value=0.0,
        threshold=100.0,
        detail="No scenario layer attached.",
    )


def _validation_component(drop_count: int | None) -> TrustComponent:
    """Score the LLM claim validator's hygiene.

    Each dropped claim subtracts 10 points from 100, floored at 0. The
    simple linear formula avoids needing a total-claims denominator
    while still penalizing chronic hallucination: a thesis that loses
    five claims grades as 50, not 100.
    """
    if drop_count is None:
        return _na(
            COMPONENT_VALIDATION,
            "No LLM validation step applies to this report.",
        )
    if drop_count < 0:
        drop_count = 0
    value = max(0.0, 100.0 - 10.0 * float(drop_count))
    threshold = 90.0  # one dropped claim is still "ok"; two or more warns.
    status: ComponentStatus = "ok" if value >= threshold else "warn"
    if drop_count == 0:
        detail = "Research validator dropped zero unsupported claims."
    elif drop_count == 1:
        detail = "Research validator dropped 1 unsupported claim."
    else:
        detail = (
            f"Research validator dropped {drop_count} unsupported claims."
        )
    return TrustComponent(
        name=COMPONENT_VALIDATION,
        weight=WEIGHTS[COMPONENT_VALIDATION],
        value=value,
        contribution=0.0,
        status=status,
        detail=detail,
    )


def _build(components: list[TrustComponent]) -> TrustScore:
    """Renormalize weights over present components and compute final score."""
    present = [c for c in components if c.value is not None]
    if not present:
        return TrustScore(
            score=0.0,
            grade="F",
            components=tuple(components),
        )
    total_weight = sum(c.weight for c in present)
    if total_weight <= 0.0:
        return TrustScore(score=0.0, grade="F", components=tuple(components))

    score = 0.0
    out: list[TrustComponent] = []
    for c in components:
        if c.value is None:
            out.append(c)
            continue
        contribution = (c.weight / total_weight) * c.value
        score += contribution
        out.append(
            TrustComponent(
                name=c.name,
                weight=c.weight,
                value=c.value,
                contribution=contribution,
                status=c.status,
                detail=c.detail,
            )
        )
    score = max(0.0, min(100.0, score))
    return TrustScore(
        score=score,
        grade=_grade(score),
        components=tuple(out),
    )


def compute_analyzer_trust_score(
    *,
    freshness: str | None,
    provider_confidence: float | None,
    analyzer_confidence: float | None,
    calibration_coverage: float | None,
    scenarios_available: bool | None,
) -> TrustScore:
    """Trust score for an Analyzer report (``/api/analyzer/{symbol}``).

    ``validation_cleanliness`` is marked ``n/a`` because the analyzer
    path is deterministic — no LLM claims are generated, so there's
    nothing for the research validator to scrub.
    """
    return _build(
        [
            _freshness_component(freshness),
            _provider_component(provider_confidence),
            _analyzer_component(analyzer_confidence),
            _calibration_component(calibration_coverage),
            _scenario_component(scenarios_available),
            _na(
                COMPONENT_VALIDATION,
                "Analyzer reports are deterministic; no LLM validation step.",
            ),
        ]
    )


def compute_research_trust_score(
    *,
    validation_drop_count: int | None,
) -> TrustScore:
    """Trust score for a Research thesis (``/api/research/{symbol}``).

    Research reports don't fetch their own fundamentals — they operate
    on the live snapshot row — so the freshness/provider/analyzer/
    calibration/scenario components are ``n/a``. The composite is
    dominated by validator hygiene, which is the legitimate signal for
    a generated thesis.
    """
    return _build(
        [
            _na(
                COMPONENT_FRESHNESS,
                "Research operates on the live snapshot, not a freshness envelope.",
            ),
            _na(
                COMPONENT_PROVIDER,
                "Research does not call the fundamentals provider chain directly.",
            ),
            _na(
                COMPONENT_ANALYZER,
                "Research uses prose synthesis rather than technical sub-scores.",
            ),
            _na(
                COMPONENT_CALIBRATION,
                "Research is text; calibration applies to numeric scores only.",
            ),
            _na(
                COMPONENT_SCENARIO,
                "Research does not attach a scenario distribution.",
            ),
            _validation_component(validation_drop_count),
        ]
    )


__all__ = [
    "COMPONENT_ANALYZER",
    "COMPONENT_CALIBRATION",
    "COMPONENT_FRESHNESS",
    "COMPONENT_PROVIDER",
    "COMPONENT_SCENARIO",
    "COMPONENT_VALIDATION",
    "WEIGHTS",
    "ComponentStatus",
    "TrustComponent",
    "TrustScore",
    "compute_analyzer_trust_score",
    "compute_research_trust_score",
]
