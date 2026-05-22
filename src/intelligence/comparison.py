"""Side-by-side comparison of two analyzer reports.

Pure, deterministic composition over the assembled analyzer wire
shape. Takes two report dicts (as emitted by ``src/api/analyzer.py``)
and produces a :class:`ComparisonView` with grouped sections and a
per-metric verdict ("left wins" / "right wins" / "tie" / "n/a").

What this is, what it isn't
---------------------------
This module decides which symbol is the better *long-side candidate*
on each metric. Long is the default framing because that's how
discretionary traders default to reading a watchlist. The directions
are explicit per metric and the UI surface labels them honestly —
"overbought_score · lower better" — so the user can invert the read
in their head when they're sizing a short.

The verdict is *not* an investment recommendation. It says "right now,
on this metric, this side scores better." Stack ten metrics together
and you get a tape-read, not a position.

Why dicts on the wire boundary
------------------------------
The analyzer endpoint already serializes its typed objects to JSON
for the wire. Having :func:`compare_reports` take dicts means we
test against the same shape the frontend renders — there's no second
representation to keep in sync. The price is that we walk untyped
fields and use ``.get`` defensively; the gain is one source of truth
about the wire contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Winner = Literal["left", "right", "tie", "n/a"]
Direction = Literal["higher_better", "lower_better"]


@dataclass(frozen=True)
class MetricComparison:
    """One row in the comparison table.

    ``left_value`` / ``right_value`` are ``None`` when the underlying
    field was missing on that side. ``winner`` is ``"n/a"`` when
    *both* sides were missing; if only one side was missing, that
    side loses (we have nothing to compare, so the side with data
    wins by default).

    ``detail`` is the one-line audit string the UI shows in a tooltip
    — useful for "why did this row score this way?" without having to
    re-derive the verdict.
    """

    metric: str
    label: str
    left_value: float | None
    right_value: float | None
    winner: Winner
    direction: Direction
    detail: str
    # When the metric is naturally percent-shaped (e.g. upside), the
    # UI renders ``42.3%``; otherwise raw number. Set to True per
    # metric in the metric_spec below so the rendering stays declarative.
    percent: bool = False


@dataclass(frozen=True)
class ComparisonSection:
    """One named group of metrics (Trust, Valuation, etc.)."""

    title: str
    metrics: tuple[MetricComparison, ...]


@dataclass(frozen=True)
class HeadlineMetric:
    """Top-of-page summary value rendered on each side of the hero.

    Mirrors :class:`MetricComparison` but carries display-ready labels
    ("Trust grade", "A+", "92 / 100") because the hero panel needs
    formatted strings, not raw floats.
    """

    label: str
    left_display: str
    right_display: str
    winner: Winner


@dataclass(frozen=True)
class ComparisonView:
    """Full structured comparison between two symbols."""

    left_symbol: str
    right_symbol: str
    headline: tuple[HeadlineMetric, ...]
    sections: tuple[ComparisonSection, ...]
    overall_winner: Winner
    # Free-text notes about degraded sides (e.g. missing fundamentals).
    # Empty in the happy path; populated when either report is thin.
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_symbol": self.left_symbol,
            "right_symbol": self.right_symbol,
            "headline": [
                {
                    "label": h.label,
                    "left_display": h.left_display,
                    "right_display": h.right_display,
                    "winner": h.winner,
                }
                for h in self.headline
            ],
            "sections": [
                {
                    "title": s.title,
                    "metrics": [
                        {
                            "metric": m.metric,
                            "label": m.label,
                            "left_value": m.left_value,
                            "right_value": m.right_value,
                            "winner": m.winner,
                            "direction": m.direction,
                            "detail": m.detail,
                            "percent": m.percent,
                        }
                        for m in s.metrics
                    ],
                }
                for s in self.sections
            ],
            "overall_winner": self.overall_winner,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Metric specifications (the comparison contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MetricSpec:
    """Declarative recipe for one metric in the comparison.

    Keeping the spec separate from the compute step means the contract
    is one read away from the test — every new metric is an entry in
    the table at the bottom of this module, not a new code path.
    """

    metric: str
    label: str
    section: str
    direction: Direction
    extract: str  # dotted path into the report dict, see _walk()
    detail: str   # rendered with .format(symbol=..., value=...)
    percent: bool = False
    # Equality tolerance — values within this range round to "tie".
    # Tuned per metric: trust scores tied within 1.0 point; per-cent
    # metrics within 0.5; valuation upsides within 0.5% etc.
    tie_within: float = 0.5


_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    # Trust
    _MetricSpec(
        metric="trust_score",
        label="Trust score",
        section="Trust",
        direction="higher_better",
        extract="trust_score.score",
        detail="{symbol} trust composite: {value:.1f} / 100.",
        tie_within=1.0,
    ),
    _MetricSpec(
        metric="provider_confidence",
        label="Provider confidence",
        section="Trust",
        direction="higher_better",
        extract="fundamentals_freshness.provider_confidence",
        detail="{symbol} chain confidence after reconciliation: {value:.2f}.",
        tie_within=0.02,
    ),
    # Valuation
    _MetricSpec(
        metric="valuation_confidence",
        label="Valuation confidence",
        section="Valuation",
        direction="higher_better",
        extract="valuation.confidence_score",
        detail="{symbol} valuation ensemble confidence: {value:.1f} / 100.",
        tie_within=2.0,
    ),
    _MetricSpec(
        metric="upside_to_base_case",
        label="Upside to base case",
        section="Valuation",
        direction="higher_better",
        extract="__derived__.upside_pct",
        detail="{symbol} base-case fair value implies {value:.1f}% from last price.",
        percent=True,
        tie_within=0.5,
    ),
    # Technical
    _MetricSpec(
        metric="overbought_score",
        label="Overbought score",
        section="Technical",
        direction="lower_better",
        extract="technicals.overbought_score",
        detail="{symbol} overbought composite: {value:.0f} / 100 (lower = better entry).",
        tie_within=2.0,
    ),
    _MetricSpec(
        metric="oversold_score",
        label="Oversold score",
        section="Technical",
        direction="higher_better",
        extract="technicals.oversold_score",
        detail="{symbol} oversold composite: {value:.0f} / 100 (higher = stronger bounce setup).",
        tie_within=2.0,
    ),
    # Momentum / Risk
    _MetricSpec(
        metric="pullback_risk",
        label="Pullback risk",
        section="Risk",
        direction="lower_better",
        extract="technicals.pullback_risk",
        detail="{symbol} pullback risk: {value:.0f} / 100 (lower = safer hold).",
        tie_within=2.0,
    ),
    _MetricSpec(
        metric="rebound_potential",
        label="Rebound potential",
        section="Risk",
        direction="higher_better",
        extract="technicals.rebound_potential",
        detail="{symbol} rebound potential: {value:.0f} / 100.",
        tie_within=2.0,
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_reports(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> ComparisonView:
    """Build a :class:`ComparisonView` from two analyzer reports.

    Either side may be ``None`` (e.g. the API couldn't assemble the
    report for that symbol). In that case every metric on that side
    resolves to ``None`` and the other side wins by default. If both
    sides are ``None`` the view is empty but well-shaped — callers
    can still render the header and warnings.

    Symbol labels are pulled from the dicts so a missing-side caller
    still passes the symbol verbatim to the response.
    """
    if left is None and right is None:
        return ComparisonView(
            left_symbol="",
            right_symbol="",
            headline=(),
            sections=(),
            overall_winner="n/a",
            warnings=("Both reports unavailable.",),
        )

    left_symbol = _safe_get(left, "symbol", default="—") or "—"
    right_symbol = _safe_get(right, "symbol", default="—") or "—"

    warnings: list[str] = []
    if left is None:
        warnings.append(f"{right_symbol}: only one report available — comparison degraded.")
    if right is None:
        warnings.append(f"{left_symbol}: only one report available — comparison degraded.")

    # Build all per-metric comparisons up front so the section group
    # step and the overall-winner tally read the same data.
    rows = [
        _build_metric(spec, left, right, left_symbol, right_symbol)
        for spec in _METRIC_SPECS
    ]
    sections = _group_into_sections(rows)
    headline = _build_headline(left, right, left_symbol, right_symbol)
    overall = _tally_overall(rows)

    return ComparisonView(
        left_symbol=left_symbol,
        right_symbol=right_symbol,
        headline=headline,
        sections=sections,
        overall_winner=overall,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Per-metric build
# ---------------------------------------------------------------------------


def _build_metric(
    spec: _MetricSpec,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    left_symbol: str,
    right_symbol: str,
) -> MetricComparison:
    left_value = _extract(left, spec.extract)
    right_value = _extract(right, spec.extract)
    winner = _decide_winner(spec, left_value, right_value)
    detail = _format_detail(spec, left_symbol, right_symbol, left_value, right_value)
    return MetricComparison(
        metric=spec.metric,
        label=spec.label,
        left_value=left_value,
        right_value=right_value,
        winner=winner,
        direction=spec.direction,
        detail=detail,
        percent=spec.percent,
    )


def _decide_winner(
    spec: _MetricSpec, left: float | None, right: float | None
) -> Winner:
    if left is None and right is None:
        return "n/a"
    if left is None:
        return "right"
    if right is None:
        return "left"
    if abs(left - right) <= spec.tie_within:
        return "tie"
    if spec.direction == "higher_better":
        return "left" if left > right else "right"
    return "left" if left < right else "right"


def _format_detail(
    spec: _MetricSpec,
    left_symbol: str,
    right_symbol: str,
    left: float | None,
    right: float | None,
) -> str:
    parts: list[str] = []
    if left is not None:
        parts.append(spec.detail.format(symbol=left_symbol, value=left))
    if right is not None:
        parts.append(spec.detail.format(symbol=right_symbol, value=right))
    if not parts:
        return f"No {spec.label.lower()} data on either side."
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Headline + grouping
# ---------------------------------------------------------------------------


def _build_headline(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    left_symbol: str,
    right_symbol: str,
) -> tuple[HeadlineMetric, ...]:
    """Format the hero strip — grade, last price, overall score.

    These read off the same fields as the metric rows but the hero
    needs display-ready strings ("A+", "$182.43") rather than raw
    floats. Keep the formatting here so the frontend stays declarative.
    """
    headline: list[HeadlineMetric] = []
    # Trust grade
    l_grade = _safe_str(_extract_raw(left, "trust_score.grade"))
    r_grade = _safe_str(_extract_raw(right, "trust_score.grade"))
    headline.append(
        HeadlineMetric(
            label="Trust grade",
            left_display=l_grade or "—",
            right_display=r_grade or "—",
            winner=_grade_winner(l_grade, r_grade),
        )
    )
    # Last price
    l_price = _extract(left, "last_price")
    r_price = _extract(right, "last_price")
    headline.append(
        HeadlineMetric(
            label="Last price",
            left_display=_fmt_price(l_price),
            right_display=_fmt_price(r_price),
            winner="n/a",  # higher price isn't "better" — informational only.
        )
    )
    # Overall analyzer score
    l_overall = _extract(left, "overall_analyzer_score")
    r_overall = _extract(right, "overall_analyzer_score")
    headline.append(
        HeadlineMetric(
            label="Overall analyzer",
            left_display=_fmt_score(l_overall),
            right_display=_fmt_score(r_overall),
            winner=_higher_wins(l_overall, r_overall, tie_within=2.0),
        )
    )
    return tuple(headline)


def _group_into_sections(
    rows: list[MetricComparison],
) -> tuple[ComparisonSection, ...]:
    """Re-group the flat metric list by ``section`` while preserving
    the order declared in ``_METRIC_SPECS``."""
    spec_section_by_metric = {s.metric: s.section for s in _METRIC_SPECS}
    section_order: list[str] = []
    rows_by_section: dict[str, list[MetricComparison]] = {}
    for row in rows:
        section = spec_section_by_metric[row.metric]
        if section not in rows_by_section:
            section_order.append(section)
            rows_by_section[section] = []
        rows_by_section[section].append(row)
    return tuple(
        ComparisonSection(title=title, metrics=tuple(rows_by_section[title]))
        for title in section_order
    )


def _tally_overall(rows: list[MetricComparison]) -> Winner:
    """Pick an overall winner by simple majority of non-tie/n/a verdicts.

    Ties on the tally collapse to ``"tie"``. When no metric resolved
    (both sides missing everything), the overall winner is ``"n/a"``.
    """
    left_count = sum(1 for r in rows if r.winner == "left")
    right_count = sum(1 for r in rows if r.winner == "right")
    if left_count == 0 and right_count == 0:
        return "n/a"
    if left_count == right_count:
        return "tie"
    return "left" if left_count > right_count else "right"


# ---------------------------------------------------------------------------
# Dict walking + display helpers
# ---------------------------------------------------------------------------


def _extract(report: dict[str, Any] | None, path: str) -> float | None:
    """Walk a dotted path and coerce to ``float | None``.

    Supports a single derived path: ``"__derived__.upside_pct"`` —
    computed from the report's base_case + last_price. Add new
    derived keys as the comparison contract grows.
    """
    if report is None:
        return None
    if path == "__derived__.upside_pct":
        return _derived_upside_pct(report)
    raw = _extract_raw(report, path)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_raw(report: dict[str, Any] | None, path: str) -> Any:
    if report is None:
        return None
    node: Any = report
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


def _derived_upside_pct(report: dict[str, Any]) -> float | None:
    """``(base_case - last_price) / last_price * 100``.

    Returns ``None`` when either input is missing or last_price is
    non-positive (avoid divide-by-zero, also non-sensical for a real
    listed security).
    """
    base = _extract_raw(report, "valuation.base_case")
    last = _extract_raw(report, "last_price")
    if base is None or last is None:
        return None
    try:
        base_f = float(base)
        last_f = float(last)
    except (TypeError, ValueError):
        return None
    if last_f <= 0:
        return None
    return (base_f - last_f) / last_f * 100.0


def _safe_get(d: dict[str, Any] | None, key: str, *, default: Any = None) -> Any:
    if d is None:
        return default
    return d.get(key, default)


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}"


def _higher_wins(
    left: float | None, right: float | None, *, tie_within: float
) -> Winner:
    if left is None and right is None:
        return "n/a"
    if left is None:
        return "right"
    if right is None:
        return "left"
    if abs(left - right) <= tie_within:
        return "tie"
    return "left" if left > right else "right"


# Letter grades sort A+ > A > B > C > D > F; use an explicit order so
# the ordering isn't accidentally alphabetic.
_GRADE_ORDER = {"A+": 6, "A": 5, "B": 4, "C": 3, "D": 2, "F": 1}


def _grade_winner(left: str | None, right: str | None) -> Winner:
    if left is None and right is None:
        return "n/a"
    if left is None:
        return "right"
    if right is None:
        return "left"
    l_rank = _GRADE_ORDER.get(left, 0)
    r_rank = _GRADE_ORDER.get(right, 0)
    if l_rank == r_rank:
        return "tie"
    return "left" if l_rank > r_rank else "right"


__all__ = [
    "ComparisonSection",
    "ComparisonView",
    "HeadlineMetric",
    "MetricComparison",
    "compare_reports",
]
