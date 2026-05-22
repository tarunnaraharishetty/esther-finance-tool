"""Sector-relative ranking across analyzer reports.

Given a collection of analyzer-report dicts (the wire shape from
``/api/analyzer/{symbol}``), groups symbols by sector and ranks each
one on every metric the ranker tracks. The output is per-symbol — the
endpoint picks one entry to return — but the computation is
cohort-wide so percentiles share a denominator and rank ordering is
consistent.

Why rank from analyzer reports
------------------------------
Analyzer reports already carry the post-pipeline signals we want to
rank on: trust score, valuation upside, technical sub-scores, provider
confidence. Recomputing those from raw inputs here would duplicate the
analyzer's grounding contract and risk drift. The ranker operates
purely over the assembled dict — same source of truth as the Trust
Score badge and the Compare verdict.

Cohort definition
-----------------
A "cohort" is every symbol that normalizes onto the same sector key
via :func:`src.intelligence.analyzer.sector_medians.normalize_sector_key`.
Symbols without a usable sector are excluded from cohort membership
(they get an empty :class:`SymbolRanking` so the caller can still
render "no peers known" without branching). Singleton cohorts (only
this symbol in the sector) also return empty rankings — a "#1 of 1"
read is meaningless and would mislead.

Direction semantics
-------------------
Each metric declares ``higher_better`` or ``lower_better``. ``rank=1``
is always the *best* on the metric per its direction. ``percentile``
is in [0, 100] where 100 is the best (top of the cohort on this
metric). This matches the framing of the compare verdicts so a trader
who scans both surfaces sees one consistent "what wins" model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.intelligence.analyzer.sector_medians import normalize_sector_key

Direction = Literal["higher_better", "lower_better"]


@dataclass(frozen=True)
class _MetricSpec:
    """Declarative recipe for one ranked metric.

    ``extract`` is a dotted path into the analyzer report dict, with
    the special prefix ``"__derived__."`` for computed metrics that
    don't sit on a single field (today: ``upside_pct``).
    """

    metric: str
    label: str
    direction: Direction
    extract: str
    # Whether the metric is naturally percent-shaped on the UI.
    percent: bool = False


# Eight metrics; intentionally the same set the comparison module
# ranks on so a trader sees one consistent cross-section across
# Compare and Sector Rank. Adding a metric here = one entry + a UI
# row + a test fixture; the ranker is direction-agnostic.
SECTOR_RANK_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec(
        metric="trust_score",
        label="Trust score",
        direction="higher_better",
        extract="trust_score.score",
    ),
    _MetricSpec(
        metric="provider_confidence",
        label="Provider confidence",
        direction="higher_better",
        extract="fundamentals_freshness.provider_confidence",
    ),
    _MetricSpec(
        metric="valuation_confidence",
        label="Valuation confidence",
        direction="higher_better",
        extract="valuation.confidence_score",
    ),
    _MetricSpec(
        metric="upside_to_base_case",
        label="Upside to base case",
        direction="higher_better",
        extract="__derived__.upside_pct",
        percent=True,
    ),
    _MetricSpec(
        metric="overbought_score",
        label="Overbought score",
        direction="lower_better",
        extract="technicals.overbought_score",
    ),
    _MetricSpec(
        metric="oversold_score",
        label="Oversold score",
        direction="higher_better",
        extract="technicals.oversold_score",
    ),
    _MetricSpec(
        metric="pullback_risk",
        label="Pullback risk",
        direction="lower_better",
        extract="technicals.pullback_risk",
    ),
    _MetricSpec(
        metric="rebound_potential",
        label="Rebound potential",
        direction="higher_better",
        extract="technicals.rebound_potential",
    ),
)


@dataclass(frozen=True)
class MetricRank:
    """One symbol's rank on one metric within its sector cohort.

    Attributes:
        metric: Metric id (matches :data:`SECTOR_RANK_METRICS`).
        label: Display label.
        value: The symbol's own value on this metric; ``None`` when
            the underlying field was missing for this symbol.
        rank: 1-indexed position within the cohort, where ``1`` is
            the best per the metric's direction. ``None`` when
            ``value`` is missing — a symbol with no data has no rank.
        cohort_size: Number of symbols in the cohort that produced a
            comparable value on this metric. Always equal across
            symbols in the same cohort *for this metric*; varies
            across metrics within the cohort when some symbols are
            missing some fields.
        percentile: ``0-100`` where 100 = best on this metric.
            Computed as ``(cohort_size - rank) / (cohort_size - 1) * 100``;
            ``None`` when ``cohort_size <= 1`` (no spread to measure).
        direction: How to interpret the metric (higher/lower better).
        percent: True when the metric renders as a percentage.
    """

    metric: str
    label: str
    value: float | None
    rank: int | None
    cohort_size: int
    percentile: float | None
    direction: Direction
    percent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "label": self.label,
            "value": self.value,
            "rank": self.rank,
            "cohort_size": self.cohort_size,
            "percentile": (
                None if self.percentile is None else round(self.percentile, 1)
            ),
            "direction": self.direction,
            "percent": self.percent,
        }


@dataclass(frozen=True)
class SymbolRanking:
    """All metric ranks for one symbol within its cohort.

    ``cohort_symbols`` is the full set of peers (including the symbol
    itself) — useful for the UI to render "AAPL vs MSFT, GOOGL, …" if
    desired. Order is alphabetic for determinism.

    ``cohort_size`` here is the *full* cohort count; per-metric
    cohort sizes (in :class:`MetricRank`) can be smaller when peers
    are missing data on individual fields.
    """

    symbol: str
    sector: str | None
    cohort_size: int
    cohort_symbols: tuple[str, ...]
    metrics: tuple[MetricRank, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "cohort_size": self.cohort_size,
            "cohort_symbols": list(self.cohort_symbols),
            "metrics": [m.to_dict() for m in self.metrics],
        }


def empty_ranking(symbol: str, sector: str | None) -> SymbolRanking:
    """Build a degraded SymbolRanking with no metrics.

    Used when the cohort is unusable (singleton or no peers). The
    caller surfaces this as "no sector peers available" rather than
    inventing a meaningless "#1 of 1" read.
    """
    return SymbolRanking(
        symbol=symbol.upper(),
        sector=sector,
        cohort_size=0,
        cohort_symbols=(),
        metrics=(),
    )


def compute_sector_rankings(
    reports: list[dict[str, Any]],
) -> dict[str, SymbolRanking]:
    """Group ``reports`` by sector and rank each symbol within its cohort.

    Symbols whose report has no usable sector key are excluded from
    cohort grouping and get a degraded :class:`SymbolRanking` with
    empty metrics. Same for singleton cohorts. The output dict is
    keyed by symbol (uppercased).
    """
    by_sector: dict[str, list[dict[str, Any]]] = {}
    sector_per_symbol: dict[str, str | None] = {}
    for report in reports:
        symbol = str(report.get("symbol", "")).upper()
        if not symbol:
            continue
        sector_raw = _extract_sector(report)
        sector_key = normalize_sector_key(sector_raw)
        sector_per_symbol[symbol] = sector_raw
        if sector_key is None:
            continue
        by_sector.setdefault(sector_key, []).append(report)

    out: dict[str, SymbolRanking] = {}
    for sector_key, cohort in by_sector.items():
        if len(cohort) < 2:
            # Singleton cohort — meaningful ranking requires peers.
            sym = str(cohort[0].get("symbol", "")).upper()
            out[sym] = empty_ranking(sym, sector_per_symbol.get(sym))
            continue
        for ranking in _rank_cohort(cohort, sector_key):
            out[ranking.symbol] = ranking

    # Symbols that never entered a cohort still need an entry so
    # callers can render the "no sector peers" state.
    for symbol, sector_raw in sector_per_symbol.items():
        if symbol not in out:
            out[symbol] = empty_ranking(symbol, sector_raw)
    return out


# ---------------------------------------------------------------------------
# Per-cohort ranking
# ---------------------------------------------------------------------------


def _rank_cohort(
    cohort: list[dict[str, Any]], sector_key: str
) -> list[SymbolRanking]:
    """Run every metric's ranking over ``cohort`` and assemble per-symbol rows."""
    symbols = sorted(
        {str(r.get("symbol", "")).upper() for r in cohort if r.get("symbol")}
    )
    # Per-symbol values per metric.
    values_by_metric: dict[str, dict[str, float | None]] = {}
    for spec in SECTOR_RANK_METRICS:
        values_by_metric[spec.metric] = {
            str(r.get("symbol", "")).upper(): _extract_metric(r, spec)
            for r in cohort
            if r.get("symbol")
        }

    # Per-symbol rank/percentile per metric, computed once per metric
    # so the ordering is consistent.
    per_symbol_ranks: dict[str, list[MetricRank]] = {sym: [] for sym in symbols}
    for spec in SECTOR_RANK_METRICS:
        symbol_values = values_by_metric[spec.metric]
        ranked = _rank_one_metric(spec, symbol_values)
        for symbol, metric_rank in ranked.items():
            per_symbol_ranks[symbol].append(metric_rank)

    rankings: list[SymbolRanking] = []
    cohort_size = len(symbols)
    raw_sector_by_symbol: dict[str, str | None] = {
        str(r.get("symbol", "")).upper(): _extract_sector(r) for r in cohort
    }
    for sym in symbols:
        rankings.append(
            SymbolRanking(
                symbol=sym,
                sector=raw_sector_by_symbol.get(sym) or sector_key,
                cohort_size=cohort_size,
                cohort_symbols=tuple(symbols),
                metrics=tuple(per_symbol_ranks[sym]),
            )
        )
    return rankings


def _rank_one_metric(
    spec: _MetricSpec, symbol_values: dict[str, float | None]
) -> dict[str, MetricRank]:
    """Compute rank + percentile for one metric across the cohort.

    Symbols whose value is ``None`` get a :class:`MetricRank` with
    ``rank=None``, ``percentile=None``, and the metric's metadata —
    so the UI can render "no data" without falling back to a missing
    row. ``cohort_size`` for this metric counts only the symbols
    that have a comparable value.
    """
    # Partition into "has value" + "missing".
    populated = [(sym, val) for sym, val in symbol_values.items() if val is not None]
    cohort_size = len(populated)
    out: dict[str, MetricRank] = {}

    if cohort_size == 0:
        # Nobody in the cohort has data on this metric — every entry
        # is a no-data row with cohort_size=0.
        for sym in symbol_values:
            out[sym] = MetricRank(
                metric=spec.metric,
                label=spec.label,
                value=None,
                rank=None,
                cohort_size=0,
                percentile=None,
                direction=spec.direction,
                percent=spec.percent,
            )
        return out

    # Sort populated symbols best-first per direction.
    reverse = spec.direction == "higher_better"
    # Tuple sort key: (value, symbol) so identical values still produce
    # stable, deterministic rank assignment.
    populated.sort(key=lambda pair: (pair[1], pair[0]), reverse=reverse)

    # Assign ranks. Ties get the same rank ("1, 1, 3" style) so the
    # UI can honestly say "tied for #1" if both symbols are dead-even.
    last_value: float | None = None
    last_rank = 0
    rank_by_symbol: dict[str, int] = {}
    for index, (sym, value) in enumerate(populated, start=1):
        if last_value is not None and value == last_value:
            rank_by_symbol[sym] = last_rank
        else:
            rank_by_symbol[sym] = index
            last_rank = index
        last_value = value

    for sym, value in populated:
        rank = rank_by_symbol[sym]
        if cohort_size <= 1:
            percentile: float | None = None
        else:
            # Percentile: 100 means best (rank 1). Formula maps
            # rank=1 → 100, rank=N → 0 with linear interpolation.
            percentile = ((cohort_size - rank) / (cohort_size - 1)) * 100.0
        out[sym] = MetricRank(
            metric=spec.metric,
            label=spec.label,
            value=value,
            rank=rank,
            cohort_size=cohort_size,
            percentile=percentile,
            direction=spec.direction,
            percent=spec.percent,
        )
    # Symbols with no value get a no-data row carrying cohort_size
    # so the UI can show "— · 12 peers ranked" if useful.
    for sym in symbol_values:
        if sym not in out:
            out[sym] = MetricRank(
                metric=spec.metric,
                label=spec.label,
                value=None,
                rank=None,
                cohort_size=cohort_size,
                percentile=None,
                direction=spec.direction,
                percent=spec.percent,
            )
    return out


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_sector(report: dict[str, Any]) -> str | None:
    """Pull the sector string off the analyzer report."""
    profile = report.get("fundamentals_profile")
    if not isinstance(profile, dict):
        return None
    sector = profile.get("sector")
    if isinstance(sector, str) and sector.strip():
        return sector
    return None


def _extract_metric(report: dict[str, Any], spec: _MetricSpec) -> float | None:
    """Walk the metric spec's dotted path and coerce to ``float | None``."""
    if spec.extract == "__derived__.upside_pct":
        return _derived_upside_pct(report)
    raw = _walk(report, spec.extract)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _walk(report: dict[str, Any], path: str) -> Any:
    node: Any = report
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


def _derived_upside_pct(report: dict[str, Any]) -> float | None:
    base = _walk(report, "valuation.base_case")
    last = report.get("last_price")
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


__all__ = [
    "SECTOR_RANK_METRICS",
    "MetricRank",
    "SymbolRanking",
    "compute_sector_rankings",
    "empty_ranking",
]
