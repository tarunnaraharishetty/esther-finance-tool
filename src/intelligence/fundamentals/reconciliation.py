"""Cross-provider reconciliation on high-trust valuation inputs.

When the primary provider in the fundamentals chain is *not* the
preferred one — e.g., FMP timed out and we fell through to Finnhub —
we make a single follow-up call to one *other* configured provider
and compare the two payloads on the four fields that most directly
drive valuation:

* ``revenue`` — top line.
* ``net_income`` — bottom line.
* ``eps_diluted`` — per-share earnings (or basic when diluted is None).
* ``total_debt`` — leverage proxy on the balance sheet.

We never auto-pick a "correct" value when the two disagree. We
surface :class:`FieldDivergence` rows so the UI can render
"providers disagree on revenue by 12%" and the analyzer can
down-weight the data. The trader decides what to do — that's the
institutional contract.

Why these four
--------------
Each of the seven valuation methods in
:mod:`src.intelligence.analyzer.valuation` consumes at least one of
the four:

* DCF needs free cash flow → derivable from net income + ratios.
* P/E needs EPS.
* EV/EBITDA needs EBITDA → net income + non-cash adjustments; also
  needs total debt for the EV calc.
* P/S needs revenue.
* PEG, historical band — EPS.

So a divergence on any of these four propagates through valuation;
divergences on capex or working-capital lines don't. Targeting the
high-trust set keeps the secondary call cheap and the noise low.

Why same fiscal date is required
--------------------------------
Different providers update at different speeds. FMP's latest annual
might be FY2024 while Yahoo's is still on FY2023. Comparing those is
apples-to-oranges — it's a release-timing artifact, not a data
disagreement. We surface it as a :class:`ReconciliationWarning`
rather than a divergence so callers can show "FMP is one fiscal
period ahead" without polluting the divergence list.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from src.intelligence.fundamentals.models import (
    AccuracyEvent,
    BalanceSheet,
    IncomeStatement,
    NormalizedFundamentals,
)


@dataclass(frozen=True)
class FieldDivergence:
    """One disagreement on one high-trust field.

    ``relative_divergence`` is ``|p - s| / max(|p|, |s|)`` — symmetric
    in primary vs secondary and scale-invariant. A sign-flip pair
    (one positive, one negative) always exceeds any non-zero threshold
    and is reported regardless.
    """

    field: str
    primary_value: float
    secondary_value: float
    primary_provider: str
    secondary_provider: str
    relative_divergence: float
    fiscal_date: datetime


@dataclass(frozen=True)
class ReconciliationWarning:
    """Operator-visible note explaining why a field was NOT compared.

    Distinct from :class:`FieldDivergence` — a warning means we found
    something irregular but it's not in itself a data-quality flag.
    Most common cause: providers' latest statements have different
    fiscal dates.
    """

    field: str
    reason: str


@dataclass(frozen=True)
class ReconciliationResult:
    """Aggregate output of :func:`reconcile`.

    ``events`` is the empirical accuracy stream — one entry per
    comparable (field, primary, secondary) tuple, recording both
    agreements and disagreements. Persisted by the service to the
    :class:`AccuracyStore`; powers the provider trust weighting.
    """

    divergences: tuple[FieldDivergence, ...]
    warnings: tuple[ReconciliationWarning, ...]
    events: tuple[AccuracyEvent, ...] = ()


# How to pull each high-trust field off a NormalizedFundamentals.
# Tuple element 1 selects which statement family the field lives on:
# "income" or "balance". This drives the same-fiscal-date check —
# we compare income-statement fields against income statements and
# balance-sheet fields against balance sheets, not across families.
_IncomeExtractor = Callable[[IncomeStatement], float | None]
_BalanceExtractor = Callable[[BalanceSheet], float | None]


def _eps_extractor(stmt: IncomeStatement) -> float | None:
    """Prefer diluted; fall back to basic when diluted is missing.

    Mirrors :func:`src.intelligence.analyzer.valuation._diluted_eps`'s
    fallback. A provider that only emits basic EPS shouldn't be
    silently excluded from reconciliation.
    """
    if stmt.eps_diluted is not None:
        return float(stmt.eps_diluted)
    if stmt.eps_basic is not None:
        return float(stmt.eps_basic)
    return None


_HIGH_TRUST_INCOME_FIELDS: dict[str, _IncomeExtractor] = {
    "revenue": lambda s: float(s.revenue) if s.revenue is not None else None,
    "net_income": lambda s: float(s.net_income) if s.net_income is not None else None,
    "eps_diluted": _eps_extractor,
}

_HIGH_TRUST_BALANCE_FIELDS: dict[str, _BalanceExtractor] = {
    "total_debt": lambda s: float(s.total_debt) if s.total_debt is not None else None,
}


def reconcile(
    primary: NormalizedFundamentals,
    secondary: NormalizedFundamentals,
    *,
    threshold: float = 0.05,
) -> ReconciliationResult:
    """Compare ``primary`` against ``secondary`` on the high-trust fields.

    ``threshold`` is the relative-divergence ratio that triggers a
    :class:`FieldDivergence`. Sign-flip pairs (one positive, one
    negative) always fire regardless of magnitude — a net-income
    flip from $+1B to $-1B is operationally severe even though
    the relative divergence (computed against the larger magnitude)
    can be small if the numbers are close to zero.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold!r}")

    divergences: list[FieldDivergence] = []
    warnings: list[ReconciliationWarning] = []
    events: list[AccuracyEvent] = []
    # Single observation timestamp for every event in this call so
    # the ledger's rolling windows align cleanly per reconciliation.
    observed_at = datetime.now(UTC)

    primary_income = primary.latest_annual_income
    secondary_income = secondary.latest_annual_income
    _compare_family(
        family="income",
        primary_stmt=primary_income,
        secondary_stmt=secondary_income,
        extractors=_HIGH_TRUST_INCOME_FIELDS,
        primary=primary,
        secondary=secondary,
        threshold=threshold,
        divergences=divergences,
        warnings=warnings,
        events=events,
        observed_at=observed_at,
    )

    primary_balance = primary.latest_annual_balance
    secondary_balance = secondary.latest_annual_balance
    _compare_family(
        family="balance",
        primary_stmt=primary_balance,
        secondary_stmt=secondary_balance,
        extractors=_HIGH_TRUST_BALANCE_FIELDS,
        primary=primary,
        secondary=secondary,
        threshold=threshold,
        divergences=divergences,
        warnings=warnings,
        events=events,
        observed_at=observed_at,
    )

    return ReconciliationResult(
        divergences=tuple(divergences),
        warnings=tuple(warnings),
        events=tuple(events),
    )


def _compare_family(
    *,
    family: str,
    primary_stmt: IncomeStatement | BalanceSheet | None,
    secondary_stmt: IncomeStatement | BalanceSheet | None,
    extractors: dict[str, _IncomeExtractor] | dict[str, _BalanceExtractor],
    primary: NormalizedFundamentals,
    secondary: NormalizedFundamentals,
    threshold: float,
    divergences: list[FieldDivergence],
    warnings: list[ReconciliationWarning],
    events: list[AccuracyEvent],
    observed_at: datetime,
) -> None:
    """Run the comparison for one statement family (income or balance).

    The early-return checks codify the rules: if either side is
    missing the relevant statement, we surface a per-family warning
    and exit. If the fiscal dates don't match, we surface per-field
    warnings (so the operator can see exactly which numbers were
    not compared and why).
    """
    primary_name = primary.primary_provider.value
    secondary_name = secondary.primary_provider.value
    if primary_stmt is None or secondary_stmt is None:
        # Sparse coverage isn't disagreement. Worth flagging as a
        # warning so the operator can see that reconciliation
        # couldn't run on this family — useful for tracing why a
        # divergence list is empty.
        for field_name in extractors:
            warnings.append(
                ReconciliationWarning(
                    field=field_name,
                    reason=(
                        f"missing {family} statement on "
                        f"{primary_name if primary_stmt is None else secondary_name}"
                    ),
                )
            )
        return

    if primary_stmt.fiscal_date != secondary_stmt.fiscal_date:
        for field_name in extractors:
            warnings.append(
                ReconciliationWarning(
                    field=field_name,
                    reason=(
                        f"fiscal_date mismatch: "
                        f"{primary_name}={primary_stmt.fiscal_date.date().isoformat()} vs "
                        f"{secondary_name}={secondary_stmt.fiscal_date.date().isoformat()}"
                    ),
                )
            )
        return

    for field_name, extractor in extractors.items():
        # The extractors are statically typed per family but Python
        # treats them as a uniform Callable here. Cast at the call
        # site by branching on what statement type we have — both
        # branches are exhaustive over the family.
        primary_value: float | None = extractor(primary_stmt)  # type: ignore[arg-type]
        secondary_value: float | None = extractor(secondary_stmt)  # type: ignore[arg-type]
        if primary_value is None or secondary_value is None:
            # No comparable observation — neither divergence nor
            # agreement; skip the event too so the accuracy ledger
            # only carries real comparisons.
            continue

        divergence = _grade(primary_value, secondary_value, threshold)
        if divergence is not None:
            divergences.append(
                FieldDivergence(
                    field=field_name,
                    primary_value=primary_value,
                    secondary_value=secondary_value,
                    primary_provider=primary_name,
                    secondary_provider=secondary_name,
                    relative_divergence=divergence,
                    fiscal_date=primary_stmt.fiscal_date,
                )
            )

        # Compute the ledger-friendly view of this comparison: a
        # symmetric relative error plus an agreement flag. The event
        # is emitted regardless of whether the field diverged — the
        # accuracy ledger needs the denominator (total comparisons)
        # to compute a rate, not just the numerator (disagreements).
        rel_error = _relative_error(primary_value, secondary_value)
        agreed = divergence is None
        events.append(
            AccuracyEvent(
                provider=primary.primary_provider,
                reference_provider=secondary.primary_provider,
                symbol=primary.profile.symbol,
                field=field_name,
                observed_value=primary_value,
                reference_value=secondary_value,
                rel_error=rel_error,
                agreed=agreed,
                fiscal_date=primary_stmt.fiscal_date,
                observed_at=observed_at,
            )
        )


def _relative_error(primary_value: float, secondary_value: float) -> float:
    """Symmetric relative error in ``[0, 1]``. Returns 1.0 when both zero.

    Mirrors the divergence formula in :func:`_grade`; computed
    independently so the event always carries a number even when
    the field agreed (in which case the value is below threshold).
    """
    abs_p = abs(primary_value)
    abs_s = abs(secondary_value)
    scale = max(abs_p, abs_s)
    if scale == 0.0:
        return 0.0
    return abs(primary_value - secondary_value) / scale


def _grade(
    primary_value: float, secondary_value: float, threshold: float
) -> float | None:
    """Return the relative divergence when it fires, else None.

    Two firing rules:

    1. *Sign mismatch.* One value positive, one negative — always
       fires. Reported value is the relative divergence on the
       absolute magnitudes; never reports a magnitude of zero.
    2. *Magnitude exceeds threshold.* ``|p - s| / max(|p|, |s|) > threshold``.

    Returns ``None`` when both rules are inactive — values agree.
    """
    p_sign = (primary_value > 0) - (primary_value < 0)
    s_sign = (secondary_value > 0) - (secondary_value < 0)
    sign_flip = p_sign != 0 and s_sign != 0 and p_sign != s_sign

    abs_p = abs(primary_value)
    abs_s = abs(secondary_value)
    scale = max(abs_p, abs_s)
    if scale == 0:
        # Both exactly zero — agreement, not divergence.
        return None

    relative = abs(primary_value - secondary_value) / scale
    if sign_flip:
        return relative
    if relative > threshold:
        return relative
    return None


__all__ = [
    "FieldDivergence",
    "ReconciliationResult",
    "ReconciliationWarning",
    "reconcile",
]
