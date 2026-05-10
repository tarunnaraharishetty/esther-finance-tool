from __future__ import annotations

from decimal import Decimal

from src.execution.position_sizer import FixedFractionSizer


def test_fixed_fraction_sizer_basic() -> None:
    sizer = FixedFractionSizer(fraction=0.05)
    qty = sizer.size(equity=Decimal(100_000), price=Decimal(200))
    assert qty == Decimal("25.0000")


def test_fixed_fraction_sizer_zero_price_returns_zero() -> None:
    sizer = FixedFractionSizer(fraction=0.05)
    assert sizer.size(equity=Decimal(100_000), price=Decimal(0)) == Decimal(0)
