from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd

from src.risk.exposure import RiskGate
from src.risk.metrics import max_drawdown, sharpe_ratio
from src.risk.var import historical_var
from src.strategy.base import Signal, SignalAction


def test_sharpe_zero_for_zero_variance() -> None:
    assert sharpe_ratio(pd.Series([0.001] * 100)) == 0.0


def test_sharpe_positive_for_positive_drift(synthetic_returns: pd.Series) -> None:
    assert sharpe_ratio(synthetic_returns) > 0


def test_max_drawdown_non_negative(equity_curve: pd.Series) -> None:
    assert max_drawdown(equity_curve) >= 0.0


def test_historical_var_positive_for_loss_returns() -> None:
    rets = pd.Series([-0.05, -0.04, -0.02, 0.01, 0.0, 0.02, -0.03, -0.06] * 10)
    var = historical_var(rets, confidence=0.95)
    assert var > 0


def test_risk_gate_blocks_zero_equity() -> None:
    gate = RiskGate()
    sig = Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.9,
        timestamp=datetime.now(UTC),
        source="test",
    )
    result = gate.check(signal=sig, equity=Decimal(0), price=Decimal(150))
    assert not result.allowed


def test_risk_gate_allows_normal_trade() -> None:
    gate = RiskGate()
    sig = Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.9,
        timestamp=datetime.now(UTC),
        source="test",
    )
    result = gate.check(signal=sig, equity=Decimal(100_000), price=Decimal(150))
    assert result.allowed
