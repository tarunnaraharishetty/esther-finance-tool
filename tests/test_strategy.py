from __future__ import annotations

from datetime import UTC, datetime

from src.strategy.base import Signal, SignalAction
from src.strategy.signal_aggregator import SignalAggregator


def _sig(symbol: str, action: SignalAction, conf: float, source: str) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=conf,
        timestamp=datetime.now(UTC),
        source=source,
    )


def test_aggregator_buy_when_majority_buy() -> None:
    agg = SignalAggregator(threshold=0.2)
    out = agg.aggregate(
        [
            _sig("AAPL", SignalAction.BUY, 0.8, "rsi"),
            _sig("AAPL", SignalAction.BUY, 0.7, "sentiment"),
            _sig("AAPL", SignalAction.SELL, 0.5, "macd"),
        ]
    )
    assert len(out) == 1
    assert out[0].action == SignalAction.BUY


def test_aggregator_holds_when_below_threshold() -> None:
    agg = SignalAggregator(threshold=0.5)
    out = agg.aggregate(
        [
            _sig("AAPL", SignalAction.BUY, 0.4, "rsi"),
            _sig("AAPL", SignalAction.SELL, 0.3, "sentiment"),
        ]
    )
    assert out[0].action == SignalAction.HOLD


def test_aggregator_respects_weights() -> None:
    agg = SignalAggregator(weights={"sentiment": 5.0, "rsi": 1.0}, threshold=0.2)
    out = agg.aggregate(
        [
            _sig("AAPL", SignalAction.BUY, 0.5, "rsi"),
            _sig("AAPL", SignalAction.SELL, 0.9, "sentiment"),
        ]
    )
    assert out[0].action == SignalAction.SELL
