"""Combine signals from multiple sources (sentiment + technicals)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.strategy.base import Signal, SignalAction


@dataclass
class SignalAggregator:
    """Weighted-vote aggregator over per-source signals.

    ``weights`` maps signal ``source`` strings to a non-negative weight.
    Sources not in the map default to weight 1.0.
    """

    weights: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.2  # net score required to emit non-HOLD

    def aggregate(self, signals: list[Signal]) -> list[Signal]:
        by_symbol: dict[str, list[Signal]] = defaultdict(list)
        for s in signals:
            by_symbol[s.symbol].append(s)

        out: list[Signal] = []
        for symbol, group in by_symbol.items():
            score = 0.0
            total_weight = 0.0
            for s in group:
                w = self.weights.get(s.source, 1.0)
                total_weight += w
                if s.action == SignalAction.BUY:
                    score += w * s.confidence
                elif s.action == SignalAction.SELL:
                    score -= w * s.confidence
            net = score / total_weight if total_weight else 0.0

            if net >= self.threshold:
                action = SignalAction.BUY
            elif net <= -self.threshold:
                action = SignalAction.SELL
            else:
                action = SignalAction.HOLD

            latest = max(group, key=lambda s: s.timestamp)
            out.append(
                Signal(
                    symbol=symbol,
                    action=action,
                    confidence=min(1.0, abs(net)),
                    timestamp=latest.timestamp,
                    source="aggregator",
                    rationale=f"net={net:.3f} from {len(group)} inputs",
                    metadata={"inputs": [s.source for s in group]},
                )
            )
        return out
