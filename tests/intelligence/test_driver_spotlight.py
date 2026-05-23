"""Tests for the watchlist Driver Spotlight composer.

Covers ranking across symbols, per-symbol cap, total cap, alphabetic
tie-break, deterministic ordering for the file cache, wire shape,
and edge cases (empty input, all baseline, validation errors).
"""

from __future__ import annotations

import pytest

from src.intelligence.driver_spotlight import (
    SpotlightEntry,
    SpotlightView,
    compute_spotlight,
)
from src.intelligence.movement_drivers import Driver, KeyDrivers


def _driver(
    *,
    kind: str = "technical",
    label: str = "Overbought technicals",
    tone: str = "warn",
    salience: float = 0.5,
    citations: tuple[str, ...] = ("technicals",),
) -> Driver:
    return Driver(
        kind=kind,  # type: ignore[arg-type]
        label=label,
        detail=f"{label} composite detail.",
        tone=tone,  # type: ignore[arg-type]
        salience=salience,
        citations=citations,
    )


def _key_drivers(symbol: str, *drivers: Driver) -> KeyDrivers:
    return KeyDrivers(
        symbol=symbol.upper(),
        drivers=drivers,
        considered=len(drivers),
        suppressed=0,
    )


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_empty_input_returns_well_shaped_empty_view() -> None:
    view = compute_spotlight({})
    assert isinstance(view, SpotlightView)
    assert view.entries == ()
    assert view.considered_symbols == 0
    assert view.surfaced_symbols == 0


def test_to_dict_round_trip_shape() -> None:
    view = compute_spotlight(
        {"AAPL": _key_drivers("AAPL", _driver(salience=0.8))}
    )
    wire = view.to_dict()
    assert set(wire.keys()) == {
        "entries",
        "considered_symbols",
        "surfaced_symbols",
    }
    for entry in wire["entries"]:
        assert set(entry.keys()) == {"rank", "symbol", "driver"}
        driver = entry["driver"]
        assert set(driver.keys()) == {
            "kind",
            "label",
            "detail",
            "tone",
            "salience",
            "citations",
        }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_entries_sorted_by_salience_descending() -> None:
    view = compute_spotlight(
        {
            "AAA": _key_drivers("AAA", _driver(salience=0.6)),
            "BBB": _key_drivers("BBB", _driver(salience=0.9)),
            "CCC": _key_drivers("CCC", _driver(salience=0.3)),
        }
    )
    saliences = [e.driver.salience for e in view.entries]
    assert saliences == sorted(saliences, reverse=True)
    assert [e.symbol for e in view.entries] == ["BBB", "AAA", "CCC"]


def test_alphabetic_tie_break_on_symbol() -> None:
    """Equal salience → alphabetic symbol order so cache keys stay stable."""
    view = compute_spotlight(
        {
            "ZZZ": _key_drivers("ZZZ", _driver(salience=0.7)),
            "AAA": _key_drivers("AAA", _driver(salience=0.7)),
            "MMM": _key_drivers("MMM", _driver(salience=0.7)),
        }
    )
    assert [e.symbol for e in view.entries] == ["AAA", "MMM", "ZZZ"]


def test_rank_is_one_indexed() -> None:
    view = compute_spotlight(
        {
            "AAA": _key_drivers("AAA", _driver(salience=0.5)),
            "BBB": _key_drivers("BBB", _driver(salience=0.9)),
        }
    )
    assert view.entries[0].rank == 1
    assert view.entries[1].rank == 2


# ---------------------------------------------------------------------------
# Per-symbol cap
# ---------------------------------------------------------------------------


def test_default_per_symbol_cap_picks_only_strongest_driver() -> None:
    """A symbol with multiple high-salience drivers contributes only its
    loudest one to the spotlight (default cap = 1)."""
    view = compute_spotlight(
        {
            "AAA": _key_drivers(
                "AAA",
                _driver(label="Driver 1", salience=0.9),
                _driver(label="Driver 2", salience=0.85),
                _driver(label="Driver 3", salience=0.8),
            ),
            "BBB": _key_drivers("BBB", _driver(salience=0.6)),
        }
    )
    aaa_entries = [e for e in view.entries if e.symbol == "AAA"]
    assert len(aaa_entries) == 1
    # The single AAA entry is the highest-salience driver.
    assert aaa_entries[0].driver.label == "Driver 1"


def test_per_symbol_cap_two_picks_top_two_per_symbol() -> None:
    view = compute_spotlight(
        {
            "AAA": _key_drivers(
                "AAA",
                _driver(label="A1", salience=0.9),
                _driver(label="A2", salience=0.85),
                _driver(label="A3", salience=0.7),
            ),
            "BBB": _key_drivers("BBB", _driver(salience=0.6)),
        },
        max_per_symbol=2,
    )
    aaa_entries = [e for e in view.entries if e.symbol == "AAA"]
    assert len(aaa_entries) == 2
    labels = {e.driver.label for e in aaa_entries}
    assert labels == {"A1", "A2"}


# ---------------------------------------------------------------------------
# Total cap
# ---------------------------------------------------------------------------


def test_max_items_caps_total_entries() -> None:
    """max_items=3 with 5 candidate symbols yields exactly 3 entries."""
    view = compute_spotlight(
        {
            f"SYM{i}": _key_drivers(f"SYM{i}", _driver(salience=0.5 + i * 0.05))
            for i in range(5)
        },
        max_items=3,
    )
    assert len(view.entries) == 3


def test_max_items_zero_returns_no_entries() -> None:
    view = compute_spotlight(
        {"AAA": _key_drivers("AAA", _driver(salience=0.9))}, max_items=0
    )
    assert view.entries == ()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_negative_max_items_rejected() -> None:
    with pytest.raises(ValueError):
        compute_spotlight({}, max_items=-1)


def test_negative_max_per_symbol_rejected() -> None:
    with pytest.raises(ValueError):
        compute_spotlight({}, max_per_symbol=-1)


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_considered_symbols_counts_input_dict_size() -> None:
    """Even symbols with empty drivers count toward ``considered``."""
    view = compute_spotlight(
        {
            "AAA": _key_drivers("AAA"),  # no drivers
            "BBB": _key_drivers("BBB", _driver(salience=0.7)),
        }
    )
    assert view.considered_symbols == 2
    assert view.surfaced_symbols == 1
    assert {e.symbol for e in view.entries} == {"BBB"}


def test_surfaced_symbols_counts_distinct_symbols_in_entries() -> None:
    """With max_per_symbol=2, AAA contributes twice → surfaced still = 1."""
    view = compute_spotlight(
        {
            "AAA": _key_drivers(
                "AAA",
                _driver(label="A1", salience=0.9),
                _driver(label="A2", salience=0.8),
            ),
            "BBB": _key_drivers("BBB", _driver(salience=0.6)),
        },
        max_per_symbol=2,
    )
    # AAA contributes 2 entries + BBB 1 = 3 total, 2 distinct symbols.
    assert len(view.entries) == 3
    assert view.surfaced_symbols == 2


# ---------------------------------------------------------------------------
# Symbol case normalization
# ---------------------------------------------------------------------------


def test_lowercase_input_symbol_normalized_to_uppercase() -> None:
    view = compute_spotlight(
        {"aapl": _key_drivers("aapl", _driver(salience=0.7))}
    )
    assert view.entries[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# Quiet watchlist
# ---------------------------------------------------------------------------


def test_all_symbols_with_no_drivers_returns_empty_entries() -> None:
    view = compute_spotlight(
        {f"SYM{i}": _key_drivers(f"SYM{i}") for i in range(5)}
    )
    assert view.entries == ()
    assert view.considered_symbols == 5
    assert view.surfaced_symbols == 0


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------


def test_spotlight_entry_carries_full_driver_payload() -> None:
    """The driver attached to each entry is the original Driver object."""
    driver = _driver(
        kind="sector",
        label="Sector outlier · Trust score",
        tone="bull",
        salience=0.95,
        citations=("sector",),
    )
    view = compute_spotlight({"AAA": _key_drivers("AAA", driver)})
    entry: SpotlightEntry = view.entries[0]
    assert entry.driver.kind == "sector"
    assert entry.driver.label == "Sector outlier · Trust score"
    assert entry.driver.tone == "bull"
    assert entry.driver.salience == pytest.approx(0.95)
    assert entry.driver.citations == ("sector",)
