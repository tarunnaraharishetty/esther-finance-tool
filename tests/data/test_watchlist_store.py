"""Tests for :class:`WatchlistStore`.

Covers the lazy-init contract, symbol normalization (upper-case +
trim), dedupe (UNIQUE pair index), sort_order monotonicity, the
remove-returns-bool contract, and the seed_default idempotency that
the signup hook relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.user_store import UserStore
from src.data.watchlist_store import (
    SymbolAlreadyWatched,
    WatchlistStore,
)


# ---------------------------------------------------------------------------
# Construction + lazy init
# ---------------------------------------------------------------------------


def test_construction_does_not_touch_disk(tmp_path: Path) -> None:
    """Same invariant as HealthStore + UserStore — construct is free."""
    store = WatchlistStore(tmp_path / "missing" / "users.db")
    assert not (tmp_path / "missing").exists()
    store.close()


def test_list_for_unknown_user_returns_empty(tmp_path: Path) -> None:
    """First call materializes the schema and returns the empty list."""
    store = WatchlistStore(tmp_path / "users.db")
    assert store.list_for(9999) == []
    store.close()


# ---------------------------------------------------------------------------
# Add / list round-trip
# ---------------------------------------------------------------------------


def test_add_and_list_round_trip(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    entry = store.add(42, "NVDA")
    assert entry.symbol == "NVDA"
    assert entry.sort_order == 0
    out = store.list_for(42)
    assert [e.symbol for e in out] == ["NVDA"]
    store.close()


def test_add_normalizes_symbol_to_upper(tmp_path: Path) -> None:
    """  nvda  → NVDA. Tickers are uppercase in our universe."""
    store = WatchlistStore(tmp_path / "users.db")
    entry = store.add(1, "  nvda  ")
    assert entry.symbol == "NVDA"
    assert store.contains(1, "NvDa") is True
    store.close()


def test_add_empty_symbol_raises(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    with pytest.raises(ValueError):
        store.add(1, "   ")
    store.close()


def test_add_duplicate_raises_symbol_already_watched(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    store.add(1, "AAPL")
    with pytest.raises(SymbolAlreadyWatched):
        store.add(1, "aapl")  # case-insensitive — same row
    store.close()


def test_add_same_symbol_different_users_is_allowed(tmp_path: Path) -> None:
    """The UNIQUE constraint is on (user_id, symbol), not symbol alone."""
    store = WatchlistStore(tmp_path / "users.db")
    store.add(1, "MSFT")
    store.add(2, "MSFT")
    assert [e.symbol for e in store.list_for(1)] == ["MSFT"]
    assert [e.symbol for e in store.list_for(2)] == ["MSFT"]
    store.close()


def test_add_preserves_insertion_order(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    for sym in ["NVDA", "AAPL", "TSLA"]:
        store.add(1, sym)
    out = store.list_for(1)
    assert [e.symbol for e in out] == ["NVDA", "AAPL", "TSLA"]
    # sort_order is monotonic 0..n-1.
    assert [e.sort_order for e in out] == [0, 1, 2]
    store.close()


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def test_remove_present_symbol_returns_true(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    store.add(1, "NVDA")
    assert store.remove(1, "NVDA") is True
    assert store.list_for(1) == []
    store.close()


def test_remove_missing_symbol_returns_false(tmp_path: Path) -> None:
    """Idempotent — removing a symbol that isn't there is a no-op."""
    store = WatchlistStore(tmp_path / "users.db")
    assert store.remove(1, "NVDA") is False
    store.close()


def test_remove_normalizes_symbol(tmp_path: Path) -> None:
    """Lowercase / whitespace input still resolves to the stored row."""
    store = WatchlistStore(tmp_path / "users.db")
    store.add(1, "NVDA")
    assert store.remove(1, "  nvda  ") is True
    store.close()


def test_remove_empty_symbol_returns_false(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    assert store.remove(1, "") is False
    assert store.remove(1, "   ") is False
    store.close()


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


def test_contains_returns_true_after_add(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    store.add(1, "AAPL")
    assert store.contains(1, "AAPL") is True
    assert store.contains(1, "MSFT") is False
    store.close()


def test_contains_empty_symbol_returns_false(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    assert store.contains(1, "") is False
    store.close()


# ---------------------------------------------------------------------------
# seed_default
# ---------------------------------------------------------------------------


def test_seed_default_inserts_in_order(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    seeded = store.seed_default(1, ["AAPL", "MSFT", "NVDA"])
    assert [e.symbol for e in seeded] == ["AAPL", "MSFT", "NVDA"]
    out = store.list_for(1)
    assert [e.symbol for e in out] == ["AAPL", "MSFT", "NVDA"]
    assert [e.sort_order for e in out] == [0, 1, 2]
    store.close()


def test_seed_default_skips_duplicates(tmp_path: Path) -> None:
    """If a symbol is already present, seed_default skips it silently."""
    store = WatchlistStore(tmp_path / "users.db")
    store.add(1, "AAPL")
    seeded = store.seed_default(1, ["AAPL", "MSFT"])
    # Only MSFT was newly inserted.
    assert [e.symbol for e in seeded] == ["MSFT"]
    assert [e.symbol for e in store.list_for(1)] == ["AAPL", "MSFT"]
    store.close()


def test_seed_default_skips_blank_entries(tmp_path: Path) -> None:
    """Empty / whitespace entries are filtered before insert."""
    store = WatchlistStore(tmp_path / "users.db")
    seeded = store.seed_default(1, ["AAPL", "", "   ", "MSFT"])
    assert [e.symbol for e in seeded] == ["AAPL", "MSFT"]
    store.close()


def test_seed_default_empty_iterable_is_noop(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    seeded = store.seed_default(1, [])
    assert seeded == []
    assert store.list_for(1) == []
    store.close()


# ---------------------------------------------------------------------------
# Foreign-key cascade with the user store
# ---------------------------------------------------------------------------


def test_shared_db_file_with_user_store_works(tmp_path: Path) -> None:
    """UserStore + WatchlistStore can share ``users.db`` peacefully.

    Both materialize their tables independently on first use (the
    schemas don't overlap). The future delete-account flow will
    explicitly remove watchlist rows before deleting the user — we
    deliberately don't rely on SQLite FK cascade for that, since the
    cross-table integrity would require a deterministic startup
    order between the two stores.
    """
    db = tmp_path / "users.db"
    users = UserStore(db)
    wl = WatchlistStore(db)
    user = users.create_user("a@b.com", "longenough")
    wl.add(user.id, "AAPL")
    wl.add(user.id, "MSFT")
    assert [e.symbol for e in wl.list_for(user.id)] == ["AAPL", "MSFT"]
    # Both stores keep working after the cross-store writes.
    assert users.get_user_by_id(user.id) is not None
    wl.close()
    users.close()


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


def test_entry_to_wire_has_expected_keys(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "users.db")
    entry = store.add(1, "AAPL")
    wire = entry.to_wire()
    assert set(wire.keys()) == {"symbol", "added_at", "sort_order"}
    assert wire["symbol"] == "AAPL"
    store.close()
