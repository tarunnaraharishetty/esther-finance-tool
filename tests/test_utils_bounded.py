"""Tests for the bounded-dict helpers used by the brief caches."""

from __future__ import annotations

import pytest

from src.utils.bounded import bounded_set, bounded_trim


def test_bounded_set_under_cap_keeps_all() -> None:
    d: dict[str, int] = {}
    for i in range(5):
        bounded_set(d, f"k{i}", i, max_entries=10)
    assert len(d) == 5
    assert d["k0"] == 0
    assert d["k4"] == 4


def test_bounded_set_evicts_oldest_when_over_cap() -> None:
    """FIFO eviction: the first key inserted is the first to leave."""
    d: dict[str, int] = {}
    for i in range(3):
        bounded_set(d, f"k{i}", i, max_entries=3)
    # One more — oldest (k0) should be evicted.
    bounded_set(d, "k3", 3, max_entries=3)
    assert "k0" not in d
    assert set(d.keys()) == {"k1", "k2", "k3"}


def test_reset_of_existing_key_moves_to_end() -> None:
    """A re-set should refresh the key's position so a hot key survives."""
    d: dict[str, int] = {}
    for i in range(3):
        bounded_set(d, f"k{i}", i, max_entries=3)
    # Refresh k0 — it should move to the end.
    bounded_set(d, "k0", 99, max_entries=3)
    # Add a new key — k1 should now be the oldest, not k0.
    bounded_set(d, "k_new", -1, max_entries=3)
    assert "k0" in d
    assert d["k0"] == 99
    assert "k1" not in d


def test_bounded_set_rejects_zero_cap() -> None:
    with pytest.raises(ValueError):
        bounded_set({}, "k", 1, max_entries=0)


def test_bounded_trim_drops_excess_in_one_pass() -> None:
    d = {f"k{i}": i for i in range(10)}
    dropped = bounded_trim(d, max_entries=4)
    assert dropped == 6
    assert len(d) == 4
    # FIFO: the latest 4 keys survive.
    assert set(d.keys()) == {"k6", "k7", "k8", "k9"}


def test_bounded_trim_under_cap_is_a_noop() -> None:
    d = {"a": 1, "b": 2}
    dropped = bounded_trim(d, max_entries=10)
    assert dropped == 0
    assert d == {"a": 1, "b": 2}
