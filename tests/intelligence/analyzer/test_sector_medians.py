"""Sector-median table lookup tests."""

from __future__ import annotations

import pytest

from src.intelligence.analyzer import sector_medians


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    sector_medians._clear_cache()


def test_known_sector_returns_seed_values() -> None:
    tech = sector_medians.lookup("Information Technology")
    assert tech.pe == 30.0
    assert tech.ev_ebitda == 20.0
    assert tech.ps == 6.0


def test_unknown_sector_falls_back_to_default() -> None:
    fallback = sector_medians.lookup("Unicorn Speculation")
    default = sector_medians.lookup("_default")
    assert fallback == default
    assert default.pe == 20.0


def test_none_or_empty_sector_returns_default() -> None:
    default = sector_medians.lookup("_default")
    assert sector_medians.lookup(None) == default
    assert sector_medians.lookup("") == default
    assert sector_medians.lookup("   ") == default


def test_alias_maps_to_canonical_sector() -> None:
    via_alias = sector_medians.lookup("Tech")
    canonical = sector_medians.lookup("Information Technology")
    assert via_alias == canonical

    via_alias = sector_medians.lookup("Healthcare")
    canonical = sector_medians.lookup("Health Care")
    assert via_alias == canonical


def test_lookup_is_case_insensitive() -> None:
    assert sector_medians.lookup("FINANCIALS") == sector_medians.lookup("Financials")


def test_missing_table_falls_back_gracefully(tmp_path, monkeypatch) -> None:
    """If the TOML file is missing, the analyzer still gets a default block."""
    missing = tmp_path / "does-not-exist.toml"
    monkeypatch.setattr(
        "src.intelligence.analyzer.sector_medians.get_settings",
        lambda: _SettingsStub(missing),
    )
    sector_medians._clear_cache()
    result = sector_medians.lookup("anything")
    assert result.pe == 20.0  # default


def test_custom_table_loads_from_settings(tmp_path, monkeypatch) -> None:
    """Pointing the settings path at a custom TOML uses those values."""
    custom = tmp_path / "custom.toml"
    custom.write_text(
        '[_default]\npe = 99.0\nev_ebitda = 88.0\nps = 7.7\npeg = 5.5\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.intelligence.analyzer.sector_medians.get_settings",
        lambda: _SettingsStub(custom),
    )
    sector_medians._clear_cache()
    result = sector_medians.lookup("anything")
    assert result.pe == 99.0
    assert result.ev_ebitda == 88.0


class _SettingsStub:
    def __init__(self, path) -> None:
        self.sector_medians_path = path
