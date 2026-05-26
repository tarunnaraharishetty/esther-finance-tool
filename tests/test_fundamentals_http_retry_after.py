"""Tests for the ``Retry-After`` parser in ``src.intelligence.fundamentals.http``.

Pinned to RFC 6585 / RFC 7231 — both delta-seconds and HTTP-date forms.
The retry queue depends on this returning honest seconds; a hostile or
buggy upstream must not be able to strand a symbol for weeks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.intelligence.fundamentals.http import _parse_retry_after


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("0", 0.0),
        ("30", 30.0),
        ("300", 300.0),
        ("-1", None),
        ("not-a-number-or-date", None),
    ],
)
def test_delta_seconds_form(value: str | None, expected: float | None) -> None:
    assert _parse_retry_after(value) == expected


def test_http_date_form_returns_seconds_until_deadline() -> None:
    """RFC 7231 absolute date form: "Wed, 21 Oct 2026 07:28:00 GMT"."""
    future = datetime.now(UTC) + timedelta(seconds=120)
    # Format the deadline as an HTTP-date — strftime works because the
    # value is timezone-aware and email.utils.parsedate_to_datetime
    # round-trips this shape cleanly.
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = _parse_retry_after(http_date)
    assert result is not None
    # 5 second tolerance for the test taking a moment to execute.
    assert 110 <= result <= 130


def test_http_date_in_the_past_returns_none() -> None:
    past = datetime.now(UTC) - timedelta(hours=1)
    http_date = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert _parse_retry_after(http_date) is None


def test_absurdly_large_value_capped_at_24h() -> None:
    """A hostile upstream cannot strand a symbol for weeks."""
    seconds_in_a_week = 7 * 24 * 3600
    assert _parse_retry_after(str(seconds_in_a_week)) == 24 * 3600.0
