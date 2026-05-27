"""Tests for the password-reset emailer Protocol + impls.

Two implementations live behind one Protocol so the auth route
doesn't bind to any single delivery channel:

* :class:`LogPasswordResetEmailer` — the no-network default
  (closed-beta operator relays the link manually).
* :class:`SmtpPasswordResetEmailer` — stdlib smtplib transport
  for real providers.

These tests stub smtplib at the module boundary; no network calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.api.password_reset_emailer import (
    LogPasswordResetEmailer,
    SmtpPasswordResetEmailer,
)

# ---------------------------------------------------------------------------
# LogPasswordResetEmailer
# ---------------------------------------------------------------------------


def test_log_emailer_returns_none_and_does_not_raise() -> None:
    """No-op transport — always succeeds. The route relies on this
    to keep the response identical whether email delivery is
    configured or not, so an enumeration probe via SMTP 5xx can't
    leak via the password-reset surface."""
    emailer = LogPasswordResetEmailer()
    # No assertion on the log itself — the structured-log emission
    # is verified by the request-log middleware tests. Behavioral
    # contract here is "doesn't raise."
    emailer.send("a@b.com", "https://example.com/reset?token=abc")


# ---------------------------------------------------------------------------
# SmtpPasswordResetEmailer
# ---------------------------------------------------------------------------


def _build_smtp_emailer() -> SmtpPasswordResetEmailer:
    return SmtpPasswordResetEmailer(
        host="smtp.example.com",
        port=587,
        username="user",
        password="pass",
        sender="noreply@example.com",
    )


def test_smtp_emailer_sends_starttls_authenticated_message(
    monkeypatch: Any,
) -> None:
    """Happy path: connect → ehlo → starttls → ehlo → login → send.
    Mocking the smtplib.SMTP context manager lets us assert the
    exact ESMTP sequence without a network."""
    fake_smtp = MagicMock()
    fake_smtp.__enter__.return_value = fake_smtp
    fake_smtp.__exit__.return_value = False
    smtp_factory = MagicMock(return_value=fake_smtp)
    monkeypatch.setattr(
        "src.api.password_reset_emailer.smtplib.SMTP", smtp_factory
    )

    _build_smtp_emailer().send("a@b.com", "https://example.com/r?token=xyz")

    smtp_factory.assert_called_once_with(
        "smtp.example.com", 587, timeout=15.0
    )
    fake_smtp.starttls.assert_called_once()
    fake_smtp.login.assert_called_once_with("user", "pass")
    fake_smtp.send_message.assert_called_once()
    msg = fake_smtp.send_message.call_args[0][0]
    assert msg["To"] == "a@b.com"
    assert msg["From"] == "noreply@example.com"
    assert "https://example.com/r?token=xyz" in msg.get_content()


def test_smtp_emailer_skips_login_when_credentials_blank(
    monkeypatch: Any,
) -> None:
    """Some self-hosted relays don't require auth; ``login`` raises
    SMTPException against them. We skip auth when either credential
    is empty so an unauthenticated relay works."""
    fake_smtp = MagicMock()
    fake_smtp.__enter__.return_value = fake_smtp
    fake_smtp.__exit__.return_value = False
    monkeypatch.setattr(
        "src.api.password_reset_emailer.smtplib.SMTP",
        MagicMock(return_value=fake_smtp),
    )

    emailer = SmtpPasswordResetEmailer(
        host="relay.example.com",
        port=25,
        username="",
        password="",
        sender="noreply@example.com",
    )
    emailer.send("a@b.com", "https://example.com/r")

    fake_smtp.login.assert_not_called()
    fake_smtp.send_message.assert_called_once()


def test_smtp_emailer_swallows_smtp_exception(monkeypatch: Any) -> None:
    """A 5xx on send must never escape into the route — the response
    shape (always 200) would diverge from the no-such-email case and
    become an enumeration side-channel. The emailer logs + returns
    instead of raising."""
    import smtplib

    fake_smtp = MagicMock()
    fake_smtp.__enter__.return_value = fake_smtp
    fake_smtp.__exit__.return_value = False
    fake_smtp.send_message.side_effect = smtplib.SMTPException("nope")
    monkeypatch.setattr(
        "src.api.password_reset_emailer.smtplib.SMTP",
        MagicMock(return_value=fake_smtp),
    )

    # Must NOT raise.
    _build_smtp_emailer().send("a@b.com", "https://example.com/r")


def test_smtp_emailer_swallows_connection_error(monkeypatch: Any) -> None:
    """Same shape as above for the more common case: server
    unreachable. ``OSError`` (subsumes ``ConnectionRefusedError``,
    ``socket.timeout``, etc.) must not propagate."""
    smtp_factory = MagicMock(side_effect=OSError("connection refused"))
    monkeypatch.setattr(
        "src.api.password_reset_emailer.smtplib.SMTP", smtp_factory
    )
    _build_smtp_emailer().send("a@b.com", "https://example.com/r")
