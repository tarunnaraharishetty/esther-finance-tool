"""Tests for the email-verification emailer.

Mirrors the structure of test_password_reset_emailer.py — same
contract (Log default, SMTP optional, swallow transport errors).
The duplication is deliberate: when the third transactional surface
lands we'll factor the shared core.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.api.email_verification_emailer import (
    LogEmailVerificationEmailer,
    SmtpEmailVerificationEmailer,
)


def test_log_emailer_returns_none_and_does_not_raise() -> None:
    """Default no-network transport — always succeeds. Same contract
    as the password-reset log emailer."""
    emailer = LogEmailVerificationEmailer()
    emailer.send("a@b.com", "https://example.com/verify-email?token=abc")


def _build_smtp_emailer() -> SmtpEmailVerificationEmailer:
    return SmtpEmailVerificationEmailer(
        host="smtp.example.com",
        port=587,
        username="user",
        password="pass",
        sender="noreply@example.com",
    )


def test_smtp_emailer_sends_starttls_authenticated_message(
    monkeypatch: Any,
) -> None:
    """ESMTP sequence + verification subject + link in body."""
    fake_smtp = MagicMock()
    fake_smtp.__enter__.return_value = fake_smtp
    fake_smtp.__exit__.return_value = False
    monkeypatch.setattr(
        "src.api.email_verification_emailer.smtplib.SMTP",
        MagicMock(return_value=fake_smtp),
    )

    _build_smtp_emailer().send(
        "a@b.com", "https://example.com/verify-email?token=xyz"
    )

    fake_smtp.starttls.assert_called_once()
    fake_smtp.login.assert_called_once_with("user", "pass")
    fake_smtp.send_message.assert_called_once()
    msg = fake_smtp.send_message.call_args[0][0]
    assert msg["To"] == "a@b.com"
    assert msg["Subject"].lower().startswith("verify")
    assert "verify-email?token=xyz" in msg.get_content()


def test_smtp_emailer_swallows_transport_errors(monkeypatch: Any) -> None:
    """Signup must not 5xx when SMTP is unreachable — the account
    was created, the user can re-request verification later."""
    import smtplib

    fake_smtp = MagicMock()
    fake_smtp.__enter__.return_value = fake_smtp
    fake_smtp.__exit__.return_value = False
    fake_smtp.send_message.side_effect = smtplib.SMTPException("nope")
    monkeypatch.setattr(
        "src.api.email_verification_emailer.smtplib.SMTP",
        MagicMock(return_value=fake_smtp),
    )
    # Must NOT raise.
    _build_smtp_emailer().send("a@b.com", "https://example.com/v")


def test_smtp_emailer_swallows_connection_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "src.api.email_verification_emailer.smtplib.SMTP",
        MagicMock(side_effect=OSError("connection refused")),
    )
    _build_smtp_emailer().send("a@b.com", "https://example.com/v")
