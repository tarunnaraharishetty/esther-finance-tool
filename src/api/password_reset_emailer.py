"""Password-reset email transport.

A tiny Protocol + two implementations so the auth route doesn't bind
to any one delivery channel:

* :class:`LogPasswordResetEmailer` (default) — emits the reset link
  to the structured log at INFO. Lets an operator running a closed
  beta manually relay the link to the user; zero external
  dependencies; works on any deploy without SMTP credentials.
* :class:`SmtpPasswordResetEmailer` — stdlib ``smtplib`` over STARTTLS.
  Lets the operator wire SendGrid / SES / Mailgun / Postmark / any
  SMTP-speaking provider without adding a Python dependency.

The auth route depends on the Protocol, not the impls. ``app.py``
picks one based on settings: if ``SMTP_HOST`` is set we build the
SMTP transport, otherwise we fall through to the log emailer.

Why Protocol, not ABC
---------------------
Structural typing keeps the test fakes one-liners (any object with
``send(email, reset_url) -> None`` satisfies it) and lines up with
how the rest of the codebase (Summarizer, ThesisGenerator) does
pluggable backends.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Protocol

from src.utils.logging import get_logger

log = get_logger(__name__)


class PasswordResetEmailer(Protocol):
    """Transport contract for a single reset-link delivery.

    Implementations may block on the network — the route invokes
    this inside a ``run_in_threadpool`` so the request loop stays
    responsive. Implementations should NOT raise on transient
    failure; log + return so the caller doesn't surface the failure
    to the requester (which would itself become an enumeration
    side-channel).
    """

    def send(self, email: str, reset_url: str) -> None:
        """Deliver the reset link to ``email``."""
        ...


class LogPasswordResetEmailer:
    """Logs the reset URL at INFO. Default when no SMTP is configured.

    Use case: friends-and-family closed beta where the operator is
    on the same channel as the user and can paste the link manually.
    The link goes to ``stderr`` via ``structlog`` so it shows up in
    the Render / Railway / docker logs alongside every other request.
    """

    def send(self, email: str, reset_url: str) -> None:
        log.info(
            "auth.password_reset.link",
            email=email,
            reset_url=reset_url,
            transport="log",
        )


class SmtpPasswordResetEmailer:
    """Stdlib SMTP over STARTTLS. Suitable for any provider whose
    submission server speaks ESMTP (Gmail, SES, Mailgun, SendGrid,
    Postmark, Fastmail, self-hosted Postfix...).

    Args:
        host: SMTP server hostname.
        port: SMTP server port (typically 587 for STARTTLS submission).
        username: SMTP auth username, or empty to skip auth.
        password: SMTP auth password / API token. Empty skips auth.
        sender: ``From`` header. Many providers reject mail whose
            sender doesn't match the authenticated identity.
        timeout_seconds: Connection + read timeout. Default 15s —
            email delivery from a web request must not stall.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        sender: str,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._sender = sender
        self._timeout = timeout_seconds

    def send(self, email: str, reset_url: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = "Reset your Esther password"
        msg["From"] = self._sender
        msg["To"] = email
        msg.set_content(
            "Someone (hopefully you) requested a password reset for "
            "your Esther account.\n\n"
            f"To choose a new password, open this link within the next hour:\n\n"
            f"  {reset_url}\n\n"
            "If you didn't request this, you can ignore this email — "
            "your password will not change.\n"
        )
        try:
            with smtplib.SMTP(
                self._host, self._port, timeout=self._timeout
            ) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                if self._username and self._password:
                    smtp.login(self._username, self._password)
                smtp.send_message(msg)
        except (OSError, smtplib.SMTPException) as exc:
            # Never raise into the route — that would let an attacker
            # probe SMTP health via password-reset 5xx responses, and
            # the response shape (always 200) would diverge from the
            # no-such-email case. Log + swallow.
            log.warning(
                "auth.password_reset.smtp_failed",
                email=email,
                error=str(exc),
                transport="smtp",
            )
            return
        log.info(
            "auth.password_reset.sent",
            email=email,
            transport="smtp",
        )


__all__ = [
    "LogPasswordResetEmailer",
    "PasswordResetEmailer",
    "SmtpPasswordResetEmailer",
]
