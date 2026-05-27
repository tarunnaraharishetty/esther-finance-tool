"""Email-verification message transport.

Mirror of :mod:`src.api.password_reset_emailer` — same Protocol shape,
same two implementations (Log default + SMTP), different message
body. Kept as a separate module rather than a shared base class
because the two transactional surfaces don't justify abstraction at
two members.

The SMTP-send mechanics duplicate ~20 lines from password_reset_emailer;
that's a deliberate trade — DRY here would force a parameterized
message-builder Protocol whose only payoff today is the duplication
saved. When the third transactional email lands (e.g. login-from-new-
device, billing) we'll factor the shared core.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Protocol

from src.utils.logging import get_logger

log = get_logger(__name__)


class EmailVerificationEmailer(Protocol):
    """Transport contract for a single verification-link delivery.

    Implementations may block on the network — the route invokes
    this inside a ``run_in_threadpool`` so the request loop stays
    responsive. Implementations should NOT raise on transient
    failure; log + return so the caller doesn't surface the failure
    to the user (which would be confusing — they signed up
    successfully and the verification email just didn't arrive).
    """

    def send(self, email: str, verify_url: str) -> None:
        """Deliver the verification link to ``email``."""
        ...


class LogEmailVerificationEmailer:
    """Logs the verification URL at INFO. Default when no SMTP is
    configured. Same use case as :class:`LogPasswordResetEmailer` —
    closed-beta operator relays the link manually."""

    def send(self, email: str, verify_url: str) -> None:
        log.info(
            "auth.email_verification.link",
            email=email,
            verify_url=verify_url,
            transport="log",
        )


class SmtpEmailVerificationEmailer:
    """Stdlib SMTP over STARTTLS. Same parameters as the password-reset
    transport so the wiring in app.py can reuse the
    ``Settings.smtp_*`` quintuple."""

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

    def send(self, email: str, verify_url: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = "Verify your Esther email"
        msg["From"] = self._sender
        msg["To"] = email
        msg.set_content(
            "Welcome to Esther! To finish setting up your account, "
            "verify your email address by opening this link within the "
            "next 24 hours:\n\n"
            f"  {verify_url}\n\n"
            "If you didn't create an Esther account, you can ignore "
            "this email.\n"
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
            # Same swallow-and-log shape as the password-reset emailer.
            # An SMTP outage during signup must NOT 5xx the signup
            # response — the account was created successfully and the
            # user can re-request the verification email later.
            log.warning(
                "auth.email_verification.smtp_failed",
                email=email,
                error=str(exc),
                transport="smtp",
            )
            return
        log.info(
            "auth.email_verification.sent",
            email=email,
            transport="smtp",
        )


__all__ = [
    "EmailVerificationEmailer",
    "LogEmailVerificationEmailer",
    "SmtpEmailVerificationEmailer",
]
