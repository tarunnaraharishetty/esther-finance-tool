import { useCallback, useState } from "react";
import { CheckCircle2, Loader2, Mail } from "lucide-react";

import { type AccountError, requestVerifyEmail } from "@/lib/account";

interface Props {
  /** Caller-supplied address — surfaced in the banner copy so the
   *  user can tell the system who they're being asked to verify as. */
  email: string;
  /** Optional override of the click handler — used by tests. The
   *  default calls ``requestVerifyEmail`` from ``@/lib/account``. */
  onResend?: () => Promise<void>;
}

/**
 * Persistent banner shown in the app shell when ``user.email_verified``
 * is False. Renders a "Resend verification" button that fires a fresh
 * mint+dispatch on the server. After a successful resend the button
 * collapses to a success message so a flood-clicker doesn't spam the
 * email queue.
 *
 * The server is the source of truth for verified state; this banner
 * disappears when the App-level session reconciliation observes
 * ``email_verified=true`` (either after the user redeems a token or
 * after they click the verification link in another tab). The banner
 * does not refresh /me on resend success — the verify request only
 * mints a new token, doesn't flip the bit.
 */
export function VerifyEmailBanner({ email, onResend }: Props) {
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClick = useCallback(async (): Promise<void> => {
    if (busy || sent) return;
    setBusy(true);
    setError(null);
    try {
      if (onResend) {
        await onResend();
      } else {
        await requestVerifyEmail();
      }
      setSent(true);
    } catch (e) {
      const err = e as AccountError;
      setError(
        err && typeof err === "object" && "message" in err
          ? err.message
          : e instanceof Error
            ? e.message
            : "Could not resend",
      );
    } finally {
      setBusy(false);
    }
  }, [busy, onResend, sent]);

  return (
    <div
      role="status"
      data-testid="verify-email-banner"
      className="mb-4 flex items-start gap-3 rounded-lg border border-warn/30 bg-warn/10 px-4 py-3 text-sm text-warn animate-fade-in"
    >
      <Mail className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="font-medium">Verify your email</div>
        <div className="text-xs text-warn/80">
          We sent a verification link to <span className="font-mono">{email}</span>.
          Open it to confirm this is you.
        </div>
        {error && (
          <div data-testid="verify-banner-error" className="mt-1 text-xs text-bear">
            {error}
          </div>
        )}
      </div>
      {sent ? (
        <span
          data-testid="verify-banner-sent"
          className="inline-flex items-center gap-1 rounded-md border border-bull/30 bg-bull/10 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-bull"
        >
          <CheckCircle2 className="h-3 w-3" />
          Sent
        </span>
      ) : (
        <button
          type="button"
          data-testid="verify-banner-resend"
          onClick={() => void handleClick()}
          disabled={busy}
          className="inline-flex items-center gap-1 rounded-md border border-warn/40 bg-warn/20 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-warn hover:bg-warn/30 transition-colors disabled:opacity-50"
        >
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}
          Resend
        </button>
      )}
    </div>
  );
}
