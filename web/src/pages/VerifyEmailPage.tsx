import { useEffect, useRef, useState } from "react";
import { AlertOctagon, CheckCircle2, Loader2, Sparkles } from "lucide-react";

import { type AccountError, confirmVerifyEmail } from "@/lib/account";

interface Props {
  /** Token extracted from the ``?token=…`` query string. */
  token: string;
  /** Invoked after a successful redemption so App.tsx can clear the
   *  token from the URL and either bounce to the dashboard (when the
   *  user is logged in) or to the LoginPage (when they aren't). */
  onSuccess: () => void;
}

/**
 * Token-redemption screen for email verification. Reached via the
 * emailed link:
 *
 *   GET https://app.example.com/verify-email?token=XXX
 *
 * Unlike the password-reset flow, verification is one-click — the
 * page POSTs the token immediately on mount, then renders a result.
 * No form fields, no confirmation. The server's confirm route is
 * idempotent on already-verified users so a double-load from
 * back-navigation shows the success screen, not an error.
 */
export function VerifyEmailPage({ token, onSuccess }: Props) {
  const [state, setState] = useState<"verifying" | "ok" | "error">(
    "verifying",
  );
  const [error, setError] = useState<string | null>(null);
  // StrictMode mounts components twice in dev — guard so we don't
  // POST the token twice on the first render. (The route is idempotent
  // so this is defense in depth, not correctness.)
  const fired = useRef(false);

  useEffect(() => {
    if (fired.current) return;
    fired.current = true;
    (async () => {
      try {
        await confirmVerifyEmail(token);
        setState("ok");
      } catch (e) {
        const err = e as AccountError;
        setState("error");
        setError(
          err && typeof err === "object" && "message" in err
            ? err.message
            : e instanceof Error
              ? e.message
              : "Verification failed",
        );
      }
    })();
  }, [token]);

  return (
    <div className="grid min-h-screen place-items-center bg-gradient-to-br from-background via-background to-card p-4">
      <div data-testid="verify-page" data-state={state} className="w-full max-w-sm text-center">
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-primary to-accent shadow-glow">
          <Sparkles className="h-5 w-5 text-primary-foreground" />
        </div>

        {state === "verifying" && (
          <>
            <h1 className="mt-3 font-display text-2xl font-semibold tracking-tightest">
              Verifying your email…
            </h1>
            <p className="mt-2 inline-flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Hang on for a moment.
            </p>
          </>
        )}

        {state === "ok" && (
          <>
            <h1 className="mt-3 font-display text-2xl font-semibold tracking-tightest">
              Email verified
            </h1>
            <p className="mt-2 inline-flex items-center gap-2 text-sm text-bull">
              <CheckCircle2 className="h-4 w-4" />
              All set.
            </p>
            <button
              type="button"
              data-testid="verify-continue"
              onClick={onSuccess}
              className="mt-5 inline-flex w-full items-center justify-center rounded-md bg-primary px-3 py-2 font-mono text-[11px] uppercase tracking-wider text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Continue
            </button>
          </>
        )}

        {state === "error" && (
          <>
            <h1 className="mt-3 font-display text-2xl font-semibold tracking-tightest">
              Couldn't verify
            </h1>
            <p
              data-testid="verify-error"
              role="alert"
              className="mt-2 inline-flex items-center gap-2 text-sm text-bear"
            >
              <AlertOctagon className="h-4 w-4" />
              {error}
            </p>
            <button
              type="button"
              data-testid="verify-continue-error"
              onClick={onSuccess}
              className="mt-5 inline-flex w-full items-center justify-center rounded-md border border-border/50 bg-card/40 px-3 py-2 font-mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
            >
              Back to log in
            </button>
          </>
        )}
      </div>
    </div>
  );
}
