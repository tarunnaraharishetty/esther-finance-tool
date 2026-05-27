import { useState, type FormEvent } from "react";
import { AlertOctagon, ArrowLeft, CheckCircle2, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { AuthError, AuthUser } from "@/lib/auth";
import {
  type PasswordResetError,
  requestPasswordReset,
} from "@/lib/passwordReset";
import { cn } from "@/lib/utils";

type Mode = "login" | "signup" | "forgot";

interface Props {
  /** Imperative login handler from the session hook. */
  onLogin: (email: string, password: string) => Promise<AuthUser>;
  /** Imperative signup handler from the session hook. */
  onSignup: (email: string, password: string) => Promise<AuthUser>;
  /** Optional pre-set mode (used by the post-reset redirect to land
   *  back on the login form with a flash banner). */
  initialMode?: Mode;
  /** Optional banner shown above the form. Used by ResetPasswordPage
   *  to surface "Password updated — please log in." after redirect. */
  flashMessage?: string | null;
}

/**
 * Login + signup + forgot-password screen. Single-column, centered,
 * no nav chrome — the unauthenticated state should feel like a
 * focused doorway, not a stripped version of the app.
 *
 * Mode toggle (login / signup / forgot) keeps the same form fields
 * across the first two; the forgot mode collapses to an email-only
 * form and a uniform "if registered" success message — never
 * leaking whether the email exists.
 */
export function LoginPage({ onLogin, onSignup, initialMode, flashMessage }: Props) {
  const [mode, setMode] = useState<Mode>(initialMode ?? "login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<AuthError | PasswordResetError | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [forgotMessage, setForgotMessage] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    if (submitting) return;
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "signup") {
        await onSignup(email.trim(), password);
      } else if (mode === "login") {
        await onLogin(email.trim(), password);
      } else {
        const res = await requestPasswordReset(email.trim());
        setForgotMessage(res.message);
      }
    } catch (e) {
      const err = e as AuthError | PasswordResetError;
      setError(
        err && typeof err === "object" && "status" in err
          ? err
          : {
              status: 0,
              message: e instanceof Error ? e.message : "Request failed",
            },
      );
    } finally {
      setSubmitting(false);
    }
  };

  const switchMode = (next: Mode): void => {
    if (next === mode) return;
    setMode(next);
    setError(null);
    setForgotMessage(null);
  };

  return (
    <div className="grid min-h-screen place-items-center bg-gradient-to-br from-background via-background to-card p-4">
      <div
        data-testid="login-page"
        data-mode={mode}
        className="w-full max-w-sm"
      >
        <header className="mb-6 text-center">
          <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-primary to-accent shadow-glow">
            <Sparkles className="h-5 w-5 text-primary-foreground" />
          </div>
          <h1 className="mt-3 font-display text-2xl font-semibold tracking-tightest">
            Esther
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            AI-powered market intelligence for discretionary traders.
          </p>
        </header>

        <div className="surface-premium p-5">
          {mode !== "forgot" && (
            <div className="mb-4 grid grid-cols-2 gap-1 rounded-md border border-border/40 bg-card/40 p-1">
              <button
                type="button"
                data-testid="mode-login"
                onClick={() => switchMode("login")}
                className={cn(
                  "rounded px-2 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors",
                  mode === "login"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                Log in
              </button>
              <button
                type="button"
                data-testid="mode-signup"
                onClick={() => switchMode("signup")}
                className={cn(
                  "rounded px-2 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors",
                  mode === "signup"
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                Sign up
              </button>
            </div>
          )}

          {mode === "forgot" && (
            <button
              type="button"
              data-testid="back-to-login"
              onClick={() => switchMode("login")}
              className="mb-3 inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="h-3 w-3" />
              Back to log in
            </button>
          )}

          {flashMessage && (
            <div
              role="status"
              data-testid="login-flash"
              className="mb-3 flex items-start gap-2 rounded-md border border-bull/40 bg-bull/10 px-3 py-2 text-[12px] text-bull"
            >
              <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{flashMessage}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-3">
            <label className="block space-y-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Email
              </span>
              <Input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
                autoComplete="email"
                data-testid="email-input"
                disabled={submitting}
              />
            </label>
            {mode !== "forgot" && (
              <label className="block space-y-1">
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Password
                </span>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={
                    mode === "signup" ? "min 8 characters" : "your password"
                  }
                  required
                  minLength={mode === "signup" ? 8 : 1}
                  autoComplete={
                    mode === "signup" ? "new-password" : "current-password"
                  }
                  data-testid="password-input"
                  disabled={submitting}
                />
                {mode === "signup" && (
                  <span className="block font-mono text-[10px] text-muted-foreground/70">
                    Minimum 8 characters. We hash with bcrypt; the plaintext
                    never persists.
                  </span>
                )}
              </label>
            )}

            {error && (
              <div
                role="alert"
                data-testid="login-error"
                className="flex items-start gap-2 rounded-md border border-bear/40 bg-bear/10 px-3 py-2 text-[12px] text-bear"
              >
                <AlertOctagon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>{error.message}</span>
              </div>
            )}

            {forgotMessage && (
              <div
                role="status"
                data-testid="forgot-success"
                className="flex items-start gap-2 rounded-md border border-bull/40 bg-bull/10 px-3 py-2 text-[12px] text-bull"
              >
                <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>{forgotMessage}</span>
              </div>
            )}

            <Button
              type="submit"
              disabled={submitting}
              data-testid="submit-button"
              className="w-full"
            >
              {submitting && <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />}
              {mode === "signup"
                ? "Create account"
                : mode === "forgot"
                  ? "Send reset link"
                  : "Log in"}
            </Button>
          </form>

          {mode === "login" && (
            <button
              type="button"
              data-testid="forgot-link"
              onClick={() => switchMode("forgot")}
              className="mt-3 block w-full text-center font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
            >
              Forgot password?
            </button>
          )}
        </div>

        <p className="mt-4 text-center font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
          paper-feed only · decision-support · no order submission
        </p>
      </div>
    </div>
  );
}
