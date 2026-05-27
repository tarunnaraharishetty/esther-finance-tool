import { useState, type FormEvent } from "react";
import {
  AlertOctagon,
  CheckCircle2,
  Loader2,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  type PasswordResetError,
  confirmPasswordReset,
} from "@/lib/passwordReset";

interface Props {
  /** Reset token extracted from the ``?token=…`` query string. */
  token: string;
  /** Invoked on successful reset so App.tsx can flip back to the
   *  LoginPage with a flash banner. */
  onSuccess: () => void;
}

/**
 * Token-redemption screen. Reached via the emailed reset URL:
 *
 *   GET https://app.example.com/reset-password?token=XXX
 *
 * App.tsx detects the ``token`` query string at mount and routes
 * here instead of the LoginPage. We never accept the token through
 * any other path — it's a one-shot credential whose only valid
 * lifecycle is "arrived via email → typed new password → consumed."
 *
 * Server-side, ``consume_password_reset_token`` marks the token used
 * before this page learns the outcome — so a doubled submit (user
 * double-clicks the button) results in the second call getting 400,
 * which we surface in the same error region as a bad token. The
 * submit button disables on the first click to make this rare.
 */
export function ResetPasswordPage({ token, onSuccess }: Props) {
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<PasswordResetError | null>(null);
  const [mismatch, setMismatch] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    if (submitting) return;
    setError(null);
    setMismatch(false);
    if (password !== confirm) {
      setMismatch(true);
      return;
    }
    setSubmitting(true);
    try {
      await confirmPasswordReset(token, password);
      // Server already revoked every session — bounce back to the
      // login screen with a flash so the user knows what happened.
      onSuccess();
    } catch (e) {
      const err = e as PasswordResetError;
      setError(
        err && typeof err === "object" && "status" in err
          ? err
          : {
              status: 0,
              message: e instanceof Error ? e.message : "Reset failed",
            },
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="grid min-h-screen place-items-center bg-gradient-to-br from-background via-background to-card p-4">
      <div data-testid="reset-page" className="w-full max-w-sm">
        <header className="mb-6 text-center">
          <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-primary to-accent shadow-glow">
            <Sparkles className="h-5 w-5 text-primary-foreground" />
          </div>
          <h1 className="mt-3 font-display text-2xl font-semibold tracking-tightest">
            Set a new password
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Choose a new password for your Esther account.
          </p>
        </header>

        <div className="surface-premium p-5">
          <form onSubmit={handleSubmit} className="space-y-3">
            <label className="block space-y-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                New password
              </span>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="min 8 characters"
                required
                minLength={8}
                autoComplete="new-password"
                data-testid="new-password-input"
                disabled={submitting}
              />
            </label>
            <label className="block space-y-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Confirm password
              </span>
              <Input
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder="repeat the new password"
                required
                minLength={8}
                autoComplete="new-password"
                data-testid="confirm-password-input"
                disabled={submitting}
              />
            </label>

            {mismatch && (
              <div
                role="alert"
                data-testid="mismatch-error"
                className="flex items-start gap-2 rounded-md border border-bear/40 bg-bear/10 px-3 py-2 text-[12px] text-bear"
              >
                <AlertOctagon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>The two passwords don't match.</span>
              </div>
            )}

            {error && (
              <div
                role="alert"
                data-testid="reset-error"
                className="flex items-start gap-2 rounded-md border border-bear/40 bg-bear/10 px-3 py-2 text-[12px] text-bear"
              >
                <AlertOctagon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>{error.message}</span>
              </div>
            )}

            <Button
              type="submit"
              disabled={submitting}
              data-testid="reset-submit"
              className="w-full"
            >
              {submitting && (
                <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
              )}
              <CheckCircle2 className="mr-2 h-3.5 w-3.5" />
              Set new password
            </Button>
          </form>
        </div>

        <p className="mt-4 text-center font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
          The link expires 1 hour after it's sent.
        </p>
      </div>
    </div>
  );
}
