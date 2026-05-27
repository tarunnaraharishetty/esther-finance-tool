import { useState, type FormEvent } from "react";
import {
  AlertOctagon,
  KeyRound,
  Loader2,
  X,
} from "lucide-react";

import { type AccountError, changePassword } from "@/lib/account";

interface Props {
  /** Close without changing. */
  onClose: () => void;
  /** Successful change callback. Parent surfaces a flash + closes. */
  onSuccess: () => void;
  /** Test seam: override the real POST call. */
  onConfirm?: (current: string, next: string) => Promise<void>;
}

/**
 * Logged-in password rotation. Three fields:
 *
 * 1. Current password (re-auth gate; server returns 403 on mismatch).
 * 2. New password (≥8 chars; frontend enforces minLength).
 * 3. Confirm new password (local match check before the POST).
 *
 * On submit the server rotates the session cookie alongside the
 * password, so the caller stays logged in but every other device
 * for the same user gets revoked. No explicit re-login required.
 */
export function ChangePasswordDialog({
  onClose,
  onSuccess,
  onConfirm,
}: Props) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirmNext, setConfirmNext] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mismatch, setMismatch] = useState(false);

  const canSubmit =
    current.length > 0 && next.length >= 8 && confirmNext.length >= 8 && !busy;

  const handleSubmit = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    if (!canSubmit) return;
    if (next !== confirmNext) {
      setMismatch(true);
      return;
    }
    setMismatch(false);
    setError(null);
    setBusy(true);
    try {
      if (onConfirm) {
        await onConfirm(current, next);
      } else {
        await changePassword(current, next);
      }
      onSuccess();
    } catch (e) {
      const err = e as AccountError;
      setError(
        err && typeof err === "object" && "message" in err
          ? err.message
          : e instanceof Error
            ? e.message
            : "Could not change password",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button
        type="button"
        aria-hidden
        tabIndex={-1}
        onClick={onClose}
        data-testid="change-password-backdrop"
        className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="change-password-title"
        data-testid="change-password-dialog"
        className="fixed left-1/2 top-1/2 z-50 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border/60 bg-card p-5 shadow-2xl"
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          data-testid="change-password-close"
          className="absolute right-3 top-3 grid h-7 w-7 place-items-center rounded-md border border-border/40 text-muted-foreground transition-colors hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>

        <h2
          id="change-password-title"
          className="flex items-center gap-2 font-display text-lg font-semibold"
        >
          <KeyRound className="h-4 w-4" />
          Change password
        </h2>
        <p className="mt-2 text-xs text-muted-foreground">
          You'll stay logged in here. Other devices using your account will
          be signed out.
        </p>

        <form onSubmit={handleSubmit} className="mt-4 space-y-3">
          <label className="block space-y-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              Current password
            </span>
            <input
              type="password"
              value={current}
              onChange={(e) => setCurrent(e.target.value)}
              required
              autoComplete="current-password"
              data-testid="current-password-input"
              disabled={busy}
              className="w-full rounded-md border border-border/60 bg-background/60 px-3 py-2 text-sm outline-none focus:border-primary/50"
            />
          </label>
          <label className="block space-y-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              New password
            </span>
            <input
              type="password"
              value={next}
              onChange={(e) => setNext(e.target.value)}
              required
              minLength={8}
              autoComplete="new-password"
              placeholder="min 8 characters"
              data-testid="new-password-input"
              disabled={busy}
              className="w-full rounded-md border border-border/60 bg-background/60 px-3 py-2 text-sm outline-none focus:border-primary/50"
            />
          </label>
          <label className="block space-y-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              Confirm new password
            </span>
            <input
              type="password"
              value={confirmNext}
              onChange={(e) => setConfirmNext(e.target.value)}
              required
              minLength={8}
              autoComplete="new-password"
              data-testid="confirm-new-password-input"
              disabled={busy}
              className="w-full rounded-md border border-border/60 bg-background/60 px-3 py-2 text-sm outline-none focus:border-primary/50"
            />
          </label>

          {mismatch && (
            <div
              role="alert"
              data-testid="change-mismatch-error"
              className="flex items-start gap-2 rounded-md border border-bear/40 bg-bear/10 px-3 py-2 text-[12px] text-bear"
            >
              <AlertOctagon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>The two new passwords don't match.</span>
            </div>
          )}

          {error && (
            <div
              role="alert"
              data-testid="change-password-error"
              className="flex items-start gap-2 rounded-md border border-bear/40 bg-bear/10 px-3 py-2 text-[12px] text-bear"
            >
              <AlertOctagon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              data-testid="change-cancel"
              disabled={busy}
              className="rounded-md border border-border/60 bg-card/40 px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmit}
              data-testid="change-submit"
              className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {busy && <Loader2 className="h-3 w-3 animate-spin" />}
              Update password
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
