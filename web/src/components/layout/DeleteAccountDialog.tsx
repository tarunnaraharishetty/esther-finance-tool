import { useState, type FormEvent } from "react";
import { AlertOctagon, Loader2, Trash2, X } from "lucide-react";

import { type AccountError, deleteAccount } from "@/lib/account";

interface Props {
  /** Email the user is deleting — shown verbatim in the confirm
   *  copy so the user can't be tricked into deleting the wrong
   *  account if they're logged in as the wrong identity. */
  email: string;
  /** Close without deleting. */
  onClose: () => void;
  /** Successful delete callback. The server already cleared the
   *  cookies on the response; the parent should refresh session
   *  state (which will resolve to anonymous) and route to the
   *  LoginPage. */
  onSuccess: () => void;
  /** Test seam: override the real DELETE call. */
  onConfirm?: (password: string) => Promise<void>;
}

/**
 * Confirmation modal for ``DELETE /api/auth/account``.
 *
 * Two-step affordance:
 * 1. User must type their current password (re-auth — even with
 *    a hijacked session the attacker can't silently delete).
 * 2. User must check the "I understand" checkbox. This is friction
 *    against accidental clicks; the password gate is the real
 *    security boundary.
 *
 * Renders as a centered overlay (no Radix dialog dep — small enough
 * to roll inline). Press Esc to close handled at the form level
 * via the close button + backdrop click.
 */
export function DeleteAccountDialog({
  email,
  onClose,
  onSuccess,
  onConfirm,
}: Props) {
  const [password, setPassword] = useState("");
  const [ack, setAck] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = ack && password.length > 0 && !busy;

  const handleSubmit = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    try {
      if (onConfirm) {
        await onConfirm(password);
      } else {
        await deleteAccount(password);
      }
      onSuccess();
    } catch (e) {
      const err = e as AccountError;
      setError(
        err && typeof err === "object" && "message" in err
          ? err.message
          : e instanceof Error
            ? e.message
            : "Could not delete account",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {/* Backdrop */}
      <button
        type="button"
        aria-hidden
        tabIndex={-1}
        onClick={onClose}
        data-testid="delete-dialog-backdrop"
        className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-account-title"
        data-testid="delete-account-dialog"
        className="fixed left-1/2 top-1/2 z-50 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-lg border border-bear/40 bg-card p-5 shadow-2xl"
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          data-testid="delete-dialog-close"
          className="absolute right-3 top-3 grid h-7 w-7 place-items-center rounded-md border border-border/40 text-muted-foreground transition-colors hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>

        <h2
          id="delete-account-title"
          className="flex items-center gap-2 font-display text-lg font-semibold text-bear"
        >
          <Trash2 className="h-4 w-4" />
          Delete account
        </h2>
        <p className="mt-2 text-xs text-muted-foreground">
          Permanently delete the account for{" "}
          <span className="font-mono text-foreground">{email}</span>. Your
          watchlist, sessions, and any pending tokens are erased. This cannot
          be undone.
        </p>

        <form onSubmit={handleSubmit} className="mt-4 space-y-3">
          <label className="block space-y-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              Confirm with your password
            </span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              data-testid="delete-password-input"
              disabled={busy}
              className="w-full rounded-md border border-border/60 bg-background/60 px-3 py-2 text-sm outline-none focus:border-primary/50"
            />
          </label>

          <label className="flex items-start gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={ack}
              onChange={(e) => setAck(e.target.checked)}
              data-testid="delete-ack-checkbox"
              disabled={busy}
              className="mt-0.5"
            />
            <span>
              I understand this is permanent and my data cannot be recovered.
            </span>
          </label>

          {error && (
            <div
              role="alert"
              data-testid="delete-dialog-error"
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
              data-testid="delete-cancel"
              disabled={busy}
              className="rounded-md border border-border/60 bg-card/40 px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmit}
              data-testid="delete-confirm"
              className="inline-flex items-center gap-1 rounded-md border border-bear/40 bg-bear/20 px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-bear hover:bg-bear/30 transition-colors disabled:opacity-50"
            >
              {busy && <Loader2 className="h-3 w-3 animate-spin" />}
              Delete account
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
