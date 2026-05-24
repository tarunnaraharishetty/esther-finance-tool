import { useState } from "react";
import { LogOut, User as UserIcon } from "lucide-react";
import { type AuthUser, displayName } from "@/lib/auth";
import { cn } from "@/lib/utils";

interface Props {
  user: AuthUser;
  onLogout: () => Promise<void>;
}

/**
 * Header avatar pill with a small popover for the logout action.
 *
 * Renders only when a real user is logged in (the unauthenticated
 * state shows the LoginPage instead of the app shell). The popover
 * surfaces the user's email + last-login timestamp + a logout
 * button — minimal account chrome appropriate to a v1 that doesn't
 * yet have a profile page.
 */
export function UserMenu({ user, onLogout }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const handleLogout = async (): Promise<void> => {
    if (busy) return;
    setBusy(true);
    try {
      await onLogout();
    } finally {
      setBusy(false);
      setOpen(false);
    }
  };

  return (
    <div className="relative" data-testid="user-menu">
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        aria-expanded={open}
        aria-label="Account menu"
        data-testid="user-menu-trigger"
        className={cn(
          "flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 py-1 pl-1 pr-3 transition-colors",
          open ? "border-primary/60" : "hover:border-primary/40",
        )}
      >
        <span className="grid h-7 w-7 place-items-center rounded-full bg-gradient-to-br from-primary to-accent shadow-glow">
          <UserIcon className="h-3.5 w-3.5 text-primary-foreground" />
        </span>
        <span className="hidden text-xs font-medium md:inline">
          {displayName(user)}
        </span>
      </button>

      {open && (
        <>
          <button
            type="button"
            aria-hidden
            tabIndex={-1}
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-30 cursor-default bg-transparent"
          />
          <div
            data-testid="user-menu-panel"
            className="absolute right-0 top-full z-40 mt-2 w-64 overflow-hidden rounded-md border border-border/60 bg-card shadow-xl"
          >
            <div className="border-b border-border/40 px-3 py-2.5">
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                Signed in as
              </div>
              <div
                className="mt-1 truncate text-sm font-medium"
                title={user.email}
              >
                {user.email}
              </div>
              {user.last_login_at && (
                <div className="mt-1 font-mono text-[10px] text-muted-foreground/70">
                  last login · {formatRelative(user.last_login_at)}
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => void handleLogout()}
              disabled={busy}
              data-testid="user-menu-logout"
              className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-muted/60 disabled:opacity-50"
            >
              <LogOut className="h-3.5 w-3.5" />
              {busy ? "Logging out…" : "Log out"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function formatRelative(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const ms = Date.now() - t;
  if (ms < 60_000) return "just now";
  const m = Math.round(ms / 60_000);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}
