import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Bell, Loader2, TrendingDown, TrendingUp } from "lucide-react";
import { useSession } from "@/lib/auth";
import { useSnapshotStream } from "@/lib/stream";
import { lookupTicker } from "@/lib/tickers";
import { useUserWatchlist } from "@/lib/userWatchlist";
import { AppShell } from "@/components/layout/AppShell";
import type { NavKey } from "@/components/layout/Sidebar";
import { DashboardPage } from "@/pages/DashboardPage";
import { LoginPage } from "@/pages/LoginPage";
import { ResetPasswordPage } from "@/pages/ResetPasswordPage";
import { VerifyEmailPage } from "@/pages/VerifyEmailPage";
import { WatchlistPage } from "@/pages/WatchlistPage";
import { MoversPage } from "@/pages/MoversPage";
import { NewsPage } from "@/pages/NewsPage";
import { AiPage } from "@/pages/AiPage";
import { ResearchPage } from "@/pages/ResearchPage";
import { AnalyzerPage } from "@/pages/AnalyzerPage";
import { ComparePage } from "@/pages/ComparePage";
import { ProvidersPage } from "@/pages/ProvidersPage";
import type { DashboardSnapshot } from "@/lib/types";

// Code-split the chart-heavy page and the palette so the initial
// bundle stays light. Recharts + cmdk are only pulled in when the
// trader actually navigates / opens them.
const ChartsPage = lazy(() =>
  import("@/pages/ChartsPage").then((m) => ({ default: m.ChartsPage })),
);
const CommandPalette = lazy(() =>
  import("@/components/dashboard/CommandPalette").then((m) => ({
    default: m.CommandPalette,
  })),
);

export default function App() {
  // Session must resolve before we know whether to render the app
  // shell or the login screen. While the /api/auth/me request is in
  // flight we render a neutral loader — flipping straight from
  // "logged out" to "logged in" without the loader would briefly
  // flash the LoginPage on every refresh.
  const session = useSession();
  // Single-page routing for the two emailed-link flows:
  //   /reset-password?token=…   → ResetPasswordPage
  //   /verify-email?token=…     → VerifyEmailPage
  // We don't run a full router for two pages — capture both the
  // pathname + token at mount, clear from the URL once consumed so
  // a back-navigation doesn't replay the form against a stale token.
  const [emailToken, setEmailToken] = useState<{
    kind: "reset" | "verify";
    token: string;
  } | null>(() => {
    if (typeof window === "undefined") return null;
    const params = new URLSearchParams(window.location.search);
    const tok = params.get("token");
    if (!tok) return null;
    const path = window.location.pathname;
    if (path.startsWith("/reset-password")) return { kind: "reset", token: tok };
    if (path.startsWith("/verify-email")) return { kind: "verify", token: tok };
    // Path doesn't match either flow — treat as a stale link and
    // fall through to the normal app (the routes above all carry
    // the ``?token=`` query string AND the matching path).
    return null;
  });
  const [resetFlash, setResetFlash] = useState<string | null>(null);

  const clearEmailToken = (): void => {
    setEmailToken(null);
    if (typeof window !== "undefined") {
      window.history.replaceState(null, "", "/");
    }
  };
  const watchlist = useUserWatchlist(session.user !== null);
  const { snapshot, status, isStale, msSinceLastEvent } = useSnapshotStream();
  const [activeSymbol, setActiveSymbol] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState<NavKey>("dashboard");
  const [paletteOpen, setPaletteOpen] = useState(false);

  // Derive a snapshot view filtered to the user's watchlist. Until
  // the watchlist GET resolves we show the unfiltered snapshot so
  // signed-in users don't see a flash of empty rows on first load.
  // Once resolved, the user's symbols are the source of truth for
  // which rows the pages render — including the empty state when
  // they've intentionally cleared their list.
  const displayedSnapshot = useMemo(() => {
    if (snapshot === null) return null;
    if (session.user === null) return snapshot;
    if (!watchlist.resolved) return snapshot;
    const wanted = new Set(watchlist.symbols);
    const filteredRows = snapshot.rows.filter((r) => wanted.has(r.symbol));
    // Drop pulse so PageHeader bull/bear counts derive from the
    // filtered rows instead of the server's full-snapshot aggregate.
    // Alerts + ranked_opportunities pass through unfiltered: they
    // help users discover symbols outside their watchlist.
    return { ...snapshot, rows: filteredRows, pulse: null };
  }, [snapshot, session.user, watchlist.resolved, watchlist.symbols]);

  useEffect(() => {
    if (displayedSnapshot === null) return;
    // If activeSymbol is missing or no longer in the filtered set,
    // snap to the first row. Empty-list users land on null and the
    // per-page empty states render.
    if (
      activeSymbol === null ||
      !displayedSnapshot.rows.some((r) => r.symbol === activeSymbol)
    ) {
      if (displayedSnapshot.rows.length > 0) {
        setActiveSymbol(displayedSnapshot.rows[0].symbol);
      } else {
        setActiveSymbol(null);
      }
    }
  }, [displayedSnapshot, activeSymbol]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((prev) => !prev);
      } else if (e.key === "Escape" && paletteOpen) {
        e.preventDefault();
        setPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [paletteOpen]);

  const handleSearch = useCallback(
    (query: string): void => {
      if (!displayedSnapshot) return;
      const match = displayedSnapshot.rows.find((r) => r.symbol === query);
      if (match) {
        setActiveSymbol(query);
        setActiveNav("charts");
        return;
      }
      // Off-watchlist symbol: only route if the universe knows about
      // it; otherwise drop the query so the search box doesn't navigate
      // into a broken state.
      if (lookupTicker(query)) {
        setActiveSymbol(query);
        setActiveNav("charts");
      }
    },
    [displayedSnapshot],
  );

  const handleSelectSymbolFromPalette = useCallback((symbol: string): void => {
    setActiveSymbol(symbol);
    setActiveNav("charts");
  }, []);

  // Pre-render gate: hold the login flip until /me settles. Once
  // resolved, an absent user redirects to the LoginPage; a present
  // user falls through to the app shell.
  if (!session.resolved) {
    return (
      <div
        data-testid="auth-loading"
        className="grid min-h-screen place-items-center bg-background"
      >
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }
  // Email-verification redemption is reachable whether the user is
  // logged in or not — the emailed link works in any session state.
  // ``onSuccess`` clears the token from the URL and refreshes the
  // session so the verified flag round-trips into the banner state.
  if (emailToken !== null && emailToken.kind === "verify") {
    return (
      <VerifyEmailPage
        token={emailToken.token}
        onSuccess={() => {
          clearEmailToken();
          void session.refresh();
        }}
      />
    );
  }

  if (session.user === null) {
    // Reset flow: URL has /reset-password?token=… and we're logged
    // out — route to the redemption form. Once it succeeds we clear
    // the token, flip on a flash banner, and fall through to the
    // LoginPage.
    if (emailToken !== null && emailToken.kind === "reset") {
      return (
        <ResetPasswordPage
          token={emailToken.token}
          onSuccess={() => {
            clearEmailToken();
            setResetFlash("Password updated. Please log in.");
          }}
        />
      );
    }
    return (
      <LoginPage
        onLogin={session.login}
        onSignup={session.signup}
        flashMessage={resetFlash}
      />
    );
  }

  return (
    <>
      <AppShell
        status={status}
        isStale={isStale}
        tick={displayedSnapshot?.tick ?? null}
        rowCount={displayedSnapshot?.rows.length ?? 0}
        activeNav={activeNav}
        onNavChange={setActiveNav}
        onSearch={handleSearch}
        onOpenPalette={() => setPaletteOpen(true)}
        authUser={session.user}
        onLogout={session.logout}
        onAccountDeleted={async () => {
          // Server cleared the cookies on the response; refresh the
          // session so /me resolves to null and the LoginPage takes
          // over. setResetFlash surfaces a one-time banner so the
          // user knows the delete completed.
          await session.refresh();
          setResetFlash("Your account has been deleted.");
        }}
        navCounts={{
          watchlist: displayedSnapshot?.rows.length ?? 0,
          news: displayedSnapshot
            ? displayedSnapshot.rows.reduce(
                (s, r) => s + r.headlines.length,
                0,
              )
            : 0,
          ai:
            (displayedSnapshot?.alerts.length ?? 0) +
            (displayedSnapshot?.ranked_opportunities.length ?? 0),
        }}
      >
        {isStale && msSinceLastEvent !== null && (
          <StaleBanner ms={msSinceLastEvent} />
        )}

        <PageHeader
          title={pageTitle(activeNav)}
          subtitle={pageSubtitle(activeNav)}
          snapshot={displayedSnapshot}
        />

        {displayedSnapshot === null ? (
          <FirstLoadSkeleton />
        ) : (
          // `key` triggers the fade-in animation on every route change.
          <div key={activeNav} className="animate-fade-in">
            <Suspense fallback={<RouteSkeleton />}>
              <RouteSwitch
                nav={activeNav}
                snapshot={displayedSnapshot}
                activeSymbol={activeSymbol}
                setActiveSymbol={setActiveSymbol}
                editor={
                  session.user !== null
                    ? {
                        symbols: watchlist.symbols,
                        onAdd: watchlist.add,
                        onRemove: watchlist.remove,
                        busy: watchlist.loading,
                        resolved: watchlist.resolved,
                      }
                    : null
                }
              />
            </Suspense>
          </div>
        )}
      </AppShell>

      {paletteOpen && (
        <Suspense fallback={null}>
          <CommandPalette
            open={paletteOpen}
            onClose={() => setPaletteOpen(false)}
            snapshot={displayedSnapshot}
            onSelectSymbol={handleSelectSymbolFromPalette}
            onNavigate={setActiveNav}
          />
        </Suspense>
      )}
    </>
  );
}

function RouteSwitch({
  nav,
  snapshot,
  activeSymbol,
  setActiveSymbol,
  editor,
}: {
  nav: NavKey;
  snapshot: DashboardSnapshot;
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
  editor: {
    symbols: string[];
    onAdd: (symbol: string) => Promise<void>;
    onRemove: (symbol: string) => Promise<void>;
    busy: boolean;
    resolved: boolean;
  } | null;
}) {
  switch (nav) {
    case "dashboard":
      return (
        <DashboardPage
          snapshot={snapshot}
          activeSymbol={activeSymbol}
          setActiveSymbol={setActiveSymbol}
        />
      );
    case "charts":
      return (
        <ChartsPage
          snapshot={snapshot}
          activeSymbol={activeSymbol}
          setActiveSymbol={setActiveSymbol}
        />
      );
    case "watchlist":
      return (
        <WatchlistPage
          snapshot={snapshot}
          activeSymbol={activeSymbol}
          setActiveSymbol={setActiveSymbol}
          editor={editor}
        />
      );
    case "movers":
      return (
        <MoversPage snapshot={snapshot} setActiveSymbol={setActiveSymbol} />
      );
    case "news":
      return (
        <NewsPage snapshot={snapshot} setActiveSymbol={setActiveSymbol} />
      );
    case "ai":
      return <AiPage snapshot={snapshot} setActiveSymbol={setActiveSymbol} />;
    case "research":
      return (
        <ResearchPage
          snapshot={snapshot}
          activeSymbol={activeSymbol}
          setActiveSymbol={setActiveSymbol}
        />
      );
    case "analyzer":
      return (
        <AnalyzerPage
          snapshot={snapshot}
          activeSymbol={activeSymbol}
          setActiveSymbol={setActiveSymbol}
        />
      );
    case "compare":
      return (
        <ComparePage
          snapshot={snapshot}
          activeSymbol={activeSymbol}
          setActiveSymbol={setActiveSymbol}
        />
      );
    case "providers":
      return <ProvidersPage />;
  }
}

function pageTitle(nav: NavKey): string {
  switch (nav) {
    case "dashboard":
      return "Market Dashboard";
    case "watchlist":
      return "Watchlist";
    case "movers":
      return "Market Movers";
    case "charts":
      return "Charts";
    case "news":
      return "News Feed";
    case "ai":
      return "AI Insights";
    case "research":
      return "Research Thesis";
    case "analyzer":
      return "Financial Analyzer";
    case "compare":
      return "Compare";
    case "providers":
      return "Data Providers";
  }
}

function pageSubtitle(nav: NavKey): string {
  switch (nav) {
    case "dashboard":
      return "Live signals · pulse · ranked opportunities";
    case "watchlist":
      return "Active tracker";
    case "movers":
      return "Gainers · losers · unusual activity";
    case "charts":
      return "Tape · timeframe · indicators";
    case "news":
      return "Headlines · sentiment · importance";
    case "ai":
      return "Per-symbol AI reads";
    case "research":
      return "Institutional-style report · per symbol";
    case "analyzer":
      return "Scored read · valuation ensemble · grounded AI";
    case "compare":
      return "Two symbols · per-metric verdicts · grounded";
    case "providers":
      return "Trust weights · accuracy ledger · per-field calibration";
  }
}

function PageHeader({
  title,
  subtitle,
  snapshot,
}: {
  title: string;
  subtitle: string;
  snapshot: DashboardSnapshot | null;
}) {
  const bull =
    snapshot?.pulse?.bullish_count ??
    snapshot?.rows.filter((r) => r.action === "buy").length ??
    0;
  const bear =
    snapshot?.pulse?.bearish_count ??
    snapshot?.rows.filter((r) => r.action === "sell").length ??
    0;
  const alerts = snapshot?.alerts.length ?? 0;
  return (
    <div className="mb-5 animate-fade-in">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="font-display text-2xl font-semibold tracking-tightest md:text-[28px]">
            {title}
          </h1>
          <p className="mt-0.5 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            {subtitle}
          </p>
        </div>
        {snapshot && (
          <div className="flex items-center gap-1.5 self-start">
            <StatPill
              icon={TrendingUp}
              label="long"
              value={bull}
              tone="bull"
            />
            <StatPill
              icon={TrendingDown}
              label="short"
              value={bear}
              tone="bear"
            />
            <StatPill
              icon={Bell}
              label="alerts"
              value={alerts}
              tone={alerts > 0 ? "warn" : "muted"}
            />
            <div className="flex items-center gap-1.5 rounded-md border border-border/50 bg-card/40 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping-soft rounded-full bg-bull/60 opacity-60" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-bull" />
              </span>
              tick {snapshot.tick}
            </div>
          </div>
        )}
      </div>
      <div className="mt-4 h-px w-full bg-gradient-to-r from-border/60 via-border/30 to-transparent" />
    </div>
  );
}

function StatPill({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
  tone: "bull" | "bear" | "warn" | "muted";
}) {
  const toneClass =
    tone === "bull"
      ? "border-bull/30 text-bull"
      : tone === "bear"
        ? "border-bear/30 text-bear"
        : tone === "warn"
          ? "border-warn/30 text-warn"
          : "border-border/50 text-muted-foreground";
  return (
    <div
      className={`flex items-center gap-1.5 rounded-md border bg-card/40 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider ${toneClass}`}
      title={`${value} ${label}`}
    >
      <Icon className="h-3 w-3" />
      <span className="font-semibold tabular-nums">{value}</span>
      <span className="opacity-70">{label}</span>
    </div>
  );
}

function StaleBanner({ ms }: { ms: number }) {
  return (
    <div
      role="status"
      className="mb-4 flex items-start gap-3 rounded-lg border border-warn/30 bg-warn/10 px-4 py-3 text-sm text-warn animate-fade-in"
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div>
        <div className="font-medium">Data may be stale</div>
        <div className="text-xs text-warn/80">
          Last update {Math.round(ms / 1000)}s ago. Stream is open but no fresh
          tick has arrived.
        </div>
      </div>
    </div>
  );
}

function FirstLoadSkeleton() {
  return (
    <div className="space-y-4 animate-fade-in">
      <div className="h-36 rounded-xl border border-border/60 bg-card/60 overflow-hidden">
        <div className="shimmer h-full w-full opacity-60" />
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="h-44 rounded-xl border border-border/60 bg-card/60 overflow-hidden"
          >
            <div className="shimmer h-full w-full opacity-50" />
          </div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="h-64 rounded-xl border border-border/60 bg-card/60 overflow-hidden"
          >
            <div className="shimmer h-full w-full opacity-40" />
          </div>
        ))}
      </div>
    </div>
  );
}

function RouteSkeleton() {
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {Array.from({ length: 3 }).map((_, i) => (
        <div
          key={i}
          className="h-64 rounded-xl border border-border/60 bg-card/60 overflow-hidden"
        >
          <div className="shimmer h-full w-full opacity-40" />
        </div>
      ))}
    </div>
  );
}
