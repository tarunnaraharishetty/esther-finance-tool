import { Suspense, lazy, useCallback, useEffect, useState } from "react";
import { AlertTriangle, Bell, Loader2, TrendingDown, TrendingUp } from "lucide-react";
import { useSession } from "@/lib/auth";
import { useSnapshotStream } from "@/lib/stream";
import { lookupTicker } from "@/lib/tickers";
import { AppShell } from "@/components/layout/AppShell";
import type { NavKey } from "@/components/layout/Sidebar";
import { DashboardPage } from "@/pages/DashboardPage";
import { LoginPage } from "@/pages/LoginPage";
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
  const { snapshot, status, isStale, msSinceLastEvent } = useSnapshotStream();
  const [activeSymbol, setActiveSymbol] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState<NavKey>("dashboard");
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    if (activeSymbol !== null) return;
    if (snapshot && snapshot.rows.length > 0) {
      setActiveSymbol(snapshot.rows[0].symbol);
    }
  }, [snapshot, activeSymbol]);

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
      if (!snapshot) return;
      const match = snapshot.rows.find((r) => r.symbol === query);
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
    [snapshot],
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
  if (session.user === null) {
    return <LoginPage onLogin={session.login} onSignup={session.signup} />;
  }

  return (
    <>
      <AppShell
        status={status}
        isStale={isStale}
        tick={snapshot?.tick ?? null}
        rowCount={snapshot?.rows.length ?? 0}
        activeNav={activeNav}
        onNavChange={setActiveNav}
        onSearch={handleSearch}
        onOpenPalette={() => setPaletteOpen(true)}
        authUser={session.user}
        onLogout={session.logout}
        navCounts={{
          watchlist: snapshot?.rows.length ?? 0,
          news: snapshot
            ? snapshot.rows.reduce((s, r) => s + r.headlines.length, 0)
            : 0,
          ai:
            (snapshot?.alerts.length ?? 0) +
            (snapshot?.ranked_opportunities.length ?? 0),
        }}
      >
        {isStale && msSinceLastEvent !== null && (
          <StaleBanner ms={msSinceLastEvent} />
        )}

        <PageHeader
          title={pageTitle(activeNav)}
          subtitle={pageSubtitle(activeNav)}
          snapshot={snapshot}
        />

        {snapshot === null ? (
          <FirstLoadSkeleton />
        ) : (
          // `key` triggers the fade-in animation on every route change.
          <div key={activeNav} className="animate-fade-in">
            <Suspense fallback={<RouteSkeleton />}>
              <RouteSwitch
                nav={activeNav}
                snapshot={snapshot}
                activeSymbol={activeSymbol}
                setActiveSymbol={setActiveSymbol}
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
            snapshot={snapshot}
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
}: {
  nav: NavKey;
  snapshot: DashboardSnapshot;
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
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
