import {
  Activity,
  Compass,
  Layers,
  Newspaper,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { useSpotlight, type SpotlightEntry } from "@/lib/spotlight";
import { toneClasses, type DriverKind } from "@/lib/movementDrivers";
import { cn } from "@/lib/utils";

interface Props {
  /** Triggers a refetch when this value changes (e.g. watchlist size). */
  watchlistFingerprint?: unknown;
  /** Clicking a spotlight entry navigates the user to that symbol. */
  onSelectSymbol: (symbol: string) => void;
}

/**
 * Watchlist-level Driver Spotlight widget.
 *
 * Renders the loudest per-symbol drivers ranked across the entire
 * watchlist. Each row links to its symbol — clicking jumps the
 * active selection so the next Analyzer/Charts/Research nav
 * lands on the surfaced symbol. Empty / quiet states are
 * surfaced honestly rather than hidden.
 */
export function DriverSpotlight({
  watchlistFingerprint,
  onSelectSymbol,
}: Props) {
  const { spotlight, loading, error, refresh } = useSpotlight(
    watchlistFingerprint,
  );

  if (error !== null) {
    return (
      <section
        data-testid="driver-spotlight"
        data-state="error"
        className="surface flex items-start gap-3 border-bear/40 p-4"
      >
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-bear" />
        <div className="flex-1">
          <div className="font-mono text-[10px] uppercase tracking-wider text-bear">
            Spotlight failed
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{error}</p>
          <button
            type="button"
            onClick={() => void refresh({ fresh: true })}
            className="mt-2 inline-flex items-center gap-1 rounded-md border border-border/40 bg-card/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider hover:bg-card"
          >
            <RefreshCw className="h-3 w-3" /> Retry
          </button>
        </div>
      </section>
    );
  }

  if (loading && spotlight === null) return <SkeletonPanel />;
  if (spotlight === null) return null;

  if (spotlight.entries.length === 0) {
    return (
      <section
        data-testid="driver-spotlight"
        data-state="empty"
        className="surface flex items-start gap-3 p-4"
      >
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/15 to-accent/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Watchlist Spotlight
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {spotlight.reason
              ? spotlight.reason
              : `No notable signals across your watchlist right now — all ${spotlight.watchlist_size} symbols sit near baseline.`}
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      data-testid="driver-spotlight"
      data-state="ready"
      data-entry-count={spotlight.entries.length}
      className="surface-premium relative overflow-hidden p-5"
    >
      <header className="flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Watchlist Spotlight
          </div>
          <h2 className="mt-1 font-display text-lg font-semibold tracking-tight">
            Loudest signals across your watchlist
          </h2>
          <p className="mt-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            · {spotlight.surfaced_symbols}/{spotlight.watchlist_size} symbols
            surfaced · {spotlight.entries.length}{" "}
            {spotlight.entries.length === 1 ? "entry" : "entries"}
          </p>
        </div>
        <button
          type="button"
          onClick={() => void refresh({ fresh: true })}
          disabled={loading}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border/40 bg-card/60 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:bg-card hover:text-foreground disabled:opacity-40"
        >
          <RefreshCw className={loading ? "h-3 w-3 animate-spin" : "h-3 w-3"} />
          Refresh
        </button>
      </header>

      <div className="mt-4 grid gap-2 md:grid-cols-2">
        {spotlight.entries.map((entry) => (
          <SpotlightCard
            key={`${entry.symbol}-${entry.rank}`}
            entry={entry}
            onSelect={onSelectSymbol}
          />
        ))}
      </div>
    </section>
  );
}

const KIND_ICONS: Record<DriverKind, LucideIcon> = {
  technical: Activity,
  sentiment: Compass,
  news: Newspaper,
  sector: Layers,
  trust: ShieldAlert,
  calibration: Sparkles,
};

function SpotlightCard({
  entry,
  onSelect,
}: {
  entry: SpotlightEntry;
  onSelect: (symbol: string) => void;
}) {
  const tone = toneClasses(entry.driver.tone);
  const Icon = KIND_ICONS[entry.driver.kind];
  const pct = Math.round(Math.max(0, Math.min(1, entry.driver.salience)) * 100);
  return (
    <button
      type="button"
      onClick={() => onSelect(entry.symbol)}
      data-testid={`spotlight-card-${entry.symbol}`}
      data-tone={entry.driver.tone}
      data-rank={entry.rank}
      className={cn(
        "group flex w-full items-start gap-3 rounded-md border bg-card/40 px-3 py-2.5 text-left transition-colors hover:bg-card/70",
        tone.border,
      )}
    >
      <div
        className={cn(
          "grid h-7 w-7 shrink-0 place-items-center rounded-md",
          tone.bg,
          tone.text,
        )}
      >
        <Icon className="h-3.5 w-3.5" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-sm font-bold tracking-tight text-foreground group-hover:text-primary">
            {entry.symbol}
          </span>
          <span
            className={cn(
              "text-[12px] font-semibold",
              tone.text,
            )}
          >
            {entry.driver.label}
          </span>
        </div>
        <p className="mt-1 text-[12px] leading-snug text-muted-foreground line-clamp-2">
          {entry.driver.detail}
        </p>
        <div className="mt-1.5 flex items-center gap-2">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted/40">
            <div
              className={cn("h-full rounded-full", tone.bar)}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
            #{entry.rank} · sal {pct}
          </span>
        </div>
      </div>
    </button>
  );
}

function SkeletonPanel() {
  return (
    <section
      data-testid="driver-spotlight"
      data-state="loading"
      className="surface p-4"
    >
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        Watchlist Spotlight
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="h-16 w-full animate-pulse rounded-md bg-muted/40"
          />
        ))}
      </div>
    </section>
  );
}
