import {
  Activity,
  Compass,
  Layers,
  Newspaper,
  ShieldAlert,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import {
  toneClasses,
  useKeyDrivers,
  type Driver,
  type DriverKind,
} from "@/lib/movementDrivers";
import { cn } from "@/lib/utils";

interface Props {
  symbol: string | null;
}

/**
 * Key Drivers panel — top-of-AnalyzerTab ranked story.
 *
 * Surfaces the 3-5 most salient signals for the current symbol as
 * a stack of tone-toned cards. Each card carries:
 *
 *  * a kind icon + label
 *  * a one-sentence grounded detail (cites a specific value)
 *  * a salience bar (how far from baseline this signal sits)
 *  * citation chips (which data streams produced the driver)
 *
 * Empty state: when no signal clears the salience floor, the panel
 * renders a "no notable drivers right now" line rather than empty
 * chrome — that's a meaningful trader signal too ("nothing
 * remarkable, watchlist quiet").
 */
export function KeyDriversPanel({ symbol }: Props) {
  const { drivers, loading, error } = useKeyDrivers(symbol);
  if (symbol === null) return null;
  if (error !== null) return null; // analyzer surface owns errors
  if (loading && drivers === null) return <SkeletonPanel />;
  if (drivers === null) return null;

  if (drivers.drivers.length === 0) {
    return (
      <section
        data-testid="key-drivers-panel"
        data-state="empty"
        className="surface flex items-start gap-3 p-4"
      >
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/15 to-accent/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Key Drivers
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            No notable signals stand out for {symbol} right now — every
            measured stream sits near its baseline.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      data-testid="key-drivers-panel"
      data-state="ready"
      data-driver-count={drivers.drivers.length}
      className="surface-premium relative overflow-hidden p-5"
    >
      <div className="flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Key Drivers
          </div>
          <h2 className="mt-1 font-display text-lg font-semibold tracking-tight">
            What's notable about {symbol} right now
          </h2>
          {drivers.suppressed > 0 && (
            <p className="mt-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
              · {drivers.suppressed} low-signal driver
              {drivers.suppressed === 1 ? "" : "s"} suppressed
            </p>
          )}
        </div>
      </div>

      <div className="mt-4 space-y-2">
        {drivers.drivers.map((driver, i) => (
          <DriverCard key={`${driver.kind}-${i}`} driver={driver} />
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

function DriverCard({ driver }: { driver: Driver }) {
  const tone = toneClasses(driver.tone);
  const Icon = KIND_ICONS[driver.kind];
  const pct = Math.round(Math.max(0, Math.min(1, driver.salience)) * 100);
  return (
    <article
      data-testid={`driver-card-${driver.kind}`}
      data-tone={driver.tone}
      data-salience={pct}
      className={cn(
        "rounded-md border bg-card/40 px-3 py-2.5",
        tone.border,
      )}
    >
      <div className="flex items-start gap-3">
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
            <span
              className={cn("text-sm font-semibold", tone.text)}
            >
              {driver.label}
            </span>
            <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/60">
              {driver.kind}
            </span>
          </div>
          <p className="mt-1 text-[13px] leading-relaxed text-foreground/85">
            {driver.detail}
          </p>
          <div className="mt-2 flex items-center gap-2">
            <div
              className="h-1 flex-1 overflow-hidden rounded-full bg-muted/40"
              title={`salience ${pct}/100`}
            >
              <div
                className={cn("h-full rounded-full", tone.bar)}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
              salience {pct}
            </span>
          </div>
          {driver.citations.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {driver.citations.map((c) => (
                <span
                  key={c}
                  className="rounded-sm border border-border/40 bg-card/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground"
                >
                  {c}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}

function SkeletonPanel() {
  return (
    <section
      data-testid="key-drivers-panel"
      data-state="loading"
      className="surface p-4"
    >
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        Key Drivers
      </div>
      <div className="mt-3 space-y-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="h-16 w-full animate-pulse rounded-md bg-muted/40"
          />
        ))}
      </div>
    </section>
  );
}
