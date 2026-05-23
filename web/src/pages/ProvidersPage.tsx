import {
  AlertOctagon,
  Database,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import {
  formatLatency,
  formatPercent,
  formatWeight,
  trustTone,
  useProviders,
  type ProviderRow,
  type ProvidersResponse,
} from "@/lib/providers";
import { cn } from "@/lib/utils";

interface Props {
  /** Window the trader picks from the top bar. v1 ships a single
   *  default ("1h" rolling) — the field is here so a future filter
   *  panel can flow through without re-plumbing. */
  window?: string;
}

/**
 * Data Providers health page.
 *
 * Surfaces what the platform's been doing under the hood per
 * fundamentals provider: rolling success rate, p95 latency, the
 * composite trust weight that multiplies provider_confidence,
 * and per-field empirical accuracy (the data moat the analyzer
 * Trust Score draws on).
 *
 * Operator UX: scan the strip of cards, click in on the degraded
 * one. Trader UX: confirm at a glance that "the numbers I see are
 * empirically calibrated." Investor UX: visible proof of the
 * data moat.
 */
export function ProvidersPage({ window = "1h" }: Props) {
  const { data, loading, error, refresh } = useProviders(window);

  if (error !== null) {
    return (
      <section className="surface flex items-start gap-3 border-bear/40 p-4">
        <AlertOctagon className="mt-0.5 h-4 w-4 shrink-0 text-bear" />
        <div className="flex-1">
          <div className="font-mono text-[10px] uppercase tracking-wider text-bear">
            Providers feed unavailable
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{error}</p>
          <button
            type="button"
            onClick={() => void refresh()}
            className="mt-2 inline-flex items-center gap-1 rounded-md border border-border/40 bg-card/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider hover:bg-card"
          >
            <RefreshCw className="h-3 w-3" /> Retry
          </button>
        </div>
      </section>
    );
  }

  if (data === null && loading) return <SkeletonPage />;
  if (data === null) return null;

  return (
    <div className="space-y-4">
      <ProvidersHero data={data} loading={loading} onRefresh={refresh} />

      {data.providers.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {data.providers.map((p) => (
            <ProviderCard key={p.provider} row={p} />
          ))}
        </div>
      )}
    </div>
  );
}

function ProvidersHero({
  data,
  loading,
  onRefresh,
}: {
  data: ProvidersResponse;
  loading: boolean;
  onRefresh: () => Promise<void>;
}) {
  const asOf = new Date(data.as_of);
  const windowH = Math.round(data.window_seconds / 3600);
  return (
    <section
      data-testid="providers-hero"
      className="surface-premium relative overflow-hidden p-5"
    >
      <div className="flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
          <ShieldCheck className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Data Providers
          </div>
          <h1 className="mt-1 font-display text-2xl font-semibold tracking-tightest md:text-[28px]">
            Empirically calibrated trust per provider
          </h1>
          <p className="mt-1 text-[13px] text-muted-foreground">
            Every confidence number this platform surfaces is multiplied by a
            trust weight derived from {data.providers.length} provider
            {data.providers.length === 1 ? "" : "s"}' observed accuracy +
            uptime + latency across a rolling {windowH || 1}h window. Click any
            card to see the breakdown.
          </p>
          <div className="mt-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            · refreshed {asOf.toLocaleTimeString()}
          </div>
        </div>
        <button
          type="button"
          onClick={() => void onRefresh()}
          disabled={loading}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border/40 bg-card/60 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:bg-card hover:text-foreground disabled:opacity-40"
        >
          <RefreshCw className={loading ? "h-3 w-3 animate-spin" : "h-3 w-3"} />
          Refresh
        </button>
      </div>
    </section>
  );
}

function ProviderCard({ row }: { row: ProviderRow }) {
  const trust = row.trust_breakdown;
  const weight = trust?.weight ?? 1.0;
  const tone = trustTone(weight);
  return (
    <section
      data-testid={`provider-card-${row.provider}`}
      data-tone={tone}
      className={cn(
        "surface flex flex-col gap-3 border p-4",
        tone === "bull" && "border-bull/40",
        tone === "warn" && "border-warn/40",
        tone === "bear" && "border-bear/40",
        tone === "muted" && "border-border/40",
      )}
    >
      <header className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="font-mono text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
            Provider
          </div>
          <h2 className="font-display text-lg font-semibold tracking-tight">
            {row.provider}
          </h2>
          {trust?.cold_start && (
            <span
              data-testid="cold-start-badge"
              className="mt-1 inline-flex items-center rounded-md border border-border/40 bg-card/40 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground"
            >
              cold start · 1.00 weight
            </span>
          )}
        </div>
        <WeightGauge weight={weight} tone={tone} />
      </header>

      <div className="grid grid-cols-3 gap-2">
        <ComponentBar
          label="Accuracy"
          value={trust?.accuracy ?? null}
          tone={tone}
        />
        <ComponentBar
          label="Uptime"
          value={trust?.uptime ?? null}
          tone={tone}
        />
        <ComponentBar
          label="Latency"
          value={trust?.latency_score ?? null}
          tone={tone}
          secondary={formatLatency(trust?.latency_p95_ms ?? null)}
        />
      </div>

      <FieldAccuracyTable row={row} />

      <footer className="flex items-center justify-between gap-2 border-t border-border/30 pt-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>
          <Database className="mr-1 inline h-3 w-3" />
          {trust?.events ?? 0} events · {trust?.health_calls ?? 0} calls
        </span>
        <span>success {formatPercent(row.success_rate)}</span>
      </footer>
    </section>
  );
}

function WeightGauge({
  weight,
  tone,
}: {
  weight: number;
  tone: "bull" | "warn" | "bear" | "muted";
}) {
  return (
    <div
      data-testid="weight-gauge"
      data-weight={formatWeight(weight)}
      className={cn(
        "shrink-0 rounded-md border px-2 py-1 text-right font-mono",
        tone === "bull" && "border-bull/40 bg-bull/15 text-bull",
        tone === "warn" && "border-warn/40 bg-warn/15 text-warn",
        tone === "bear" && "border-bear/40 bg-bear/15 text-bear",
        tone === "muted" && "border-border/40 bg-card/40 text-muted-foreground",
      )}
    >
      <div className="text-[9px] uppercase tracking-wider">trust weight</div>
      <div className="text-xl font-bold tabular-nums">
        {formatWeight(weight)}
      </div>
    </div>
  );
}

function ComponentBar({
  label,
  value,
  tone,
  secondary,
}: {
  label: string;
  value: number | null;
  tone: "bull" | "warn" | "bear" | "muted";
  secondary?: string;
}) {
  const pct = value === null ? 0 : Math.round(value * 100);
  const display = value === null ? "—" : `${pct}%`;
  return (
    <div
      data-testid={`component-${label.toLowerCase()}`}
      className="rounded-md border border-border/30 bg-card/40 px-2 py-1.5"
    >
      <div className="flex items-baseline justify-between gap-1">
        <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-[10px] font-semibold tabular-nums">
          {display}
        </span>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-muted/40">
        <div
          className={cn(
            "h-full rounded-full transition-all",
            value === null && "bg-muted-foreground/30",
            tone === "bull" && value !== null && "bg-bull",
            tone === "warn" && value !== null && "bg-warn",
            tone === "bear" && value !== null && "bg-bear",
            tone === "muted" && value !== null && "bg-muted-foreground/60",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      {secondary && (
        <div className="mt-0.5 font-mono text-[9px] text-muted-foreground">
          {secondary}
        </div>
      )}
    </div>
  );
}

function FieldAccuracyTable({ row }: { row: ProviderRow }) {
  const block = row.field_accuracy;
  if (block === null || block.by_field.length === 0) {
    return (
      <div
        data-testid="field-accuracy-empty"
        className="rounded-md border border-border/30 bg-card/30 px-3 py-2 text-[11px] text-muted-foreground"
      >
        No reconciliation events for {row.provider} in this window. The trust
        weight defaults to 1.0 until ~20 comparisons accumulate.
      </div>
    );
  }
  return (
    <div
      data-testid="field-accuracy-table"
      className="rounded-md border border-border/30 bg-card/30"
    >
      <header className="border-b border-border/30 px-3 py-1.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        Per-field agree rate · {block.total_events} events
      </header>
      <div>
        {block.by_field.map((f) => (
          <div
            key={f.field}
            data-testid={`field-row-${f.field}`}
            className="grid grid-cols-[1fr_auto_auto] items-center gap-2 border-t border-border/20 px-3 py-1.5 first:border-t-0"
          >
            <span className="font-mono text-[11px] text-foreground">
              {f.field}
            </span>
            <span className="font-mono text-[10px] text-muted-foreground">
              {f.agreed}/{f.total}
            </span>
            <span className="font-mono text-[11px] font-semibold tabular-nums">
              {Math.round(f.accuracy * 100)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <section className="surface flex items-start gap-3 p-4">
      <Database className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div>
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          No provider data yet
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          No fundamentals fetches have completed in this window. Trust weights
          default to 1.0 until the ledger warms up.
        </p>
      </div>
    </section>
  );
}

function SkeletonPage() {
  return (
    <div className="space-y-4">
      <div className="surface-premium p-5">
        <div className="h-4 w-32 animate-pulse rounded bg-muted/40" />
        <div className="mt-3 h-8 w-2/3 animate-pulse rounded bg-muted/40" />
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-48 animate-pulse rounded-md bg-muted/40" />
        ))}
      </div>
    </div>
  );
}
