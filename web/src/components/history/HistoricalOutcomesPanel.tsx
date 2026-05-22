import { History } from "lucide-react";
import {
  formatOutcomeName,
  formatScoreName,
  useHistoricalOutcomes,
  type HistoricalBucket,
  type HistoricalOutcomes,
} from "@/lib/history";
import { cn } from "@/lib/utils";

interface Props {
  symbol: string | null;
}

/**
 * Per-symbol drill-down on the calibration data.
 *
 * The pooled ``CalibrationStrip`` shows what *the watchlist as a
 * whole* has done at a given score. This panel zooms in: what has
 * *this symbol* specifically done?
 *
 * Empty state — symbols with no observations yet — renders a clear
 * "no history yet" message rather than an empty card with phantom
 * headers. Disabled-store (404) renders nothing at all; the analyzer
 * tab already says "calibration off" via the analyzer card, so a
 * second "off" indicator here would be noise.
 */
export function HistoricalOutcomesPanel({ symbol }: Props) {
  const { outcomes, loading, error } = useHistoricalOutcomes(symbol);

  // Calibration disabled at the deployment level → no panel at all.
  if (error && error.toLowerCase().includes("disabled")) {
    return null;
  }

  if (loading && outcomes === null) {
    return (
      <section
        data-testid="history-panel"
        data-status="loading"
        className="surface p-5"
      >
        <Header symbol={symbol} subtitle="Loading per-symbol outcomes…" />
      </section>
    );
  }

  if (error) {
    return (
      <section
        data-testid="history-panel"
        data-status="error"
        className="surface p-5"
      >
        <Header
          symbol={symbol}
          subtitle="Per-symbol drill-down on the calibration data."
        />
        <p className="mt-3 text-sm text-bear">{error}</p>
      </section>
    );
  }

  if (outcomes === null) {
    return null;
  }

  if (outcomes.settled_observations === 0) {
    return (
      <section
        data-testid="history-panel"
        data-status="empty"
        className="surface p-5"
      >
        <Header
          symbol={outcomes.symbol}
          subtitle="Per-symbol drill-down on the calibration data."
        />
        <p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          No observations recorded for {outcomes.symbol} yet. Live observations
          accumulate via the snapshot loop; matured outcomes appear here once
          the calibration worker settles them.
        </p>
        <Counts outcomes={outcomes} />
      </section>
    );
  }

  return (
    <section
      data-testid="history-panel"
      data-status="ready"
      className="surface p-5"
    >
      <Header
        symbol={outcomes.symbol}
        subtitle="Per-symbol drill-down on the calibration data."
      />
      <Counts outcomes={outcomes} />
      <ul className="mt-4 space-y-3">
        {groupByPairing(outcomes.buckets).map(([key, buckets]) => (
          <li key={key}>
            <PairingBlock buckets={buckets} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function Header({
  symbol,
  subtitle,
}: {
  symbol: string | null;
  subtitle: string;
}) {
  return (
    <header className="flex items-baseline justify-between">
      <div>
        <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          <History className="h-3 w-3" />
          {symbol ? `Historical Outcomes · ${symbol}` : "Historical Outcomes"}
        </div>
        <h3 className="mt-1 text-base font-semibold tracking-tight">
          {subtitle}
        </h3>
      </div>
    </header>
  );
}

function Counts({ outcomes }: { outcomes: HistoricalOutcomes }) {
  return (
    <p className="mt-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
      {outcomes.total_observations} observations ·{" "}
      {outcomes.settled_observations} settled ·{" "}
      {outcomes.total_observations - outcomes.settled_observations} maturing
    </p>
  );
}

function groupByPairing(
  buckets: HistoricalBucket[],
): [string, HistoricalBucket[]][] {
  const map = new Map<string, HistoricalBucket[]>();
  for (const b of buckets) {
    const key = `${b.score_name}/${b.outcome_name}`;
    const list = map.get(key) ?? [];
    list.push(b);
    map.set(key, list);
  }
  // Sort each pairing's buckets by ascending bucket_lo for natural reading.
  for (const list of map.values()) {
    list.sort((a, b) => a.bucket_lo - b.bucket_lo);
  }
  return Array.from(map.entries()).sort((a, b) =>
    a[0].localeCompare(b[0]),
  );
}

function PairingBlock({ buckets }: { buckets: HistoricalBucket[] }) {
  const first = buckets[0];
  const score = formatScoreName(first.score_name);
  const direction = formatOutcomeName(first.outcome_name);
  return (
    <div
      data-testid={`history-pairing-${first.score_name}`}
      className="rounded-md border border-border/40 bg-card/40 p-3"
    >
      <div className="flex items-baseline justify-between">
        <div className="font-mono text-sm font-semibold text-foreground">
          {score} → {direction}
        </div>
        <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
          {first.horizon_days}d horizon
        </span>
      </div>
      <ul className="mt-2 space-y-1.5">
        {buckets.map((b) => (
          <li key={`${b.bucket_lo}-${b.bucket_hi}`}>
            <BucketRow bucket={b} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function BucketRow({ bucket }: { bucket: HistoricalBucket }) {
  const range = `[${bucket.bucket_lo}–${bucket.bucket_hi})`;
  if (!bucket.bucket_published) {
    return (
      <div
        data-testid={`history-bucket-${bucket.bucket_lo}`}
        data-status="pending"
        className="flex items-baseline justify-between font-mono text-[11px] tabular-nums"
      >
        <span className="text-foreground">
          {range} · {bucket.n_observations} obs · {bucket.n_hits}/
          {bucket.n_observations} hit
        </span>
        <span className="text-[9px] uppercase tracking-wider text-muted-foreground">
          insufficient sample
        </span>
      </div>
    );
  }
  return (
    <div
      data-testid={`history-bucket-${bucket.bucket_lo}`}
      data-status="published"
      className="flex flex-wrap items-baseline justify-between gap-x-2 font-mono text-[11px] tabular-nums"
    >
      <span className="text-foreground">
        {range} · {bucket.n_observations} obs · {bucket.n_hits}/
        {bucket.n_observations} ({(bucket.hit_rate * 100).toFixed(0)}%)
      </span>
      <span className={cn(toneForHit(bucket), "uppercase tracking-wider")}>
        CI {(bucket.confidence_low * 100).toFixed(0)}–
        {(bucket.confidence_high * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function toneForHit(b: HistoricalBucket): string {
  if (b.hit_rate < 0.5) return "text-muted-foreground";
  if (b.outcome_name === "return_negative") return "text-bear";
  if (b.outcome_name === "return_positive") return "text-bull";
  return "text-muted-foreground";
}
