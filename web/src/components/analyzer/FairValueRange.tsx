import type { ValuationEnsemble } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  valuation: ValuationEnsemble | null;
  lastPrice: number | null;
}

/**
 * Bear/base/bull fair-value spectrum.
 *
 * Bear → Bull is a horizontal axis; the current price is a vertical
 * marker positioned by linear interpolation across the [bear, bull]
 * span. When the price sits outside the span we clamp the marker but
 * surface a "above bull" / "below bear" badge so the chart doesn't
 * lie.
 */
export function FairValueRange({ valuation, lastPrice }: Props) {
  if (valuation === null || valuation.bear_case === null) {
    return <EmptyState />;
  }
  const bear = valuation.bear_case;
  const base = valuation.base_case ?? (bear + (valuation.bull_case ?? bear)) / 2;
  const bull = valuation.bull_case ?? bear;
  const span = Math.max(0.0001, bull - bear);

  const pricePct =
    lastPrice !== null
      ? ((lastPrice - bear) / span) * 100
      : null;
  const pctClamped =
    pricePct === null ? null : Math.max(0, Math.min(100, pricePct));
  const basePct = ((base - bear) / span) * 100;

  const banner = priceBanner(lastPrice, bear, bull);

  return (
    <section className="surface p-5">
      <header className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Fair Value Range
          </div>
          <h3 className="mt-1 text-base font-semibold tracking-tight">
            ${valuation.weighted_ai_fair_value?.toFixed(2) ?? "—"}{" "}
            <span className="font-mono text-xs font-normal text-muted-foreground">
              · weighted AI fair value
            </span>
          </h3>
        </div>
        {banner && (
          <span
            className={cn(
              "rounded-md border px-2 py-1 font-mono text-[10px] uppercase tracking-wider",
              banner.tone,
            )}
          >
            {banner.label}
          </span>
        )}
      </header>

      <div className="relative h-10">
        <div className="absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-gradient-to-r from-bear/40 via-warn/30 to-bull/40" />
        <Marker pct={basePct} label="BASE" />
        {pctClamped !== null && (
          <CurrentPriceMarker pct={pctClamped} priceLabel={lastPrice ?? 0} />
        )}
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums">
        <CaseTile
          label="Bear"
          value={bear}
          tone="bear"
          delta={lastPrice}
        />
        <CaseTile label="Base" value={base} tone="warn" delta={lastPrice} />
        <CaseTile label="Bull" value={bull} tone="bull" delta={lastPrice} />
      </div>

      <ul className="mt-4 space-y-2">
        {valuation.estimates.map((est) => (
          <li
            key={est.method}
            className="flex items-baseline justify-between gap-3 border-b border-border/30 pb-1.5 last:border-0"
          >
            <div className="min-w-0">
              <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                {est.method.replace(/_/g, " ")}
              </div>
              <div className="truncate text-[11px] text-muted-foreground">
                {est.inputs_used}
              </div>
            </div>
            <div className="text-right">
              <div className="font-mono text-sm tabular-nums">
                ${est.fair_value.toFixed(2)}
              </div>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                conf {Math.round(est.confidence * 100)}%
              </div>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Marker({ pct, label }: { pct: number; label: string }) {
  return (
    <div
      className="absolute -top-1 h-12 w-0.5 bg-border/80"
      style={{ left: `calc(${pct}% - 1px)` }}
    >
      <span className="absolute -top-4 left-1/2 -translate-x-1/2 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

function CurrentPriceMarker({
  pct,
  priceLabel,
}: {
  pct: number;
  priceLabel: number;
}) {
  return (
    <div
      className="absolute top-1/2 z-10 -translate-y-1/2"
      style={{ left: `calc(${pct}% - 6px)` }}
    >
      <div className="grid place-items-center">
        <div className="h-3 w-3 rounded-full bg-foreground shadow-[0_0_0_3px_hsl(var(--background))]" />
        <span className="mt-1 whitespace-nowrap rounded-md bg-foreground px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-background">
          ${priceLabel.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function CaseTile({
  label,
  value,
  tone,
  delta,
}: {
  label: string;
  value: number;
  tone: "bull" | "bear" | "warn";
  delta: number | null;
}) {
  const deltaPct = delta !== null ? ((value - delta) / delta) * 100 : null;
  const deltaTone =
    deltaPct === null
      ? "text-muted-foreground"
      : deltaPct >= 0
        ? "text-bull"
        : "text-bear";
  const toneClass =
    tone === "bull"
      ? "border-bull/40 text-bull"
      : tone === "bear"
        ? "border-bear/40 text-bear"
        : "border-warn/40 text-warn";
  return (
    <div className={cn("rounded-md border bg-card/40 px-3 py-2", toneClass)}>
      <div className="font-mono text-[9px] uppercase tracking-wider opacity-80">
        {label}
      </div>
      <div className="mt-1 text-sm">${value.toFixed(2)}</div>
      {deltaPct !== null && (
        <div className={cn("font-mono text-[10px]", deltaTone)}>
          {deltaPct >= 0 ? "+" : ""}
          {deltaPct.toFixed(1)}%
        </div>
      )}
    </div>
  );
}

function priceBanner(
  lastPrice: number | null,
  bear: number,
  bull: number,
): { label: string; tone: string } | null {
  if (lastPrice === null) return null;
  if (lastPrice > bull) {
    return {
      label: "Price above bull case",
      tone: "border-bear/40 bg-bear/10 text-bear",
    };
  }
  if (lastPrice < bear) {
    return {
      label: "Price below bear case",
      tone: "border-bull/40 bg-bull/10 text-bull",
    };
  }
  return null;
}

function EmptyState() {
  return (
    <section className="surface p-5">
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        Fair Value Range
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        Valuation unavailable — no fundamentals reached the chain.
      </p>
    </section>
  );
}
