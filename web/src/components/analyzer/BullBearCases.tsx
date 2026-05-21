import type { ValuationEnsemble } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  valuation: ValuationEnsemble | null;
  lastPrice: number | null;
}

/**
 * Three-up bull / base / bear case tiles, each carrying the case
 * price + percent-vs-last and a stacked list of methods that drove
 * the percentile boundary. Designed to read at a glance: each tile
 * answers "what does this scenario imply about price?".
 */
export function BullBearCases({ valuation, lastPrice }: Props) {
  if (valuation === null || valuation.bear_case === null) {
    return null;
  }
  const cases: CaseEntry[] = [
    {
      label: "Bear case",
      value: valuation.bear_case,
      tone: "bear",
      sub: "25th percentile of populated methods",
      methods: lowestThree(valuation),
    },
    {
      label: "Base case",
      value: valuation.base_case ?? valuation.bear_case,
      tone: "warn",
      sub: "Median across methods",
      methods: centralThree(valuation),
    },
    {
      label: "Bull case",
      value: valuation.bull_case ?? valuation.bear_case,
      tone: "bull",
      sub: "75th percentile of populated methods",
      methods: highestThree(valuation),
    },
  ];

  return (
    <section className="grid gap-3 md:grid-cols-3">
      {cases.map((c) => (
        <CaseCard key={c.label} entry={c} lastPrice={lastPrice} />
      ))}
    </section>
  );
}

interface CaseEntry {
  label: string;
  value: number;
  tone: "bull" | "bear" | "warn";
  sub: string;
  methods: { method: string; fair_value: number }[];
}

function CaseCard({
  entry,
  lastPrice,
}: {
  entry: CaseEntry;
  lastPrice: number | null;
}) {
  const deltaPct =
    lastPrice !== null && lastPrice !== 0
      ? ((entry.value - lastPrice) / lastPrice) * 100
      : null;
  const deltaTone =
    deltaPct === null
      ? "text-muted-foreground"
      : deltaPct >= 0
        ? "text-bull"
        : "text-bear";
  const toneClass =
    entry.tone === "bull"
      ? "border-bull/40 bg-gradient-to-br from-bull/10 to-transparent"
      : entry.tone === "bear"
        ? "border-bear/40 bg-gradient-to-br from-bear/10 to-transparent"
        : "border-warn/40 bg-gradient-to-br from-warn/10 to-transparent";
  const labelTone =
    entry.tone === "bull"
      ? "text-bull"
      : entry.tone === "bear"
        ? "text-bear"
        : "text-warn";

  return (
    <article
      className={cn(
        "surface p-4 border",
        toneClass,
      )}
    >
      <div className={cn("font-mono text-[10px] uppercase tracking-[0.22em]", labelTone)}>
        {entry.label}
      </div>
      <div className="mt-1 font-display text-2xl font-semibold tracking-tight tabular-nums">
        ${entry.value.toFixed(2)}
      </div>
      {deltaPct !== null && (
        <div className={cn("font-mono text-xs tabular-nums", deltaTone)}>
          {deltaPct >= 0 ? "+" : ""}
          {deltaPct.toFixed(1)}% vs last
        </div>
      )}
      <p className="mt-2 text-xs text-muted-foreground">{entry.sub}</p>
      {entry.methods.length > 0 && (
        <ul className="mt-3 space-y-1 border-t border-border/40 pt-2">
          {entry.methods.map((m) => (
            <li
              key={m.method}
              className="flex items-baseline justify-between font-mono text-[11px]"
            >
              <span className="uppercase tracking-wider text-muted-foreground">
                {m.method.replace(/_/g, " ")}
              </span>
              <span className="tabular-nums">${m.fair_value.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

function lowestThree(v: ValuationEnsemble): CaseEntry["methods"] {
  return [...v.estimates]
    .sort((a, b) => a.fair_value - b.fair_value)
    .slice(0, 3)
    .map((e) => ({ method: e.method, fair_value: e.fair_value }));
}

function highestThree(v: ValuationEnsemble): CaseEntry["methods"] {
  return [...v.estimates]
    .sort((a, b) => b.fair_value - a.fair_value)
    .slice(0, 3)
    .map((e) => ({ method: e.method, fair_value: e.fair_value }));
}

function centralThree(v: ValuationEnsemble): CaseEntry["methods"] {
  const sorted = [...v.estimates].sort((a, b) => a.fair_value - b.fair_value);
  const mid = Math.floor(sorted.length / 2);
  const start = Math.max(0, mid - 1);
  return sorted
    .slice(start, start + 3)
    .map((e) => ({ method: e.method, fair_value: e.fair_value }));
}
