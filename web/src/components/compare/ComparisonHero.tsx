import { GitCompareArrows, Trophy } from "lucide-react";
import type { ComparisonView } from "@/lib/compare";
import { TrustScoreBadge } from "@/components/trust/TrustScoreBadge";
import { cn } from "@/lib/utils";

interface Props {
  view: ComparisonView;
}

/**
 * Top-of-page banner for the comparison view.
 *
 * Layout: left symbol header · overall verdict pill · right symbol header.
 * Below that, the headline metric strip (trust grade, last price,
 * overall analyzer score). The trust score badges link to the same
 * detailed breakdown that the analyzer page renders, so a user who
 * wants to dig into the trust composition for either symbol can do
 * it inline.
 */
export function ComparisonHero({ view }: Props) {
  return (
    <section className="surface-premium relative overflow-hidden p-5 md:p-6">
      <div
        aria-hidden
        className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-primary/15 blur-3xl"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-24 -left-12 h-48 w-48 rounded-full bg-accent/15 blur-3xl"
      />
      <div className="relative">
        <div className="flex items-center gap-2">
          <GitCompareArrows className="h-3.5 w-3.5 text-primary" />
          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">
            Side-by-Side Comparison
          </span>
        </div>

        <div className="mt-3 grid items-center gap-4 md:grid-cols-[1fr_auto_1fr]">
          <SymbolHero
            symbol={view.left_symbol}
            side="left"
            report={view.left_report}
            winner={view.overall_winner}
          />
          <OverallVerdict winner={view.overall_winner} />
          <SymbolHero
            symbol={view.right_symbol}
            side="right"
            report={view.right_report}
            winner={view.overall_winner}
            align="right"
          />
        </div>

        <div
          data-testid="compare-headline-strip"
          className="mt-5 grid grid-cols-3 gap-3 rounded-md border border-border/40 bg-card/40 p-3"
        >
          {view.headline.map((m) => (
            <HeadlineCell key={m.label} metric={m} />
          ))}
        </div>
      </div>
    </section>
  );
}

function SymbolHero({
  symbol,
  side,
  report,
  winner,
  align,
}: {
  symbol: string;
  side: "left" | "right";
  report: ComparisonView["left_report"];
  winner: ComparisonView["overall_winner"];
  align?: "right";
}) {
  const isWinner = winner === side;
  return (
    <div
      data-testid={`compare-hero-${side}`}
      className={cn(
        "min-w-0",
        align === "right" && "md:text-right",
      )}
    >
      <h1
        className={cn(
          "font-display text-3xl font-semibold tracking-tightest md:text-4xl",
          isWinner && "text-bull",
        )}
      >
        {symbol}
      </h1>
      <div
        className={cn(
          "mt-2 flex flex-wrap items-center gap-2",
          align === "right" && "md:justify-end",
        )}
      >
        {report ? (
          <TrustScoreBadge trust={report.trust_score} variant="inline" />
        ) : (
          <span className="rounded-md border border-border/40 bg-card/40 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            No trust score
          </span>
        )}
      </div>
    </div>
  );
}

function OverallVerdict({ winner }: { winner: ComparisonView["overall_winner"] }) {
  if (winner === "n/a") {
    return (
      <div className="grid place-items-center">
        <span className="rounded-full border border-border/40 bg-card/40 px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          insufficient data
        </span>
      </div>
    );
  }
  if (winner === "tie") {
    return (
      <div className="grid place-items-center">
        <span className="rounded-full border border-warn/40 bg-warn/15 px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-warn">
          metrics balanced
        </span>
      </div>
    );
  }
  return (
    <div className="grid place-items-center">
      <span
        data-testid="overall-verdict"
        data-winner={winner}
        className="inline-flex items-center gap-1.5 rounded-full border border-bull/40 bg-bull/15 px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-bull"
      >
        <Trophy className="h-3 w-3" />
        {winner} side leads
      </span>
    </div>
  );
}

function HeadlineCell({
  metric,
}: {
  metric: ComparisonView["headline"][number];
}) {
  return (
    <div className="text-center" data-testid={`headline-${metric.label.toLowerCase().replace(/\s+/g, "-")}`}>
      <div className="font-mono text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
        {metric.label}
      </div>
      <div className="mt-1 flex items-center justify-center gap-2">
        <span
          className={cn(
            "font-mono text-base font-semibold tabular-nums",
            metric.winner === "left" && "text-bull",
            metric.winner === "right" && "text-muted-foreground",
            metric.winner === "tie" && "text-warn",
            metric.winner === "n/a" && "text-muted-foreground/60",
          )}
        >
          {metric.left_display}
        </span>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/60">
          vs
        </span>
        <span
          className={cn(
            "font-mono text-base font-semibold tabular-nums",
            metric.winner === "right" && "text-bull",
            metric.winner === "left" && "text-muted-foreground",
            metric.winner === "tie" && "text-warn",
            metric.winner === "n/a" && "text-muted-foreground/60",
          )}
        >
          {metric.right_display}
        </span>
      </div>
    </div>
  );
}
