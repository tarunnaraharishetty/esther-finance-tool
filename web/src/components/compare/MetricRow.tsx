import { ArrowDown, ArrowUp, Equal, Minus } from "lucide-react";
import {
  formatMetricValue,
  type MetricComparison,
  winnerTone,
} from "@/lib/compare";
import { cn } from "@/lib/utils";

interface Props {
  metric: MetricComparison;
}

/**
 * One row in a comparison section. Shows the left value, a winner
 * chip, and the right value with directional cues so the trader
 * can scan top-to-bottom and read "this side is winning here" with
 * zero cognitive load.
 *
 * The direction icon (↑ higher_better / ↓ lower_better) lives next
 * to the label rather than the value because it's a property of the
 * metric itself, not of either side — moving it would make the row
 * read like the chip is the verdict on the *direction*, which it
 * isn't.
 */
export function MetricRow({ metric }: Props) {
  const leftTone = winnerTone("left", metric.winner);
  const rightTone = winnerTone("right", metric.winner);
  const isTie = metric.winner === "tie";
  const isNA = metric.winner === "n/a";

  return (
    <div
      data-testid={`metric-row-${metric.metric}`}
      data-winner={metric.winner}
      className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 border-t border-border/30 px-3 py-2.5 first:border-t-0"
    >
      <ValueCell value={metric.left_value} percent={metric.percent} tone={leftTone} />
      <div className="flex flex-col items-center gap-1 text-center">
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          {metric.label}
        </span>
        <WinnerChip winner={metric.winner} />
        <DirectionBadge direction={metric.direction} muted={isTie || isNA} />
      </div>
      <ValueCell
        value={metric.right_value}
        percent={metric.percent}
        tone={rightTone}
        align="right"
      />
    </div>
  );
}

function ValueCell({
  value,
  percent,
  tone,
  align = "left",
}: {
  value: number | null;
  percent: boolean;
  tone: "bull" | "bear" | "warn" | "muted";
  align?: "left" | "right";
}) {
  const text = formatMetricValue(value, percent);
  return (
    <div
      className={cn(
        "font-mono text-base font-semibold tabular-nums",
        align === "right" && "text-right",
        tone === "bull" && "text-bull",
        tone === "bear" && "text-muted-foreground",
        tone === "warn" && "text-warn",
        tone === "muted" && "text-muted-foreground/60",
      )}
    >
      {text}
    </div>
  );
}

function WinnerChip({ winner }: { winner: MetricComparison["winner"] }) {
  if (winner === "n/a") {
    return (
      <span
        data-testid="winner-chip"
        data-state="na"
        className="inline-flex items-center gap-1 rounded-full border border-border/40 bg-card/40 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground/70"
      >
        <Minus className="h-2.5 w-2.5" />
        n/a
      </span>
    );
  }
  if (winner === "tie") {
    return (
      <span
        data-testid="winner-chip"
        data-state="tie"
        className="inline-flex items-center gap-1 rounded-full border border-warn/40 bg-warn/15 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-warn"
      >
        <Equal className="h-2.5 w-2.5" />
        tie
      </span>
    );
  }
  // Left or right winner: arrow points toward the winning side.
  const Icon = winner === "left" ? ArrowUp : ArrowDown;
  return (
    <span
      data-testid="winner-chip"
      data-state={winner}
      className="inline-flex items-center gap-1 rounded-full border border-bull/40 bg-bull/15 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-bull"
    >
      <Icon className="h-2.5 w-2.5" />
      {winner}
    </span>
  );
}

function DirectionBadge({
  direction,
  muted,
}: {
  direction: MetricComparison["direction"];
  muted: boolean;
}) {
  const label =
    direction === "higher_better" ? "higher · better" : "lower · better";
  return (
    <span
      data-testid="direction-badge"
      className={cn(
        "font-mono text-[8px] uppercase tracking-wider",
        muted ? "text-muted-foreground/50" : "text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}
