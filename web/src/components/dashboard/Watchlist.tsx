import { Activity, ArrowDown, ArrowUp, Minus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { MiniSpark } from "@/components/ui/MiniSpark";
import {
  actionTone,
  fmtConfidence,
  fmtPrice,
  tierLabel,
} from "@/lib/format";
import { sparklineFor } from "@/lib/sparkline";
import { cn } from "@/lib/utils";
import type { RecommendationRow } from "@/lib/types";

interface Props {
  rows: RecommendationRow[];
  activeSymbol: string | null;
  onSelect: (symbol: string) => void;
}

export function Watchlist({ rows, activeSymbol, onSelect }: Props) {
  if (rows.length === 0) {
    return (
      <div className="grid place-items-center rounded-lg border border-dashed border-border/60 bg-card/40 p-8 text-sm text-muted-foreground">
        Watchlist is empty — add a symbol from the CLI to begin.
      </div>
    );
  }

  return (
    <div className="overflow-hidden">
      <div className="grid grid-cols-[3.5rem_3.5rem_1fr_5rem_4.5rem] items-center gap-3 border-b border-border/40 px-3 pb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground md:grid-cols-[4rem_4rem_1fr_5rem_5rem_3rem]">
        <span>Symbol</span>
        <span>Action</span>
        <span className="hidden md:inline">Trend</span>
        <span className="text-right">Last</span>
        <span className="text-right">Conf</span>
        <span className="hidden md:inline text-right">News</span>
      </div>

      <ul className="space-y-1 pt-2">
        {rows.map((row) => (
          <WatchlistRow
            key={row.symbol}
            row={row}
            active={row.symbol === activeSymbol}
            onSelect={onSelect}
          />
        ))}
      </ul>
    </div>
  );
}

function WatchlistRow({
  row,
  active,
  onSelect,
}: {
  row: RecommendationRow;
  active: boolean;
  onSelect: (s: string) => void;
}) {
  const tone = actionTone(row.action);
  const TrendIcon =
    row.action === "buy" ? ArrowUp : row.action === "sell" ? ArrowDown : Minus;
  // Sentiment "heat": 5-band scale used for the colored dot intensity.
  const heat = sentimentHeat(row.sentiment_score);

  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(row.symbol)}
        aria-pressed={active}
        className={cn(
          "relative grid w-full grid-cols-[3.5rem_3.5rem_1fr_5rem_4.5rem] items-center gap-3 rounded-lg border px-3 py-2 text-left transition-all duration-150 md:grid-cols-[4rem_4rem_1fr_5rem_5rem_3rem]",
          active
            ? "border-primary/50 bg-primary/10 shadow-glow"
            : "border-transparent hover:border-border/60 hover:bg-card/60 hover:-translate-y-px",
        )}
      >
        {active && (
          <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2 rounded-r-full bg-primary" />
        )}

        <div className="min-w-0">
          <div className="truncate font-mono text-sm font-semibold">
            {row.symbol}
          </div>
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
            <HeatDot heat={heat} />
            <span className="font-mono">{row.tier.split("_")[0]}</span>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <Badge variant={tone} className="px-1.5">
            <TrendIcon className="h-2.5 w-2.5" />
            {row.action}
          </Badge>
        </div>

        <div className="hidden md:block">
          <MiniSpark
            data={sparklineFor(row.symbol, row.action, row.last_price)}
            tone={tone}
            height={28}
          />
        </div>

        <div className="text-right font-mono text-sm tabular">
          ${fmtPrice(row.last_price)}
        </div>

        <div className="text-right">
          <div className="font-mono text-xs tabular">
            {fmtConfidence(row.confidence)}
          </div>
          <div className="mt-0.5 h-1 w-full overflow-hidden rounded-full bg-muted/60">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-300",
                tone === "bull" && "bg-gradient-to-r from-bull/70 to-bull",
                tone === "bear" && "bg-gradient-to-r from-bear/70 to-bear",
                tone === "warn" && "bg-gradient-to-r from-warn/70 to-warn",
              )}
              style={{ width: `${Math.round(row.confidence * 100)}%` }}
            />
          </div>
        </div>

        <div
          className="hidden items-center justify-end gap-1 text-right font-mono text-[11px] text-muted-foreground md:flex"
          title={`${row.num_news_articles} headlines · ${tierLabel(row.tier)}`}
        >
          <Activity className="h-3 w-3" />
          {row.num_news_articles}
        </div>
      </button>
    </li>
  );
}

type Heat = "+2" | "+1" | "0" | "-1" | "-2";

function sentimentHeat(score: number): Heat {
  if (score > 0.5) return "+2";
  if (score > 0.15) return "+1";
  if (score < -0.5) return "-2";
  if (score < -0.15) return "-1";
  return "0";
}

function HeatDot({ heat }: { heat: Heat }) {
  // Concentric ring renders the heat level; outer ring fades on neutral.
  const colorClass =
    heat === "+2" || heat === "+1"
      ? "bg-bull"
      : heat === "-2" || heat === "-1"
        ? "bg-bear"
        : "bg-muted-foreground/40";
  const ringClass =
    heat === "+2" || heat === "-2"
      ? "ring-2"
      : heat === "+1" || heat === "-1"
        ? "ring-1"
        : "";
  const ringColorClass =
    heat === "+2" || heat === "+1"
      ? "ring-bull/30"
      : heat === "-2" || heat === "-1"
        ? "ring-bear/30"
        : "";
  return (
    <span
      title={`sentiment ${heat}`}
      className={cn(
        "h-1.5 w-1.5 rounded-full",
        colorClass,
        ringClass,
        ringColorClass,
      )}
    />
  );
}
