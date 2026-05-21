import { Suspense, lazy, useState } from "react";
import { Activity, Newspaper, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ConfidenceMeter } from "./ConfidenceMeter";
import { WhyMoving } from "./WhyMoving";
import { generateSymbolSummary } from "@/lib/aiSummary";
import type { Timeframe } from "@/lib/chartData";
import { actionTone, fmtPrice, tierLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  symbol: string | null;
  initialTimeframe?: Timeframe;
}

const PriceChart = lazy(() =>
  import("./PriceChart").then((m) => ({ default: m.PriceChart })),
);

export function SymbolDetail({ snapshot, symbol, initialTimeframe = "1M" }: Props) {
  const [timeframe, setTimeframe] = useState<Timeframe>(initialTimeframe);

  if (!symbol) {
    return (
      <div className="grid h-full place-items-center text-sm text-muted-foreground">
        Select a symbol from the watchlist.
      </div>
    );
  }
  const row = snapshot.rows.find((r) => r.symbol === symbol);
  if (!row) {
    return (
      <div className="grid h-full place-items-center text-sm text-muted-foreground">
        {symbol} is not on the watchlist.
      </div>
    );
  }

  const tone = actionTone(row.action);

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-display text-2xl font-semibold tracking-tightest md:text-[28px]">
              {row.symbol}
            </h2>
            <span className="font-mono text-base font-semibold tabular text-muted-foreground">
              ${fmtPrice(row.last_price)}
            </span>
            <Badge variant={tone}>{row.action.toUpperCase()}</Badge>
            <Badge variant="outline">{tierLabel(row.tier)}</Badge>
          </div>
          <div className="mt-1 flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            <Activity className="h-3 w-3" />
            {row.signal_quality} · {row.stability}
          </div>
        </div>
        <div className="min-w-[220px] flex-1 sm:max-w-xs">
          <ConfidenceMeter
            value={row.confidence}
            tone={tone}
            label="Confidence"
            size="md"
          />
        </div>
      </header>

      <Suspense fallback={<ChartSkeleton />}>
        <PriceChart
          symbol={row.symbol}
          action={row.action}
          lastPrice={row.last_price}
          timeframe={timeframe}
          onTimeframeChange={setTimeframe}
        />
      </Suspense>

      <AiRationale row={row} />

      <WhyMoving row={row} />

      <StatsGrid row={row} />

      <Headlines row={row} />
    </div>
  );
}

function ChartSkeleton() {
  return (
    <div className="rounded-xl border border-border/40 bg-card/40 p-4">
      <div className="mb-3 flex items-center justify-between">
        <Skeleton className="h-9 w-40" />
        <div className="flex gap-2">
          <Skeleton className="h-7 w-24" />
          <Skeleton className="h-7 w-32" />
        </div>
      </div>
      <Skeleton className="h-72 w-full opacity-60" />
    </div>
  );
}

function AiRationale({ row }: { row: RecommendationRow }) {
  return (
    <div className="surface-premium relative overflow-hidden p-5">
      <div
        aria-hidden
        className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-primary/15 blur-2xl"
      />
      <div className="relative">
        <div className="mb-2 flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-primary" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            AI Read
          </span>
        </div>
        <p className="text-sm leading-relaxed">{generateSymbolSummary(row)}</p>
        {row.reasoning && (
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            {row.reasoning}
          </p>
        )}
      </div>
    </div>
  );
}

function StatsGrid({ row }: { row: RecommendationRow }) {
  const rsiTone =
    row.rsi > 70 ? "bear" : row.rsi < 30 ? "bull" : "warn";
  const macdTone = row.macd > 0 ? "bull" : row.macd < 0 ? "bear" : "warn";
  const sentTone =
    row.sentiment_score > 0.15
      ? "bull"
      : row.sentiment_score < -0.15
        ? "bear"
        : "warn";
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      <Stat label="Last" value={`$${fmtPrice(row.last_price)}`} />
      <Stat
        label="RSI"
        value={row.rsi.toFixed(1)}
        tone={rsiTone}
        hint={row.rsi > 70 ? "overbought" : row.rsi < 30 ? "oversold" : "neutral"}
      />
      <Stat label="MACD" value={row.macd.toFixed(2)} tone={macdTone} />
      <Stat label="Bollinger" value={row.bollinger.toFixed(2)} />
      <Stat
        label="Sentiment"
        value={row.sentiment_score.toFixed(2)}
        tone={sentTone}
      />
      <Stat label="Headlines" value={row.num_news_articles.toString()} />
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
  hint,
}: {
  label: string;
  value: string;
  tone?: "bull" | "bear" | "warn";
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-border/40 bg-card/40 px-3 py-2.5 transition-colors hover:border-border/60 hover:bg-card/60">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "mt-0.5 font-mono text-sm font-semibold tabular",
          tone === "bull" && "text-bull",
          tone === "bear" && "text-bear",
          tone === "warn" && "text-warn",
        )}
      >
        {value}
      </div>
      {hint && (
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          {hint}
        </div>
      )}
    </div>
  );
}

function Headlines({ row }: { row: RecommendationRow }) {
  if (row.headlines.length === 0) return null;
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Newspaper className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Recent Headlines
        </span>
      </div>
      <ul className="space-y-2">
        {row.headlines.slice(0, 5).map((h, i) => {
          const url = h.match(/https?:\/\/\S+/)?.[0];
          const href =
            url ?? `https://news.google.com/search?q=${encodeURIComponent(h)}`;
          return (
            <li key={i}>
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-md border border-border/40 bg-card/30 px-3 py-2 text-sm transition-all hover:-translate-y-px hover:border-border hover:bg-card"
              >
                {h}
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
