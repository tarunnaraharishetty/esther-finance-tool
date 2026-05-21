import { useMemo } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { MiniSpark } from "@/components/ui/MiniSpark";
import { fmtPct, fmtPrice } from "@/lib/format";
import { sparklineFor, sparklineDelta } from "@/lib/sparkline";
import { cn } from "@/lib/utils";

interface IndexTile {
  symbol: string;
  label: string;
  basePrice: number;
}

// Mock prices for the strip. These look like reasonable mid-2026
// values for shape; real ones would come from a market-data API.
// Anchored so the seeded sparklines drift plausibly off these.
const INDICES: IndexTile[] = [
  { symbol: "SPY", label: "S&P 500", basePrice: 612.4 },
  { symbol: "QQQ", label: "Nasdaq 100", basePrice: 524.8 },
  { symbol: "DIA", label: "Dow 30", basePrice: 444.9 },
  { symbol: "IWM", label: "Russell 2K", basePrice: 235.6 },
  { symbol: "VIX", label: "Volatility", basePrice: 14.6 },
  { symbol: "BTCUSD", label: "Bitcoin", basePrice: 96420 },
  { symbol: "ETHUSD", label: "Ethereum", basePrice: 3540 },
  { symbol: "GLD", label: "Gold", basePrice: 248.1 },
];

/**
 * Top-of-app market overview strip — the Bloomberg-terminal moment.
 *
 * Renders eight indices/crypto as a horizontal scrollable strip
 * directly below the topnav. Each tile shows symbol + label + price
 * + signed change + sparkline. The series are deterministic seeded
 * from `(symbol, faux-action)` so they don't churn between renders
 * but still feel alive on hard reload (different seed window).
 */
export function MarketTickerStrip() {
  return (
    <div className="relative shrink-0 border-b border-border/40 bg-background/40 backdrop-blur-xl">
      {/* Edge fades so the scroll-strip feels embedded, not abrupt. */}
      <div
        aria-hidden
        className="pointer-events-none absolute left-0 top-0 z-10 h-full w-12 bg-gradient-to-r from-background to-transparent"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute right-0 top-0 z-10 h-full w-12 bg-gradient-to-l from-background to-transparent"
      />
      <ul
        className="flex gap-0 overflow-x-auto px-2 py-2"
        style={{ scrollbarWidth: "none" }}
      >
        {INDICES.map((idx) => (
          <li key={idx.symbol} className="shrink-0">
            <TickerTile tile={idx} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function TickerTile({ tile }: { tile: IndexTile }) {
  // Deterministic faux change. Some up, some down — VIX inversely
  // correlated to the rest so the strip reads diversified.
  const action: "buy" | "sell" | "hold" =
    tile.symbol === "VIX"
      ? "sell"
      : tile.symbol.endsWith("USD")
        ? "buy"
        : tile.symbol === "GLD"
          ? "hold"
          : "buy";

  const data = useMemo(
    () => sparklineFor(tile.symbol, action, tile.basePrice),
    [tile.symbol, tile.basePrice, action],
  );
  const delta = sparklineDelta(data);
  const tone =
    delta > 0.0005 ? "bull" : delta < -0.0005 ? "bear" : "warn";
  const Icon = tone === "bull" ? TrendingUp : TrendingDown;
  const last = data[data.length - 1]?.v ?? tile.basePrice;

  return (
    <div
      className={cn(
        "group flex min-w-[180px] items-center gap-3 border-r border-border/30 px-4 py-1.5",
        "transition-colors hover:bg-card/40",
      )}
    >
      <div className="min-w-0">
        <div className="flex items-baseline gap-1.5">
          <span className="font-mono text-[11px] font-semibold tracking-tight">
            {tile.symbol}
          </span>
          <span className="truncate font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
            {tile.label}
          </span>
        </div>
        <div className="mt-0.5 flex items-baseline gap-1.5">
          <span className="font-mono text-xs font-semibold tabular">
            {last >= 1000 ? last.toLocaleString("en-US", { maximumFractionDigits: 0 }) : `$${fmtPrice(last)}`}
          </span>
          <span
            className={cn(
              "inline-flex items-center gap-0.5 font-mono text-[10px] tabular",
              tone === "bull" && "text-bull",
              tone === "bear" && "text-bear",
              tone === "warn" && "text-warn",
            )}
          >
            <Icon className="h-2.5 w-2.5" />
            {fmtPct(delta, 2)}
          </span>
        </div>
      </div>
      <div className="hidden h-7 w-16 shrink-0 sm:block">
        <MiniSpark data={data} tone={tone} height={28} />
      </div>
    </div>
  );
}
