import { Suspense, lazy, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  CandlestickChart as CandleIcon,
  LineChart as LineIcon,
  AreaChart as AreaIcon,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { CandlestickChart } from "./CandlestickChart";
import {
  TIMEFRAMES,
  chartDataFor,
  formatChartTime,
  seriesDelta,
  type Timeframe,
} from "@/lib/chartData";
import { fmtPct, fmtPrice } from "@/lib/format";
import { cn } from "@/lib/utils";

// TradingView embed is heavy (iframes + tv.js ~75 KB), only load it
// when the trader actually clicks the TV tab.
const TradingViewChart = lazy(() =>
  import("./TradingViewChart").then((m) => ({ default: m.TradingViewChart })),
);

type ChartKind = "area" | "line" | "candle" | "tradingview";

interface Props {
  symbol: string;
  action: "buy" | "sell" | "hold";
  lastPrice: number;
  timeframe: Timeframe;
  onTimeframeChange: (tf: Timeframe) => void;
  height?: number;
  /** Default chart type. Off-watchlist symbols force "tradingview". */
  defaultKind?: ChartKind;
}

export function PriceChart({
  symbol,
  action,
  lastPrice,
  timeframe,
  onTimeframeChange,
  height = 380,
  defaultKind = "area",
}: Props) {
  const [kind, setKind] = useState<ChartKind>(defaultKind);

  const data = useMemo(
    () => chartDataFor(symbol, timeframe, action, lastPrice),
    [symbol, timeframe, action, lastPrice],
  );
  const delta = useMemo(() => seriesDelta(data), [data]);
  const tone =
    delta.pct > 0.0001 ? "bull" : delta.pct < -0.0001 ? "bear" : "warn";

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-baseline gap-3">
            <span className="font-display text-4xl font-semibold tracking-tightest tabular">
              ${fmtPrice(lastPrice)}
            </span>
            <div
              className={cn(
                "inline-flex items-center rounded-full border px-2.5 py-1 font-mono text-xs tabular",
                tone === "bull" && "border-bull/40 bg-bull/10 text-bull",
                tone === "bear" && "border-bear/40 bg-bear/10 text-bear",
                tone === "warn" && "border-warn/40 bg-warn/10 text-warn",
              )}
            >
              <span className="font-semibold">
                {delta.abs >= 0 ? "+" : ""}
                {fmtPrice(delta.abs)}
              </span>
              <span className="mx-1.5 opacity-60">·</span>
              <span>{fmtPct(delta.pct)}</span>
            </div>
          </div>
          <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            {kind === "tradingview"
              ? `${timeframe} · TradingView live`
              : `${timeframe} change · mock data · ${data.length} bars`}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <ChartKindTabs value={kind} onChange={setKind} />
          <TimeframeTabs value={timeframe} onChange={onTimeframeChange} />
        </div>
      </div>

      <div className="rounded-xl border border-border/40 bg-card/30 p-3">
        {kind === "candle" ? (
          <CandlestickChart data={data} timeframe={timeframe} height={height} />
        ) : kind === "tradingview" ? (
          <Suspense
            fallback={<Skeleton className="h-[460px] w-full opacity-60" />}
          >
            <TradingViewChart
              symbol={symbol}
              timeframe={timeframe}
              height={height}
            />
          </Suspense>
        ) : (
          <RechartsView
            kind={kind}
            data={data}
            tone={tone}
            timeframe={timeframe}
            height={height}
            symbol={symbol}
          />
        )}
      </div>
    </div>
  );
}

function RechartsView({
  kind,
  data,
  tone,
  timeframe,
  height,
  symbol,
}: {
  kind: "area" | "line";
  data: ReturnType<typeof chartDataFor>;
  tone: "bull" | "bear" | "warn";
  timeframe: Timeframe;
  height: number;
  symbol: string;
}) {
  const colorVar =
    tone === "bull"
      ? "var(--bull)"
      : tone === "bear"
        ? "var(--bear)"
        : "var(--warn)";
  const gradientId = `pc-${symbol}-${timeframe}-${kind}`;

  const common = (
    <>
      <CartesianGrid
        stroke="hsl(var(--border))"
        strokeOpacity={0.3}
        vertical={false}
      />
      <XAxis
        dataKey="t"
        tickFormatter={(v: number) => formatChartTime(v, timeframe)}
        minTickGap={64}
        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
        tickLine={false}
        axisLine={{ stroke: "hsl(var(--border))", strokeOpacity: 0.4 }}
      />
      <YAxis
        domain={["auto", "auto"]}
        tickFormatter={(v: number) => `$${fmtPrice(v)}`}
        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
        tickLine={false}
        axisLine={false}
        width={64}
      />
      <Tooltip
        content={<ChartTooltip timeframe={timeframe} />}
        cursor={{
          stroke: "hsl(var(--muted-foreground))",
          strokeDasharray: "3 3",
          strokeOpacity: 0.6,
        }}
      />
    </>
  );

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        {kind === "area" ? (
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={`hsl(${colorVar})`} stopOpacity={0.42} />
                <stop offset="60%" stopColor={`hsl(${colorVar})`} stopOpacity={0.12} />
                <stop offset="100%" stopColor={`hsl(${colorVar})`} stopOpacity={0} />
              </linearGradient>
            </defs>
            {common}
            <Area
              type="monotone"
              dataKey="price"
              stroke={`hsl(${colorVar})`}
              strokeWidth={2}
              fill={`url(#${gradientId})`}
              isAnimationActive={false}
              activeDot={{
                r: 4,
                fill: `hsl(${colorVar})`,
                stroke: "hsl(var(--background))",
                strokeWidth: 2,
              }}
            />
          </AreaChart>
        ) : (
          <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            {common}
            <Line
              type="monotone"
              dataKey="price"
              stroke={`hsl(${colorVar})`}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              activeDot={{
                r: 4,
                fill: `hsl(${colorVar})`,
                stroke: "hsl(var(--background))",
                strokeWidth: 2,
              }}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function ChartKindTabs({
  value,
  onChange,
}: {
  value: ChartKind;
  onChange: (k: ChartKind) => void;
}) {
  const items: { kind: ChartKind; icon: typeof CandleIcon; title: string }[] = [
    { kind: "area", icon: AreaIcon, title: "Area" },
    { kind: "line", icon: LineIcon, title: "Line" },
    { kind: "candle", icon: CandleIcon, title: "Candle" },
    { kind: "tradingview", icon: Sparkles, title: "TradingView" },
  ];
  return (
    <div
      role="tablist"
      aria-label="Chart type"
      className="flex items-center gap-0.5 rounded-lg border border-border/60 bg-secondary/30 p-1"
    >
      {items.map(({ kind, icon: Icon, title }) => (
        <Button
          key={kind}
          role="tab"
          aria-selected={value === kind}
          variant={value === kind ? "secondary" : "ghost"}
          size="icon"
          title={title}
          onClick={() => onChange(kind)}
          className={cn("h-7 w-7", value === kind && "bg-card shadow-sm")}
        >
          <Icon className="h-3.5 w-3.5" />
        </Button>
      ))}
    </div>
  );
}

function TimeframeTabs({
  value,
  onChange,
}: {
  value: Timeframe;
  onChange: (tf: Timeframe) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="Timeframe"
      className="flex items-center gap-0.5 rounded-lg border border-border/60 bg-secondary/30 p-1"
    >
      {TIMEFRAMES.map((tf) => (
        <Button
          key={tf}
          role="tab"
          aria-selected={value === tf}
          variant={value === tf ? "secondary" : "ghost"}
          size="sm"
          onClick={() => onChange(tf)}
          className={cn(
            "h-7 px-3 text-xs font-medium",
            value === tf && "bg-card text-foreground shadow-sm",
          )}
        >
          {tf}
        </Button>
      ))}
    </div>
  );
}

interface TooltipPayloadEntry {
  payload?: { t?: number; price?: number; high?: number; low?: number };
}

function ChartTooltip({
  active,
  payload,
  timeframe,
}: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  timeframe: Timeframe;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  if (!p || p.t === undefined || p.price === undefined) return null;
  return (
    <div className="rounded-md border border-border/60 bg-card/95 px-3 py-2 text-xs shadow-lg backdrop-blur animate-fade-in-fast">
      <div className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
        {formatChartTime(p.t, timeframe)}
      </div>
      <div className="mt-0.5 font-mono text-sm font-semibold tabular">
        ${fmtPrice(p.price)}
      </div>
      {p.high !== undefined && p.low !== undefined && (
        <div className="mt-1 font-mono text-[10px] text-muted-foreground">
          H ${fmtPrice(p.high)} · L ${fmtPrice(p.low)}
        </div>
      )}
    </div>
  );
}
