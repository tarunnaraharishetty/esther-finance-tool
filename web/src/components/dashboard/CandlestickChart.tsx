import { useMemo, useRef, useState } from "react";
import {
  formatChartTime,
  type ChartPoint,
  type Timeframe,
} from "@/lib/chartData";
import { fmtPrice } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Props {
  data: ChartPoint[];
  timeframe: Timeframe;
  height: number;
}

interface HoverState {
  index: number;
  x: number;
  y: number;
}

/**
 * Dependency-free SVG candlestick. Hand-rolled because Recharts has
 * no native candlestick and pulling in another chart lib for one
 * view would bloat the bundle. The TradingView embed is the
 * premium option for traders who want full TA tooling; this is the
 * fast, themed, on-brand variant.
 *
 * Bigger / wider candles than the previous version: body width is
 * now ~70% of the per-bar slot (was ~55%), wicks are thicker, axis
 * labels are rendered below the chart, and the OHLC HUD floats
 * inside a `surface-premium` panel on hover.
 */
export function CandlestickChart({ data, timeframe, height }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const layout = useMemo(() => computeLayout(data, height), [data, height]);

  function onMove(e: React.PointerEvent<HTMLDivElement>): void {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const idx = Math.max(
      0,
      Math.min(
        data.length - 1,
        Math.round((x / rect.width) * (data.length - 1)),
      ),
    );
    setHover({ index: idx, x, y });
  }

  const W = 100;
  const H = height;
  const hovered = hover ? data[hover.index] : null;

  return (
    <div
      ref={containerRef}
      className="relative"
      style={{ height }}
      onPointerMove={onMove}
      onPointerLeave={() => setHover(null)}
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        width="100%"
        height="100%"
        className="overflow-visible"
      >
        {/* Y gridlines at quartiles. */}
        {[0.2, 0.4, 0.6, 0.8].map((p) => (
          <line
            key={p}
            x1={0}
            x2={W}
            y1={H * p}
            y2={H * p}
            stroke="hsl(var(--border))"
            strokeOpacity={0.35}
            strokeDasharray="2 3"
          />
        ))}

        {/* Price labels at gridlines. */}
        {[0.2, 0.5, 0.8].map((p) => {
          const value =
            layout.min + (1 - p / (height / height)) * (layout.max - layout.min);
          return (
            <text
              key={`label-${p}`}
              x={W - 0.5}
              y={H * p - 0.5}
              fontSize={2.4}
              textAnchor="end"
              fill="hsl(var(--muted-foreground))"
              opacity={0.6}
            >
              ${fmtPrice(value)}
            </text>
          );
        })}

        {layout.candles.map((c, i) => {
          const isBull = data[i].close >= data[i].open;
          const color = isBull ? "hsl(var(--bull))" : "hsl(var(--bear))";
          const dim = hover && hover.index !== i ? 0.45 : 1;
          return (
            <g key={i} opacity={dim}>
              <line
                x1={c.x}
                x2={c.x}
                y1={c.highY}
                y2={c.lowY}
                stroke={color}
                strokeWidth={0.6}
                vectorEffect="non-scaling-stroke"
              />
              <rect
                x={c.x - c.bodyW / 2}
                y={c.bodyY}
                width={c.bodyW}
                height={Math.max(0.8, c.bodyH)}
                fill={color}
                fillOpacity={isBull ? 0.9 : 0.95}
                stroke={color}
                strokeWidth={0.4}
                vectorEffect="non-scaling-stroke"
                rx={0.35}
              />
            </g>
          );
        })}

        {/* Volume strip at the bottom. */}
        {layout.volumeBars.map((v, i) => {
          const isBull = data[i].close >= data[i].open;
          return (
            <rect
              key={`v-${i}`}
              x={v.x - layout.candles[i].bodyW / 2}
              y={H - v.h}
              width={layout.candles[i].bodyW}
              height={v.h}
              fill={isBull ? "hsl(var(--bull))" : "hsl(var(--bear))"}
              fillOpacity={0.22}
            />
          );
        })}

        {/* Crosshair */}
        {hover && hovered && (
          <g>
            <line
              x1={layout.candles[hover.index].x}
              x2={layout.candles[hover.index].x}
              y1={0}
              y2={H}
              stroke="hsl(var(--muted-foreground))"
              strokeOpacity={0.55}
              strokeDasharray="2 3"
              vectorEffect="non-scaling-stroke"
            />
          </g>
        )}
      </svg>

      {/* HUD pinned top-left while hovering. */}
      {hover && hovered && (
        <div className="pointer-events-none absolute left-3 top-3 rounded-lg border border-border/60 bg-card/95 px-3.5 py-2.5 text-[11px] shadow-lg backdrop-blur animate-fade-in-fast">
          <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            {formatChartTime(hovered.t, timeframe)}
          </div>
          <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 font-mono">
            <span className="text-muted-foreground">Open</span>
            <span className="text-right tabular">${fmtPrice(hovered.open)}</span>
            <span className="text-muted-foreground">High</span>
            <span className="text-right tabular text-bull">
              ${fmtPrice(hovered.high)}
            </span>
            <span className="text-muted-foreground">Low</span>
            <span className="text-right tabular text-bear">
              ${fmtPrice(hovered.low)}
            </span>
            <span className="text-muted-foreground">Close</span>
            <span
              className={cn(
                "text-right tabular font-semibold",
                hovered.close >= hovered.open ? "text-bull" : "text-bear",
              )}
            >
              ${fmtPrice(hovered.close)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

interface Candle {
  x: number;
  bodyY: number;
  bodyH: number;
  bodyW: number;
  highY: number;
  lowY: number;
}

interface VolumeBar {
  x: number;
  h: number;
}

function computeLayout(
  data: ChartPoint[],
  height: number,
): { candles: Candle[]; volumeBars: VolumeBar[]; min: number; max: number } {
  if (data.length === 0) {
    return { candles: [], volumeBars: [], min: 0, max: 0 };
  }

  const lows = data.map((d) => d.low);
  const highs = data.map((d) => d.high);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const range = max - min || 1;

  const W = 100;
  const chartH = height * 0.82;
  const volH = height * 0.16;
  const pad = 1.5;
  const step = W / data.length;
  // Wider candle body: 70% of the slot. Previously 55%.
  const bodyW = Math.max(0.7, step * 0.7);

  const yFor = (v: number): number => {
    const t = (v - min) / range;
    return pad + (1 - t) * (chartH - pad * 2);
  };

  const magnitudes = data.map((d) => Math.abs(d.close - d.open));
  const magMax = Math.max(...magnitudes, 1e-6);

  const candles: Candle[] = [];
  const volumeBars: VolumeBar[] = [];

  for (let i = 0; i < data.length; i++) {
    const d = data[i];
    const x = step * (i + 0.5);
    const bodyTop = yFor(Math.max(d.open, d.close));
    const bodyBottom = yFor(Math.min(d.open, d.close));
    candles.push({
      x,
      bodyY: bodyTop,
      bodyH: bodyBottom - bodyTop,
      bodyW,
      highY: yFor(d.high),
      lowY: yFor(d.low),
    });
    volumeBars.push({
      x,
      h: (magnitudes[i] / magMax) * (volH * 0.9),
    });
  }
  return { candles, volumeBars, min, max };
}
