import { useMemo } from "react";
import type { SparkPoint } from "@/lib/sparkline";

interface Props {
  data: SparkPoint[];
  tone: "bull" | "bear" | "warn";
  height?: number;
  className?: string;
}

/**
 * Dependency-free SVG sparkline. Used on dense row grids
 * (Watchlist, Movers, TopPicks) where pulling in Recharts would
 * be overkill and would bloat the initial bundle by ~150 KB gz.
 * The full Recharts `PriceChart` is reserved for the Symbol
 * Detail / Charts page where axes and a hover tooltip are worth
 * the weight.
 */
export function MiniSpark({ data, tone, height = 36, className }: Props) {
  const path = useMemo(() => buildPath(data, height), [data, height]);
  const colorVar =
    tone === "bull"
      ? "var(--bull)"
      : tone === "bear"
        ? "var(--bear)"
        : "var(--warn)";

  if (!path) return <div style={{ height }} aria-hidden />;

  const gradientId = `mini-spark-${Math.random().toString(36).slice(2, 9)}`;

  return (
    <svg
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      width="100%"
      height={height}
      className={className}
      aria-hidden
    >
      <defs>
        <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={`hsl(${colorVar})`} stopOpacity={0.4} />
          <stop offset="100%" stopColor={`hsl(${colorVar})`} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={path.fill} fill={`url(#${gradientId})`} />
      <path
        d={path.line}
        fill="none"
        stroke={`hsl(${colorVar})`}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function buildPath(
  data: SparkPoint[],
  height: number,
): { line: string; fill: string } | null {
  if (data.length < 2) return null;
  const vs = data.map((p) => p.v);
  const min = Math.min(...vs);
  const max = Math.max(...vs);
  const range = max - min || 1;
  const w = 100;
  const h = height;
  const pad = 2;
  const innerH = h - pad * 2;
  const step = w / (data.length - 1);

  const pts = data.map((p, i) => {
    const x = i * step;
    const y = pad + (1 - (p.v - min) / range) * innerH;
    return { x, y };
  });

  const line = pts
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
    .join(" ");
  const fill = `${line} L${w.toFixed(2)},${h} L0,${h} Z`;
  return { line, fill };
}
