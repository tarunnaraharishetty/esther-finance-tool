/** Mock OHLC chart data per (symbol, timeframe).
 *
 * Backend doesn't ship per-symbol price history yet, so the chart
 * series is generated deterministically from a seed derived from
 * the row identity + timeframe — no flicker across rerenders. The
 * shape mirrors what a real API would return so the swap is a
 * one-line change.
 *
 * Phase 3 contract: each point has a wall-clock `t` (ms epoch), a
 * `price` for the line/area, and OHLC fields so a candlestick
 * variant can render the same series. */

export type Timeframe = "1D" | "1W" | "1M" | "1Y";

export interface ChartPoint {
  t: number;
  price: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface TimeframeSpec {
  points: number;
  /** Wall-clock window in ms between points. */
  step: number;
  /** Per-step drift amplitude as a fraction of price. */
  drift: number;
  /** Per-step noise amplitude as a fraction of price. */
  noise: number;
}

const SPECS: Record<Timeframe, TimeframeSpec> = {
  "1D": { points: 78, step: 5 * 60_000, drift: 0.0006, noise: 0.0035 },
  "1W": { points: 84, step: 2 * 60 * 60_000, drift: 0.0008, noise: 0.0055 },
  "1M": { points: 90, step: 8 * 60 * 60_000, drift: 0.001, noise: 0.0075 },
  "1Y": { points: 120, step: 3 * 24 * 60 * 60_000, drift: 0.0014, noise: 0.012 },
};

function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function chartDataFor(
  symbol: string,
  timeframe: Timeframe,
  action: "buy" | "sell" | "hold",
  basePrice: number,
  now: number = Date.now(),
): ChartPoint[] {
  const spec = SPECS[timeframe];
  const rand = mulberry32(hashStr(`${symbol}|${timeframe}|${action}`));
  const driftSign = action === "buy" ? 1 : action === "sell" ? -1 : 0;

  // Anchor the *last* point near the supplied basePrice so the chart
  // visually matches the row's last_price. Work backward to the start.
  const out: ChartPoint[] = [];
  let close = basePrice / (1 + driftSign * spec.drift * spec.points * 0.5);
  for (let i = 0; i < spec.points; i++) {
    const open = close;
    const step =
      (rand() - 0.5) * 2 * spec.noise * basePrice +
      driftSign * spec.drift * basePrice;
    close = Math.max(0.01, open + step);
    const wickHi = Math.abs(rand() - 0.5) * spec.noise * basePrice;
    const wickLo = Math.abs(rand() - 0.5) * spec.noise * basePrice;
    const high = Math.max(open, close) + wickHi;
    const low = Math.max(0.01, Math.min(open, close) - wickLo);
    out.push({
      t: now - (spec.points - 1 - i) * spec.step,
      price: close,
      open,
      high,
      low,
      close,
    });
  }
  return out;
}

export function seriesDelta(points: ChartPoint[]): {
  abs: number;
  pct: number;
} {
  if (points.length < 2) return { abs: 0, pct: 0 };
  const first = points[0].price;
  const last = points[points.length - 1].price;
  return {
    abs: last - first,
    pct: first === 0 ? 0 : (last - first) / first,
  };
}

export function formatChartTime(t: number, timeframe: Timeframe): string {
  const d = new Date(t);
  if (timeframe === "1D") {
    return d.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  }
  if (timeframe === "1W") {
    return d.toLocaleDateString("en-US", {
      weekday: "short",
      hour: "2-digit",
      hour12: false,
    });
  }
  if (timeframe === "1M") {
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  }
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

export const TIMEFRAMES: Timeframe[] = ["1D", "1W", "1M", "1Y"];
