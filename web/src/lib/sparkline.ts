/** Deterministic mock OHLC-ish series for sparklines.
 *
 * The backend snapshot doesn't ship per-symbol price history yet, so
 * the sparkline shown on watchlist rows / mover cards is generated
 * from a stable seed derived from the row identity. It nudges
 * upward when the action is bullish, downward when bearish, and
 * stays jittery for hold. The series is stable across rerenders
 * because the seed is deterministic — no flicker.
 *
 * Phase 3 will swap this for real OHLCV once the API exposes it. */

const POINTS = 24;

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

export interface SparkPoint {
  i: number;
  v: number;
}

export function sparklineFor(
  symbol: string,
  action: "buy" | "sell" | "hold",
  basePrice: number,
): SparkPoint[] {
  const rand = mulberry32(hashStr(`${symbol}|${action}`));
  const drift = action === "buy" ? 0.012 : action === "sell" ? -0.012 : 0;
  const noise = action === "hold" ? 0.014 : 0.009;

  const series: SparkPoint[] = [];
  let v = basePrice * (1 - drift * POINTS * 0.5);
  for (let i = 0; i < POINTS; i++) {
    const step = (rand() - 0.5) * 2 * noise * basePrice + drift * basePrice;
    v += step;
    series.push({ i, v: Math.max(0.01, v) });
  }
  return series;
}

export function sparklineDelta(series: SparkPoint[]): number {
  if (series.length < 2) return 0;
  const first = series[0].v;
  const last = series[series.length - 1].v;
  if (first === 0) return 0;
  return (last - first) / first;
}
