/** Shared formatters so the UI never re-derives currency / pct rules. */

export function fmtPrice(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export function fmtPct(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${(n * 100).toFixed(digits)}%`;
}

export function fmtConfidence(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return `${Math.round(n * 100)}%`;
}

export function tierLabel(tier: string): string {
  return tier.replace(/_/g, " ").toUpperCase();
}

export function actionTone(
  action: string,
): "bull" | "bear" | "warn" {
  if (action === "buy") return "bull";
  if (action === "sell") return "bear";
  return "warn";
}

export function tierTone(tier: string): "bull" | "bear" | "warn" {
  if (tier === "strong_buy" || tier === "buy") return "bull";
  if (tier === "strong_sell" || tier === "sell") return "bear";
  return "warn";
}
