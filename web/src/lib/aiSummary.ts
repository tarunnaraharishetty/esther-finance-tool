/** Deterministic auto-summary generator (NOT an LLM).
 *
 * Templated language that reads natural while staying grounded in
 * what the snapshot actually says. No forecasts, no invented facts —
 * everything mirrors fields already on `DashboardSnapshot`. The
 * `Heuristic` badge in `AiSummary` and `AiPage` surfaces this
 * contract to the user so the "AI" framing is not mistaken for
 * model output.
 */

import type {
  DashboardSnapshot,
  RankedOpportunity,
  RecommendationRow,
} from "./types";

export interface AiSummary {
  headline: string;
  paragraph: string;
  highlights: string[];
}

const SENTIMENT_OPENERS: Record<string, string[]> = {
  bullish: [
    "Risk-on bid is steady",
    "Buyers are in control",
    "The tape is leaning constructive",
  ],
  bearish: [
    "Risk appetite is fading",
    "Sellers have the upper hand",
    "The tape is leaning defensive",
  ],
  mixed: [
    "The tape is split",
    "Conviction is uneven",
    "Crosscurrents on the board",
  ],
  neutral: ["The tape is balanced", "No clear bias on the board"],
};

const REGIME_PHRASES: Record<string, string> = {
  "risk-on": "Regime read: risk-on — leaders are leading.",
  "risk-off": "Regime read: risk-off — defense is bid.",
  mixed: "Regime read: mixed — internals are out of sync.",
  indeterminate: "",
};

function pickStable<T>(arr: T[], seed: number): T {
  if (arr.length === 0) throw new Error("pickStable: empty array");
  const i = Math.abs(seed) % arr.length;
  return arr[i];
}

function snapshotSeed(snap: DashboardSnapshot): number {
  // Hash that changes when meaningful state changes (counts, top
  // opportunity, regime) but is stable across cosmetic re-renders.
  const bull = snap.pulse?.bullish_count ?? 0;
  const bear = snap.pulse?.bearish_count ?? 0;
  const top = snap.ranked_opportunities[0]?.symbol ?? "";
  const regime = snap.pulse_evolution?.regime ?? "";
  const s = `${bull}|${bear}|${top}|${regime}`;
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function generateMarketSummary(snap: DashboardSnapshot): AiSummary {
  const pulse = snap.pulse;
  const rows = snap.rows;
  const top = snap.ranked_opportunities[0];
  const seed = snapshotSeed(snap);

  const sentimentKey = pulse?.sentiment ?? "neutral";
  const opener = pickStable(
    SENTIMENT_OPENERS[sentimentKey] ?? SENTIMENT_OPENERS.neutral,
    seed,
  );

  const bullCount =
    pulse?.bullish_count ?? rows.filter((r) => r.action === "buy").length;
  const bearCount =
    pulse?.bearish_count ?? rows.filter((r) => r.action === "sell").length;
  const breadth = pulse?.momentum_breadth ?? 0;
  const breadthLine =
    breadth >= 0.6
      ? "with broad momentum confirming the move"
      : breadth >= 0.3
        ? "though breadth remains uneven"
        : "with narrow participation under the surface";

  const conviction = pulse?.conviction;
  const convictionTail =
    conviction === "strong"
      ? "Conviction reads strong — these signals are not weakly held."
      : conviction === "weak"
        ? "Conviction reads weak — these signals could fade on the next print."
        : "";

  const leader =
    top !== undefined
      ? `${top.symbol} leads the board with a ${tierLower(top.tier)} read — driven by ${shortRationale(top)}.`
      : null;

  const regimeKey = snap.pulse_evolution?.regime ?? "indeterminate";
  const regimeLine = REGIME_PHRASES[regimeKey] ?? "";

  const headline = pulse?.summary
    ? capitalize(pulse.summary)
    : `${bullCount} buys · ${bearCount} sells across ${rows.length} symbols`;

  const paragraph = [
    `${opener} across the watchlist — ${bullCount} buys and ${bearCount} sells across ${rows.length} tracked symbols, ${breadthLine}.`,
    leader,
    regimeLine || null,
    convictionTail || null,
  ]
    .filter((s): s is string => s !== null && s.length > 0)
    .join(" ");

  const highlights = buildHighlights(snap);

  return { headline, paragraph, highlights };
}

const SYMBOL_FRAMING: Record<RecommendationRow["action"], (sym: string) => string> = {
  buy: (sym) => `${sym} reads constructively`,
  sell: (sym) => `${sym} is under distribution`,
  hold: (sym) => `${sym} is range-bound`,
};

export function generateSymbolSummary(row: RecommendationRow): string {
  const drivers: string[] = [];
  if (row.sentiment_score > 0.25) drivers.push("positive news sentiment");
  else if (row.sentiment_score > 0.1) drivers.push("incrementally constructive news flow");
  else if (row.sentiment_score < -0.25) drivers.push("negative news sentiment");
  else if (row.sentiment_score < -0.1) drivers.push("incrementally weaker news flow");

  if (row.technical_score > 0.3) drivers.push("technicals firmly aligned to the upside");
  else if (row.technical_score > 0.1) drivers.push("technicals tilting bullish");
  else if (row.technical_score < -0.3) drivers.push("technicals breaking down");
  else if (row.technical_score < -0.1) drivers.push("technicals softening");

  if (row.rsi > 70) drivers.push("RSI in overbought territory");
  else if (row.rsi < 30) drivers.push("RSI flagging oversold conditions");

  if (row.macd > 0.5) drivers.push("MACD trending higher");
  else if (row.macd < -0.5) drivers.push("MACD trending lower");

  if (drivers.length === 0) {
    drivers.push(`${row.num_news_articles} recent headlines without a clear bias`);
  }

  const tail = drivers.slice(0, 2).join(" and ");
  const framing = SYMBOL_FRAMING[row.action](row.symbol);
  const confidence = Math.round(row.confidence * 100);
  const tierClause =
    row.tier === "strong_buy"
      ? "Promoted to STRONG BUY by the 5-tier engine."
      : row.tier === "strong_sell"
        ? "Promoted to STRONG SELL by the 5-tier engine."
        : "";

  const base = `${framing} (${confidence}% confidence) on ${tail}.`;
  return tierClause ? `${base} ${tierClause}` : base;
}

function shortRationale(opp: RankedOpportunity): string {
  if (opp.rationale.length > 0) return opp.rationale[0];
  const drivers: { label: string; v: number }[] = [
    { label: "technical alignment", v: opp.technical_alignment },
    { label: "sentiment alignment", v: opp.sentiment_alignment },
    { label: "momentum persistence", v: opp.momentum_persistence },
    { label: "reversal strength", v: opp.reversal_strength },
    { label: "confidence acceleration", v: opp.confidence_acceleration },
  ];
  drivers.sort((a, b) => b.v - a.v);
  return drivers[0].label;
}

function buildHighlights(snap: DashboardSnapshot): string[] {
  const out: string[] = [];
  const top = snap.ranked_opportunities.slice(0, 3);
  for (const opp of top) {
    out.push(
      `${opp.symbol} · ${tierLower(opp.tier)} · composite ${(opp.composite_score * 100).toFixed(0)}`,
    );
  }
  if (out.length === 0) {
    const strongRow = snap.rows
      .filter((r) => r.action !== "hold")
      .sort((a, b) => b.confidence - a.confidence)[0];
    if (strongRow) {
      out.push(
        `${strongRow.symbol} · ${strongRow.action.toUpperCase()} · ${Math.round(strongRow.confidence * 100)}%`,
      );
    }
  }
  return out;
}

function tierLower(tier: string): string {
  return tier.replace(/_/g, " ").toLowerCase();
}

function capitalize(s: string): string {
  return s.length === 0 ? s : s[0].toUpperCase() + s.slice(1);
}
