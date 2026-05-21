/** Heuristic importance scoring for news headlines.
 *
 * The backend doesn't ship per-article importance scores, so we
 * derive a 0..1 score client-side from observable signal:
 *   - keyword density (earnings, FDA, merger, guidance, etc.)
 *   - underlying row's tier (STRONG_* moves are more notable)
 *   - sentiment magnitude (|score| > 0.5)
 *   - news flow density on the symbol
 *
 * Anything ≥ 0.55 gets pinned to the Important News section with a
 * "why it matters" string built from the same heuristics — so the
 * trader sees the *reason* the engine flagged it, never just a
 * mystery badge.
 */

import type { RecommendationRow } from "./types";

export interface NewsImportance {
  score: number;
  reasons: string[];
}

interface KeywordRule {
  pattern: RegExp;
  weight: number;
  reason: string;
}

const KEYWORD_RULES: KeywordRule[] = [
  { pattern: /\bearnings?\b/i, weight: 0.35, reason: "Earnings catalyst" },
  { pattern: /\b(beat|miss)\b/i, weight: 0.25, reason: "Beat/miss reported" },
  { pattern: /\bguidance\b/i, weight: 0.3, reason: "Forward guidance update" },
  { pattern: /\bdividend\b/i, weight: 0.18, reason: "Dividend-related" },
  { pattern: /\bbuyback\b/i, weight: 0.22, reason: "Buyback announced" },
  { pattern: /\b(merger|acquisition|acquir(e|ing|ed)|deal)\b/i, weight: 0.32, reason: "M&A activity" },
  { pattern: /\b(FDA|approval|trial)\b/i, weight: 0.32, reason: "Regulatory / clinical" },
  { pattern: /\b(SEC|lawsuit|investigation|probe|charged)\b/i, weight: 0.34, reason: "Regulatory / legal" },
  { pattern: /\bCEO\b/i, weight: 0.22, reason: "Leadership change" },
  { pattern: /\b(downgrade|upgrade)\b/i, weight: 0.2, reason: "Analyst rating change" },
  { pattern: /\b(beats? estimates?|tops? expectations?)\b/i, weight: 0.28, reason: "Beat estimates" },
  { pattern: /\b(misses? estimates?|falls? short)\b/i, weight: 0.28, reason: "Missed estimates" },
  { pattern: /\b(record|all-time high|ATH)\b/i, weight: 0.22, reason: "Record print" },
  { pattern: /\b(crash|plunge|tumble|surge|soar|spike)\b/i, weight: 0.18, reason: "Sharp price move" },
  { pattern: /\b(layoffs?|restructur|cuts?)\b/i, weight: 0.2, reason: "Cost / headcount action" },
  { pattern: /\b(launch|unveil|announce)\b/i, weight: 0.14, reason: "Product / partnership" },
];

export function importanceFor(
  headline: string,
  row: RecommendationRow,
): NewsImportance {
  let score = 0;
  const reasons: string[] = [];

  for (const rule of KEYWORD_RULES) {
    if (rule.pattern.test(headline)) {
      score += rule.weight;
      reasons.push(rule.reason);
    }
  }

  if (row.tier === "strong_buy" || row.tier === "strong_sell") {
    score += 0.22;
    reasons.push(`On a ${row.tier.replace("_", " ").toUpperCase()} signal`);
  }

  const sentMag = Math.abs(row.sentiment_score);
  if (sentMag > 0.6) {
    score += 0.2;
    reasons.push(
      row.sentiment_score > 0
        ? "Strong positive sentiment"
        : "Strong negative sentiment",
    );
  }

  if (row.num_news_articles >= 4) {
    score += 0.12;
    reasons.push("Elevated news flow today");
  }

  // Dedupe reason strings so a headline matched by both "beat" and
  // "beats estimates" doesn't render the same chip twice.
  const dedup = Array.from(new Set(reasons));
  return { score: Math.min(1, score), reasons: dedup };
}

/** Build a one-sentence "why it matters" derived from the reasons. */
export function whyItMatters(imp: NewsImportance): string {
  if (imp.reasons.length === 0) return "Flagged by aggregate signal density.";
  if (imp.reasons.length === 1) return imp.reasons[0] + ".";
  const last = imp.reasons[imp.reasons.length - 1];
  const head = imp.reasons.slice(0, -1).join(", ");
  return `${head} and ${last.toLowerCase()}.`;
}
