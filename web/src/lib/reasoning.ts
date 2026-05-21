/** Bull/bear factor extraction from a `RecommendationRow`.
 *
 * Powers the "Why is this moving?" panel. Every factor is sourced
 * directly from snapshot fields — no invented data. Weight is a
 * relative 0..1 score so the bar widths read meaningfully even
 * without per-factor normalization on the backend. */

import type { RecommendationRow } from "./types";

export interface Factor {
  label: string;
  detail: string;
  weight: number;
  tone: "bull" | "bear";
}

export interface ReasoningSummary {
  bullish: Factor[];
  bearish: Factor[];
  net: "bullish" | "bearish" | "balanced";
  tilt: number;
}

export function summarizeReasoning(row: RecommendationRow): ReasoningSummary {
  const bullish: Factor[] = [];
  const bearish: Factor[] = [];

  // Sentiment
  if (row.sentiment_score > 0.15) {
    bullish.push({
      label: "Positive news sentiment",
      detail: `Sentiment ${row.sentiment_score.toFixed(2)} across ${row.num_news_articles} headlines`,
      weight: Math.min(1, row.sentiment_score),
      tone: "bull",
    });
  } else if (row.sentiment_score < -0.15) {
    bearish.push({
      label: "Negative news sentiment",
      detail: `Sentiment ${row.sentiment_score.toFixed(2)} across ${row.num_news_articles} headlines`,
      weight: Math.min(1, Math.abs(row.sentiment_score)),
      tone: "bear",
    });
  }

  // Technicals (combined)
  if (row.technical_score > 0.15) {
    bullish.push({
      label: "Technicals aligned up",
      detail: `Indicator stack +${row.technical_score.toFixed(2)}`,
      weight: Math.min(1, row.technical_score),
      tone: "bull",
    });
  } else if (row.technical_score < -0.15) {
    bearish.push({
      label: "Technicals breaking down",
      detail: `Indicator stack ${row.technical_score.toFixed(2)}`,
      weight: Math.min(1, Math.abs(row.technical_score)),
      tone: "bear",
    });
  }

  // RSI extremes
  if (row.rsi > 70) {
    bearish.push({
      label: "RSI overbought",
      detail: `RSI ${row.rsi.toFixed(0)} — extended`,
      weight: Math.min(1, (row.rsi - 70) / 30),
      tone: "bear",
    });
  } else if (row.rsi < 30) {
    bullish.push({
      label: "RSI oversold",
      detail: `RSI ${row.rsi.toFixed(0)} — coiled for mean revert`,
      weight: Math.min(1, (30 - row.rsi) / 30),
      tone: "bull",
    });
  }

  // MACD
  if (row.macd > 0.5) {
    bullish.push({
      label: "MACD trending up",
      detail: `MACD ${row.macd.toFixed(2)}`,
      weight: Math.min(1, row.macd / 2),
      tone: "bull",
    });
  } else if (row.macd < -0.5) {
    bearish.push({
      label: "MACD trending down",
      detail: `MACD ${row.macd.toFixed(2)}`,
      weight: Math.min(1, Math.abs(row.macd) / 2),
      tone: "bear",
    });
  }

  // Bollinger position (>1 means above upper band; <-1 below lower)
  if (row.bollinger > 0.7) {
    bearish.push({
      label: "Stretched above Bollinger",
      detail: `Band position ${row.bollinger.toFixed(2)}`,
      weight: Math.min(1, row.bollinger - 0.5),
      tone: "bear",
    });
  } else if (row.bollinger < -0.7) {
    bullish.push({
      label: "Stretched below Bollinger",
      detail: `Band position ${row.bollinger.toFixed(2)}`,
      weight: Math.min(1, Math.abs(row.bollinger) - 0.5),
      tone: "bull",
    });
  }

  // Headline density (volume of news = caution flag for any direction)
  if (row.num_news_articles >= 4) {
    // News density without sentiment lean reads neutral — only add as
    // a softer factor on whichever side already leads.
    const leader = bullish.length > bearish.length ? bullish : bearish;
    leader.push({
      label: "Elevated news flow",
      detail: `${row.num_news_articles} headlines this session`,
      weight: 0.35,
      tone: leader === bullish ? "bull" : "bear",
    });
  }

  bullish.sort((a, b) => b.weight - a.weight);
  bearish.sort((a, b) => b.weight - a.weight);

  const bullSum = bullish.reduce((s, f) => s + f.weight, 0);
  const bearSum = bearish.reduce((s, f) => s + f.weight, 0);
  const total = bullSum + bearSum;
  const tilt = total === 0 ? 0 : (bullSum - bearSum) / total;
  const net: ReasoningSummary["net"] =
    tilt > 0.2 ? "bullish" : tilt < -0.2 ? "bearish" : "balanced";

  return { bullish, bearish, net, tilt };
}
