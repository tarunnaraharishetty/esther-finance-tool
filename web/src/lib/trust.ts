/**
 * Report Trust Score — TypeScript surface for `src/intelligence/trust_score.py`.
 *
 * Mirrors the wire shape one-to-one. Adding a field on the Python side
 * without updating this file will break `tsc --strict`, which is the
 * intended contract.
 *
 * The trust score is the platform's north-star quality metric: a single
 * 0–100 + letter grade composed from data freshness, provider
 * confidence, analyzer confidence, calibration coverage, scenario
 * availability, and LLM validation cleanliness. Components that don't
 * apply to a given report type are marked ``n/a`` and excluded from
 * the weighted sum.
 */

export type TrustComponentStatus = "ok" | "warn" | "missing" | "n/a";

export interface TrustComponent {
  name: string;
  weight: number;
  value: number | null; // [0, 100] or null when status is missing/n/a
  contribution: number | null; // weighted contribution after renormalization
  status: TrustComponentStatus;
  detail: string;
}

export interface TrustScore {
  score: number; // [0, 100]
  grade: "A+" | "A" | "B" | "C" | "D" | "F";
  components: TrustComponent[];
}

/**
 * Map a letter grade to one of the app's semantic palette tones.
 * Used by the badge to colour the chip without baking palette decisions
 * into every render site.
 */
export function gradeTone(grade: TrustScore["grade"]): "bull" | "warn" | "bear" {
  if (grade === "A+" || grade === "A") return "bull";
  if (grade === "B" || grade === "C") return "warn";
  return "bear";
}

/**
 * Human-readable label for a component name. The Python module uses
 * snake_case keys; the UI prefers Title Case with the right framing.
 */
export const COMPONENT_LABELS: Record<string, string> = {
  freshness: "Data freshness",
  provider_confidence: "Provider confidence",
  analyzer_confidence: "Analyzer confidence",
  calibration_coverage: "Calibration coverage",
  scenario_availability: "Scenario layer",
  validation_cleanliness: "AI validation",
};

export function componentLabel(name: string): string {
  return COMPONENT_LABELS[name] ?? name;
}
