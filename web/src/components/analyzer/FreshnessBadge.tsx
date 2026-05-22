import type { FundamentalsFreshness } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  freshness: FundamentalsFreshness | null;
}

/**
 * Compact freshness chip for the analyzer header.
 *
 * Renders the four-tier freshness signal — `fresh / aging / stale /
 * expired` — using the app's standard semantic palette (bull / warn /
 * bear) so the trader scans staleness the same way they scan tier
 * tones elsewhere. Includes:
 *
 *  * A colored dot for at-a-glance tier read.
 *  * The age in human terms ("51d", "11mo").
 *  * Provider chain ("fmp → finnhub") so the source is visible.
 *  * A divergence chip when reconciliation found disagreement.
 *
 * Returns ``null`` when freshness is absent (no fundamentals on this
 * report). Absence has a meaning distinct from "expired" — both
 * states are honest, neither should fake the other.
 */
export function FreshnessBadge({ freshness }: Props) {
  if (freshness === null) return null;
  const tone = toneFor(freshness.freshness);
  return (
    <div
      data-testid="freshness-badge"
      className="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-wider"
    >
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5",
          tone.border,
          tone.bg,
          tone.text,
        )}
        data-freshness={freshness.freshness}
      >
        <span className={cn("h-1.5 w-1.5 rounded-full", tone.dot)} />
        {freshness.freshness}
        <span className="text-muted-foreground">· {ageLabel(freshness.data_age_days)}</span>
      </span>
      {freshness.source_chain.length > 0 && (
        <span className="text-muted-foreground">
          via {freshness.source_chain.join(" → ")}
        </span>
      )}
      {freshness.divergence_count > 0 && (
        <span
          data-testid="freshness-divergence"
          className="rounded-md border border-warn/40 bg-warn/10 px-1.5 py-0.5 text-warn"
        >
          + {freshness.divergence_count} disagree
        </span>
      )}
    </div>
  );
}

interface Tone {
  border: string;
  bg: string;
  text: string;
  dot: string;
}

function toneFor(level: FundamentalsFreshness["freshness"]): Tone {
  // The freshness palette reuses the app's existing semantic tokens:
  // bull = good, warn = caution, bear = problem. Mapping the four
  // tiers onto three colors gives traders a stable scanning model —
  // ``stale`` and ``expired`` both bear because both block alerts.
  switch (level) {
    case "fresh":
      return {
        border: "border-bull/40",
        bg: "bg-bull/10",
        text: "text-bull",
        dot: "bg-bull",
      };
    case "aging":
      return {
        border: "border-warn/40",
        bg: "bg-warn/10",
        text: "text-warn",
        dot: "bg-warn",
      };
    case "stale":
    case "expired":
      return {
        border: "border-bear/40",
        bg: "bg-bear/10",
        text: "text-bear",
        dot: "bg-bear",
      };
  }
}

function ageLabel(days: number): string {
  // Compact age formatting: < 1 day → "today", < 60d → days, < 24mo →
  // months, otherwise years. Two significant figures, no decimals.
  if (days < 1) return "today";
  if (days < 60) return `${Math.round(days)}d`;
  const months = Math.round(days / 30);
  if (months < 24) return `${months}mo`;
  const years = Math.floor(days / 365);
  return `${years}y`;
}
