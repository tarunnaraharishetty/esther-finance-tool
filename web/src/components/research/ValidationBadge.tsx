import { ShieldCheck, ShieldAlert } from "lucide-react";
import type { ValidationReport } from "@/lib/research";
import { cn } from "@/lib/utils";

interface Props {
  validation: ValidationReport;
}

/**
 * Trust-signal badge for the research header.
 *
 * Renders the post-hoc claim validator's outcome as a chip:
 *
 *  * Zero drops → ``Grounded · 0 dropped`` in the bull palette.
 *    The thesis was published verbatim — every numeric claim
 *    anchored in the input bundle.
 *  * One or more drops → ``Grounded · N dropped`` in the warn palette,
 *    with a ``title`` attribute showing the dropped sentences so the
 *    operator can audit on hover.
 *
 * Framing rule (codified here): a drop is *positive*. The system
 * refused to publish a claim it couldn't anchor. The badge surfaces
 * that refusal as evidence of grounding, not as a defect. Hiding or
 * softening the count would erode the institutional contract.
 */
export function ValidationBadge({ validation }: Props) {
  const dropped = validation.drop_count;
  const tone = dropped === 0 ? "bull" : "warn";
  const Icon = dropped === 0 ? ShieldCheck : ShieldAlert;
  const tooltip =
    dropped === 0
      ? "Every numeric claim in this thesis anchored to the input data."
      : validation.dropped_claims
          .map((c) => `${c.section}: ${c.sentence}`)
          .join("\n");
  return (
    <span
      data-testid="validation-badge"
      data-drop-count={dropped}
      title={tooltip}
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[10px] uppercase tracking-wider",
        tone === "bull" && "border-bull/40 bg-bull/15 text-bull",
        tone === "warn" && "border-warn/40 bg-warn/15 text-warn",
      )}
    >
      <Icon className="h-3 w-3" />
      Grounded · {dropped} {dropped === 1 ? "claim dropped" : "claims dropped"}
    </span>
  );
}
