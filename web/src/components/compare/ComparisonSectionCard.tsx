import type { ComparisonSection } from "@/lib/compare";
import { MetricRow } from "./MetricRow";

interface Props {
  section: ComparisonSection;
}

/**
 * One named group of metrics (Trust / Valuation / Technical / Risk).
 *
 * Renders as a sectioned surface card so the page reads as four
 * clear blocks instead of one long table. The header carries the
 * section title and a running winner tally for that section — a
 * trader scanning the page can ignore individual rows and just read
 * "right wins 3/4 on Risk".
 */
export function ComparisonSectionCard({ section }: Props) {
  const tally = sectionTally(section);
  return (
    <section
      data-testid={`compare-section-${section.title.toLowerCase()}`}
      className="surface overflow-hidden"
    >
      <header className="flex items-center justify-between gap-2 border-b border-border/40 bg-card/30 px-4 py-2.5">
        <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          {section.title}
        </div>
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          {tally.left}–{tally.right}
          {tally.tie > 0 ? ` · ${tally.tie} tie` : ""}
        </div>
      </header>
      <div>
        {section.metrics.map((m) => (
          <MetricRow key={m.metric} metric={m} />
        ))}
      </div>
    </section>
  );
}

function sectionTally(section: ComparisonSection): {
  left: number;
  right: number;
  tie: number;
} {
  let left = 0;
  let right = 0;
  let tie = 0;
  for (const m of section.metrics) {
    if (m.winner === "left") left += 1;
    else if (m.winner === "right") right += 1;
    else if (m.winner === "tie") tie += 1;
  }
  return { left, right, tie };
}
