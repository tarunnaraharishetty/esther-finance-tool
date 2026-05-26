import { Sparkles } from "lucide-react";
import { AiSummary } from "@/components/dashboard/AiSummary";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Panel } from "@/components/layout/Panel";
import { generateSymbolSummary } from "@/lib/aiSummary";
import { actionTone, fmtConfidence, tierLabel } from "@/lib/format";
import type { DashboardSnapshot } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  setActiveSymbol: (s: string) => void;
}

/**
 * Auto-insights surface: hero auto-briefing up top + a card per
 * non-hold row with the per-symbol auto rationale.
 *
 * Both the hero card and the per-symbol cards render *templated*
 * prose built from the snapshot fields already shown elsewhere on
 * the page — they are not LLM-generated. The "Heuristic" badges
 * surface this contract to the user. For LLM-written analysis,
 * see the Research and Analyzer pages.
 */
export function AiPage({ snapshot, setActiveSymbol }: Props) {
  const interesting = snapshot.rows
    .filter((r) => r.action !== "hold")
    .sort((a, b) => b.confidence - a.confidence);

  return (
    <div className="space-y-4">
      <AiSummary snapshot={snapshot} />

      <Panel
        title="Per-symbol Auto Reads"
        subtitle={`${interesting.length} actionable · heuristic, not LLM-generated`}
      >
        {interesting.length === 0 ? (
          <div className="grid place-items-center rounded-md border border-dashed border-border/40 bg-card/30 py-8 font-mono text-xs uppercase tracking-wider text-muted-foreground">
            No actionable signals · all hold
          </div>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {interesting.map((row) => (
              <Card
                key={row.symbol}
                role="button"
                tabIndex={0}
                onClick={() => setActiveSymbol(row.symbol)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setActiveSymbol(row.symbol);
                  }
                }}
                className="cursor-pointer p-3.5 transition-all hover:-translate-y-0.5 hover:shadow-lg"
              >
                <div className="mb-2 flex flex-wrap items-center gap-1.5">
                  <Sparkles className="h-3 w-3 text-primary" />
                  <span className="font-mono text-sm font-semibold">
                    {row.symbol}
                  </span>
                  <Badge variant={actionTone(row.action)}>
                    {row.action.toUpperCase()}
                  </Badge>
                  <Badge variant="outline">{tierLabel(row.tier)}</Badge>
                  <Badge
                    variant="outline"
                    className="font-mono text-[9px] uppercase tracking-wider"
                    title="Templated prose. Not LLM-generated."
                  >
                    Heuristic
                  </Badge>
                  <span className="ml-auto font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                    {fmtConfidence(row.confidence)}
                  </span>
                </div>
                <p className="text-[13px] leading-relaxed text-muted-foreground">
                  {generateSymbolSummary(row)}
                </p>
              </Card>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
