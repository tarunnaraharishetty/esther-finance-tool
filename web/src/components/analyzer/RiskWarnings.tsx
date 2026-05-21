import { AlertTriangle } from "lucide-react";
import type { AnalyzerExplanation } from "@/lib/analyzer";
import { parseClaimCitations } from "@/lib/analyzer";
import { CitationChip } from "./AiExplanation";

interface Props {
  explanation: AnalyzerExplanation;
  warnings: string[];
}

/**
 * Two-tier risk surface:
 *
 * 1. **Risk warnings** — grounded analyzer risk reads (e.g. "ATR is
 *    elevated [technicals]"). Same citation contract as the AI
 *    claims.
 * 2. **Data warnings** — plain-prose strings appended by the route
 *    when a data stream failed to land (chain exhaustion, missing
 *    bars, etc.). These are infrastructure-level, not investment
 *    risks, so they get a distinct visual.
 */
export function RiskWarnings({ explanation, warnings }: Props) {
  const risks = explanation.risk_warnings.map((raw) => parseClaimCitations(raw));
  if (risks.length === 0 && warnings.length === 0) return null;

  return (
    <section className="surface p-5">
      <header className="mb-3 flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 text-warn" />
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Risk Warnings
          </div>
          <h3 className="text-base font-semibold tracking-tight text-warn">
            What could go wrong
          </h3>
        </div>
      </header>

      {risks.length > 0 && (
        <ul className="space-y-2.5">
          {risks.map((r, i) => (
            <li
              key={`r-${i}`}
              className="flex flex-col gap-1 border-l-2 border-warn/40 pl-3"
            >
              <p className="text-sm leading-relaxed text-foreground/90">
                {r.text}
              </p>
              {r.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {r.tags.map((tag) => (
                    <CitationChip key={tag} tag={tag} />
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {warnings.length > 0 && (
        <div className="mt-3 space-y-1 border-t border-border/40 pt-3">
          <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            Data coverage
          </div>
          <ul className="space-y-1 text-xs text-muted-foreground">
            {warnings.map((w, i) => (
              <li key={`w-${i}`} className="leading-relaxed">
                • {w}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
