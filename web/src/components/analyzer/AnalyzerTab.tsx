import { AlertOctagon, RefreshCw } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { useAnalyzerReport } from "@/lib/analyzer";
import { AiExplanation } from "./AiExplanation";
import { BullBearCases } from "./BullBearCases";
import { FairValueRange } from "./FairValueRange";
import { OverboughtSpectrum } from "./OverboughtSpectrum";
import { RiskWarnings } from "./RiskWarnings";
import { ScoreHeader } from "./ScoreHeader";

interface Props {
  symbol: string | null;
}

/**
 * Container for the Financial Analyzer tab. Pulls the assembled
 * report from `/api/analyzer/{symbol}`, threads it through the six
 * presentational sub-components, and surfaces refresh + error UX.
 */
export function AnalyzerTab({ symbol }: Props) {
  const { report, loading, error, refresh } = useAnalyzerReport(symbol);

  if (symbol === null) {
    return (
      <section className="surface p-6 text-center text-sm text-muted-foreground">
        Select a symbol to load its Financial Analyzer report.
      </section>
    );
  }

  if (error) {
    return (
      <section className="surface flex items-start gap-3 border-bear/40 p-4">
        <AlertOctagon className="mt-0.5 h-4 w-4 shrink-0 text-bear" />
        <div className="flex-1">
          <div className="font-mono text-[10px] uppercase tracking-wider text-bear">
            Analyzer load failed
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{error}</p>
          <button
            type="button"
            onClick={() => {
              void refresh({ fresh: true });
            }}
            className="mt-2 inline-flex items-center gap-1 rounded-md border border-border/40 bg-card/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-foreground hover:bg-card"
          >
            <RefreshCw className="h-3 w-3" /> Retry
          </button>
        </div>
      </section>
    );
  }

  if (!report) {
    return <AnalyzerSkeleton />;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end">
        <button
          type="button"
          onClick={() => {
            void refresh({ fresh: true });
          }}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-md border border-border/40 bg-card/60 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:bg-card hover:text-foreground disabled:opacity-40"
        >
          <RefreshCw
            className={loading ? "h-3 w-3 animate-spin" : "h-3 w-3"}
          />
          {report.cache === "hit"
            ? `Cached · ${report.cache_age_seconds}s ago`
            : "Refresh"}
        </button>
      </div>

      <ScoreHeader
        symbol={report.symbol}
        lastPrice={report.last_price}
        overall={report.overall_analyzer_score}
        technical={report.technical_score}
        fundamental={report.fundamental_score}
        valuation={report.valuation_score}
      />

      <OverboughtSpectrum technicals={report.technicals} />

      <FairValueRange
        valuation={report.valuation}
        lastPrice={report.last_price}
      />

      <BullBearCases
        valuation={report.valuation}
        lastPrice={report.last_price}
      />

      <AiExplanation explanation={report.explanation} />

      <RiskWarnings
        explanation={report.explanation}
        warnings={report.warnings}
      />
    </div>
  );
}

function AnalyzerSkeleton() {
  return (
    <div className="space-y-4">
      <div className="surface-premium p-5">
        <Skeleton className="h-4 w-32 opacity-60" />
        <Skeleton className="mt-2 h-8 w-40 opacity-60" />
        <Skeleton className="mt-4 h-20 w-full opacity-50" />
      </div>
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="surface p-4">
          <Skeleton className="h-4 w-1/3 opacity-60" />
          <Skeleton className="mt-3 h-20 w-full opacity-50" />
        </div>
      ))}
    </div>
  );
}
