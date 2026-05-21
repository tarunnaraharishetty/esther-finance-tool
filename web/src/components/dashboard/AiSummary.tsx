import { ArrowRight, Sparkles, TrendingDown, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { generateMarketSummary } from "@/lib/aiSummary";
import { useTypewriter } from "@/lib/useTypewriter";
import type { DashboardSnapshot } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
}

/**
 * Hero AI summary card. Aurora-drifting mesh background, gradient
 * brand mark, typewriter reveal on headline + paragraph, soft
 * pulsing live dot, breadth chips, top-pick highlights as clickable
 * tracking-wide pills. Phase 4 swaps the templated text for a real
 * streamed LLM response.
 */
export function AiSummary({ snapshot }: Props) {
  const summary = generateMarketSummary(snapshot);
  const headline = useTypewriter(summary.headline, 90);
  const paragraph = useTypewriter(summary.paragraph, 240);
  const bullCount =
    snapshot.pulse?.bullish_count ??
    snapshot.rows.filter((r) => r.action === "buy").length;
  const bearCount =
    snapshot.pulse?.bearish_count ??
    snapshot.rows.filter((r) => r.action === "sell").length;
  const conviction = snapshot.pulse?.conviction;

  return (
    <section className="surface-premium relative overflow-hidden">
      {/* Aurora-drift background. Subtle; reads as ambient light not animation. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-gradient-mesh opacity-90"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -right-24 -top-24 h-64 w-64 rounded-full bg-primary/25 blur-3xl"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-24 -left-12 h-52 w-52 rounded-full bg-accent/20 blur-3xl"
      />

      <div className="relative p-5 md:p-6">
        <div className="mb-3 flex items-center gap-2.5">
          <div className="relative grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-primary to-accent shadow-glow">
            <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
            <span
              aria-hidden
              className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-bull animate-ping-soft"
            />
            <span
              aria-hidden
              className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-bull"
            />
          </div>
          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">
            AI Briefing
          </span>
          <span className="text-muted-foreground/40">·</span>
          <span
            className={
              paragraph.done
                ? "font-mono text-[10px] uppercase tracking-wider text-bull"
                : "font-mono text-[10px] uppercase tracking-wider text-primary"
            }
          >
            {paragraph.done ? "Live" : "Streaming"}
          </span>
          {conviction && (
            <Badge variant="outline" className="ml-auto">
              {conviction}
            </Badge>
          )}
        </div>

        <h2 className="font-display text-xl font-semibold leading-tight tracking-tightest text-gradient-primary md:text-2xl">
          {headline.visible}
          {!headline.done && (
            <span
              aria-hidden
              className="ml-0.5 inline-block h-[1.05em] w-[2px] -mb-0.5 animate-pulse rounded-sm bg-primary align-middle"
            />
          )}
        </h2>

        <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-muted-foreground min-h-[2.5rem]">
          {paragraph.visible}
          {headline.done && !paragraph.done && (
            <span
              aria-hidden
              className="ml-0.5 inline-block h-[0.9em] w-[2px] -mb-0.5 animate-pulse rounded-sm bg-muted-foreground align-middle"
            />
          )}
        </p>

        <div className="mt-4 flex flex-wrap items-center gap-1.5">
          <span className="inline-flex items-center gap-1.5 rounded-md border border-bull/30 bg-bull/10 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-bull">
            <TrendingUp className="h-3 w-3" /> {bullCount} long
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-md border border-bear/30 bg-bear/10 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-bear">
            <TrendingDown className="h-3 w-3" /> {bearCount} short
          </span>
          {summary.highlights.length > 0 && (
            <span className="mx-1 hidden h-3 w-px bg-border/60 md:inline" />
          )}
          {summary.highlights.map((h) => (
            <span
              key={h}
              className="group inline-flex items-center gap-1 rounded-md border border-border/60 bg-secondary/40 px-2 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
            >
              {h}
              <ArrowRight className="h-2.5 w-2.5 transition-transform group-hover:translate-x-0.5" />
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
