import { AlertTriangle, Brain, Clock, Cpu, RefreshCw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RatingDial } from "./RatingDial";
import { ValidationBadge } from "./ValidationBadge";
import {
  ratingLabel,
  ratingTone,
  type ResearchThesis,
} from "@/lib/research";
import { cn } from "@/lib/utils";

interface Props {
  thesis: ResearchThesis;
  onRefresh: () => void;
  refreshing: boolean;
}

/**
 * Top-of-report band. Symbol + rating dial + cache state + tagline
 * + regenerate. Designed to land like a tear-sheet cover page, not a
 * page header — the prose tagline is the AI's one-line gist.
 */
export function ResearchHeader({ thesis, onRefresh, refreshing }: Props) {
  const tone = ratingTone(thesis.rating);
  const generated = new Date(thesis.generated_at);
  return (
    <section className="surface-premium relative overflow-hidden">
      <div
        aria-hidden
        className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-primary/15 blur-3xl"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-24 -left-12 h-48 w-48 rounded-full bg-accent/15 blur-3xl"
      />
      <div className="relative grid gap-4 p-5 md:grid-cols-[1fr_auto] md:p-6">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">
              Research Thesis
            </span>
            <span className="text-muted-foreground/40">·</span>
            <ModeBadge mode={thesis.mode} model={thesis.model} />
          </div>
          <h1 className="mt-2 font-display text-3xl font-semibold tracking-tightest md:text-4xl">
            {thesis.symbol}
          </h1>
          <p className="mt-2 max-w-2xl text-[14px] leading-relaxed text-foreground/90">
            {thesis.tagline}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span
              className={cn(
                "rounded-md border px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-wider",
                tone === "bull" && "border-bull/40 bg-bull/15 text-bull",
                tone === "bear" && "border-bear/40 bg-bear/15 text-bear",
                tone === "warn" && "border-warn/40 bg-warn/15 text-warn",
              )}
            >
              {ratingLabel(thesis.rating)}
            </span>
            <span className="rounded-md border border-border/50 bg-card/40 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              {Math.round(thesis.confidence * 100)}% confidence
            </span>
            <span
              className="flex items-center gap-1 rounded-md border border-border/50 bg-card/40 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground"
              title={`Generated ${generated.toLocaleString()}`}
            >
              <Clock className="h-3 w-3" />
              {formatRelative(generated)}
            </span>
            <span
              className="flex items-center gap-1 rounded-md border border-border/50 bg-card/40 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground"
              title="Cache state"
            >
              <Cpu className="h-3 w-3" />
              cache · {thesis.cache}
            </span>
            <ValidationBadge validation={thesis.validation} />
            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              disabled={refreshing}
              className="ml-auto h-7 font-mono text-[10px] uppercase tracking-wider"
            >
              <RefreshCw className={cn("h-3 w-3", refreshing && "animate-spin")} />
              {refreshing ? "Regenerating…" : "Regenerate"}
            </Button>
          </div>
          {thesis.warning && (
            <div className="mt-3 flex items-start gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-[12px] text-warn">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{thesis.warning}</span>
            </div>
          )}
        </div>
        <div className="grid place-items-center">
          <RatingDial
            rating={thesis.rating}
            confidence={thesis.confidence}
            size="lg"
          />
        </div>
      </div>
    </section>
  );
}

function ModeBadge({ mode, model }: { mode: "llm" | "template"; model: string }) {
  const isLLM = mode === "llm";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider",
        isLLM
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-border/50 bg-card/40 text-muted-foreground",
      )}
      title={model}
    >
      {isLLM ? (
        <>
          <Brain className="h-2.5 w-2.5" />
          claude · {model}
        </>
      ) : (
        <>{model}</>
      )}
    </span>
  );
}

function formatRelative(d: Date): string {
  const ms = Date.now() - d.getTime();
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const days = Math.round(h / 24);
  return `${days}d ago`;
}
