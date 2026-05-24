import {
  AlertTriangle,
  Brain,
  ChevronDown,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useState } from "react";
import {
  narrativeSections,
  useComparisonNarrative,
  type ComparisonNarrative,
  type NarrativeMode,
  type NarrativeValidation,
} from "@/lib/compareNarrative";
import { ProvenanceProse } from "@/components/prose/ProvenanceProse";
import { TrustScoreBadge } from "@/components/trust/TrustScoreBadge";
import { cn } from "@/lib/utils";

interface Props {
  leftSymbol: string | null;
  rightSymbol: string | null;
  /** Default mode when the user clicks Generate. ``"auto"`` prefers
   *  the LLM if configured, falls back to template. */
  defaultMode?: NarrativeMode;
}

/**
 * AI-narrative panel for the comparison page.
 *
 * Collapsed by default with a single "Generate AI narrative" CTA —
 * the LLM call is expensive and the structured comparison view above
 * already covers the load-bearing trust signal. When the trader wants
 * prose synthesis they opt in explicitly.
 *
 * On render the panel shows:
 *  * The tagline + a trust badge + a validation chip
 *  * Six sections (Headline / Momentum / Valuation / Risk / Quality /
 *    Bottom line) — each is a paragraph plus 2-4 bullets
 *  * A Regenerate button + a Force-fresh button
 *
 * Validation framing: a non-zero ``drop_count`` is *positive* — the
 * validator refused to publish claims it couldn't ground. The chip
 * surfaces the count and the dropped sentences are inspectable on
 * hover via the chip's tooltip.
 */
export function NarrativePanel({
  leftSymbol,
  rightSymbol,
  defaultMode = "auto",
}: Props) {
  const { narrative, loading, error, trigger } = useComparisonNarrative(
    leftSymbol,
    rightSymbol,
  );

  if (leftSymbol === null || rightSymbol === null || leftSymbol === rightSymbol) {
    return null;
  }

  if (narrative === null && !loading && error === null) {
    return (
      <section
        data-testid="narrative-panel"
        data-state="empty"
        className="surface flex items-start gap-3 p-4"
      >
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            AI Narrative
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Generate a grounded prose comparison of{" "}
            <strong className="text-foreground">{leftSymbol}</strong> vs{" "}
            <strong className="text-foreground">{rightSymbol}</strong>. Every
            claim is validated against the underlying data; unsupported
            sentences are dropped, not softened.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void trigger({ mode: defaultMode })}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-primary hover:bg-primary/20"
        >
          <Sparkles className="h-3 w-3" />
          Generate
        </button>
      </section>
    );
  }

  if (loading && narrative === null) {
    return (
      <section
        data-testid="narrative-panel"
        data-state="loading"
        className="surface flex items-start gap-3 p-4"
      >
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
          <Brain className="h-4 w-4 animate-pulse text-primary" />
        </div>
        <div className="flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            AI Narrative
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Generating grounded comparison narrative…
          </p>
        </div>
      </section>
    );
  }

  if (error !== null) {
    return (
      <section
        data-testid="narrative-panel"
        data-state="error"
        className="surface flex items-start gap-3 border-bear/40 p-4"
      >
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-bear" />
        <div className="flex-1">
          <div className="font-mono text-[10px] uppercase tracking-wider text-bear">
            Narrative failed
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{error}</p>
          <button
            type="button"
            onClick={() => void trigger({ mode: defaultMode, fresh: true })}
            className="mt-2 inline-flex items-center gap-1 rounded-md border border-border/40 bg-card/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider hover:bg-card"
          >
            <RefreshCw className="h-3 w-3" /> Retry
          </button>
        </div>
      </section>
    );
  }

  // narrative !== null here.
  return (
    <NarrativeBody
      narrative={narrative!}
      loading={loading}
      onRegenerate={() => void trigger({ mode: defaultMode, fresh: true })}
    />
  );
}

function NarrativeBody({
  narrative,
  loading,
  onRegenerate,
}: {
  narrative: ComparisonNarrative;
  loading: boolean;
  onRegenerate: () => void;
}) {
  return (
    <section
      data-testid="narrative-panel"
      data-state="ready"
      className="surface-premium relative overflow-hidden p-5"
    >
      <div className="flex flex-wrap items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
          <Sparkles className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            AI Narrative
          </div>
          <p className="mt-1 text-[14px] leading-relaxed text-foreground/90">
            {narrative.tagline}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <ModeBadge mode={narrative.mode} model={narrative.model} />
            <ValidationChip validation={narrative.validation} />
            <TrustScoreBadge trust={narrative.trust_score} variant="inline" />
            {narrative.cache === "hit" && (
              <span className="rounded-md border border-border/40 bg-card/40 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                cached
              </span>
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={onRegenerate}
          disabled={loading}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border/40 bg-card/60 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:bg-card hover:text-foreground disabled:opacity-40"
        >
          <RefreshCw className={loading ? "h-3 w-3 animate-spin" : "h-3 w-3"} />
          Regenerate
        </button>
      </div>

      {narrative.warning && (
        <div className="mt-3 flex items-start gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-[12px] text-warn">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{narrative.warning}</span>
        </div>
      )}

      <div className="mt-5 space-y-3">
        {narrativeSections(narrative).map(({ key, section }) => (
          <NarrativeSectionBlock key={key} sectionKey={key} section={section} />
        ))}
      </div>
    </section>
  );
}

function NarrativeSectionBlock({
  sectionKey,
  section,
}: {
  sectionKey: string;
  section: ComparisonNarrative["headline"];
}) {
  const [open, setOpen] = useState(true);
  const empty = !section.body.trim() && section.bullets.length === 0;
  return (
    <article
      data-testid={`narrative-section-${sectionKey}`}
      className="rounded-md border border-border/40 bg-card/30"
    >
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
      >
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          {section.title}
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 text-muted-foreground transition-transform",
            !open && "-rotate-90",
          )}
        />
      </button>
      {open && (
        <div className="border-t border-border/30 px-3 py-3">
          {empty ? (
            <p className="text-[12px] italic text-muted-foreground">
              Every claim in this section dropped during validation — no
              grounded prose to render.
            </p>
          ) : (
            <>
              {section.body.trim() && (
                <p className="text-[13px] leading-relaxed text-foreground/90">
                  <ProvenanceProse
                    body={section.body}
                    provenance={section.provenance}
                  />
                </p>
              )}
              {section.bullets.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {section.bullets.map((b, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-[12px] text-muted-foreground"
                    >
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground/60" />
                      <ProvenanceProse
                        body={b}
                        provenance={section.provenance}
                      />
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </article>
  );
}

function ModeBadge({
  mode,
  model,
}: {
  mode: ComparisonNarrative["mode"];
  model: string;
}) {
  const isLLM = mode === "llm";
  return (
    <span
      title={model}
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[10px] uppercase tracking-wider",
        isLLM
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-border/50 bg-card/40 text-muted-foreground",
      )}
    >
      {isLLM ? (
        <>
          <Brain className="h-3 w-3" />
          claude · {mode}
        </>
      ) : (
        <>template</>
      )}
    </span>
  );
}

function ValidationChip({ validation }: { validation: NarrativeValidation }) {
  const dropped = validation.drop_count;
  const tone = dropped === 0 ? "bull" : "warn";
  const Icon = dropped === 0 ? ShieldCheck : ShieldAlert;
  const tooltip =
    dropped === 0
      ? "Every claim in this narrative anchored to the comparison data."
      : validation.dropped_claims
          .map((c) => `${c.section}: ${c.sentence}`)
          .join("\n");
  return (
    <span
      data-testid="narrative-validation-chip"
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
