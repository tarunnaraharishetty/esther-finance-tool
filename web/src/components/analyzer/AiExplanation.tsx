import type { AnalyzerExplanation, CitationTag } from "@/lib/analyzer";
import { parseClaimCitations } from "@/lib/analyzer";
import { cn } from "@/lib/utils";

interface Props {
  explanation: AnalyzerExplanation;
}

/**
 * Renders the grounded AI explanation.
 *
 * Each claim has its citation tags stripped from the prose and
 * surfaced as colored chips. The chips are the contract that's
 * enforced server-side by `validate_citations` — if the chip is
 * present, the claim was grounded in that input stream.
 */
export function AiExplanation({ explanation }: Props) {
  const claims = explanation.claims.map((raw) => ({
    raw,
    parsed: parseClaimCitations(raw),
  }));

  const generator = explanation.model.includes("template")
    ? "Deterministic (template)"
    : `LLM · ${explanation.model}`;

  return (
    <section className="surface p-5">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            AI Explanation · Grounded
          </div>
          <h3 className="mt-1 text-base font-semibold tracking-tight">
            {explanation.summary || "Analyzer summary unavailable."}
          </h3>
        </div>
        <span className="rounded-md border border-border/40 bg-card/40 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          {generator}
        </span>
      </header>

      {claims.length === 0 ? (
        <p className="mt-3 text-sm text-muted-foreground">
          No grounded claims could be produced — none of the analyzer's
          input streams had usable data this run.
        </p>
      ) : (
        <ul className="mt-4 space-y-2.5">
          {claims.map((c, i) => (
            <ClaimRow key={i} text={c.parsed.text} tags={c.parsed.tags} />
          ))}
        </ul>
      )}
    </section>
  );
}

function ClaimRow({ text, tags }: { text: string; tags: CitationTag[] }) {
  return (
    <li className="flex flex-col gap-1 border-l-2 border-primary/30 pl-3">
      <p className="text-sm leading-relaxed text-foreground/90">{text}</p>
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {tags.map((tag) => (
            <CitationChip key={tag} tag={tag} />
          ))}
        </div>
      )}
    </li>
  );
}

export function CitationChip({ tag }: { tag: CitationTag }) {
  return (
    <span
      className={cn(
        "rounded-sm border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider",
        toneFor(tag),
      )}
      title={`Grounded in ${tag} input`}
    >
      {tag}
    </span>
  );
}

function toneFor(tag: CitationTag): string {
  switch (tag) {
    case "technicals":
      return "border-bull/40 bg-bull/10 text-bull";
    case "fundamentals":
      return "border-primary/40 bg-primary/10 text-primary";
    case "valuation":
      return "border-accent/40 bg-accent/10 text-accent";
    case "news":
      return "border-border/60 bg-card/60 text-foreground";
    case "sentiment":
      return "border-warn/40 bg-warn/10 text-warn";
    case "volume":
      return "border-bear/30 bg-bear/10 text-bear";
  }
}
