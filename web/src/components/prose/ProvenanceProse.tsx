import { useMemo } from "react";
import type { ProvenanceMap } from "@/lib/provenance";
import { cn } from "@/lib/utils";

interface Props {
  body: string;
  provenance: ProvenanceMap;
  /** Optional className passed through to the wrapping span. */
  className?: string;
}

/**
 * Renders a prose body with hover-tooltips over numeric tokens that
 * have provenance entries.
 *
 * Shared by every grounded-AI surface: the research thesis sections,
 * the compare narrative sections, and any future LLM surface that
 * emits a {token: source_field} provenance map alongside its prose.
 * We tokenize the body using the SAME regex the backend validators
 * use, wrap any token present in the provenance map in a
 * ``<span title="Sourced from …">``, and render the rest as plain
 * text.
 *
 * Why hover tooltips (not click popovers): the audit story is
 * "every number is traceable in one mouseover." A click would force
 * a focus shift the trader doesn't want mid-read. Tooltips also
 * compose into the existing FreshnessBadge / TrustScoreBadge tooltip
 * pattern.
 */
export function ProvenanceProse({ body, provenance, className }: Props) {
  const segments = useMemo(
    () => tokenize(body, provenance),
    [body, provenance],
  );
  if (!body || body.length === 0) return null;

  return (
    <span className={className} data-testid="provenance-prose">
      {segments.map((seg, i) =>
        seg.kind === "token" ? (
          <span
            key={i}
            data-testid="prose-token"
            data-source={seg.source}
            data-token={seg.text}
            title={`Sourced from ${seg.source}`}
            className={cn(
              "underline decoration-dotted decoration-primary/40 underline-offset-2 cursor-help",
              "hover:decoration-primary",
            )}
          >
            {seg.text}
          </span>
        ) : (
          // The plain text segments are short — fragments are fine and
          // keep the DOM lean.
          <span key={i} data-testid="prose-text">
            {seg.text}
          </span>
        ),
      )}
    </span>
  );
}

type Segment =
  | { kind: "text"; text: string }
  | { kind: "token"; text: string; source: string };

/**
 * Mirror of the backend validators' `_TOKEN_PATTERN` (kept in sync
 * manually across `research_validator.py` and
 * `comparison_narrative_validator.py`). Splits the body into
 * alternating plain-text + numeric-token segments so the renderer
 * can wrap tokens individually.
 *
 * If a backend's regex evolves, update this one too — the contract
 * is "the validators' numeric tokens are the provenance keys" and
 * the frontend must match the same surface forms.
 */
const TOKEN_RE =
  /(\$(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?[KMBT]?|-?\d{1,3}(?:\.\d+)?%|\d+(?:\.\d+)?[xX]|\bQ[1-4](?:\s*\d{2,4})?\b|\b20\d{2}\b|\b\d+\.\d+\b|\b\d{6,}\b)/g;

function tokenize(body: string, provenance: ProvenanceMap): Segment[] {
  if (!body) return [];
  const segments: Segment[] = [];
  let lastEnd = 0;
  for (const match of body.matchAll(TOKEN_RE)) {
    const token = match[0];
    const start = match.index ?? 0;
    if (start > lastEnd) {
      segments.push({ kind: "text", text: body.slice(lastEnd, start) });
    }
    const source = provenance[token];
    if (source) {
      segments.push({ kind: "token", text: token, source });
    } else {
      // Token without provenance (e.g. allowance-only like the current
      // year or a small integer) — render as plain text so the tooltip
      // doesn't promise a source we can't deliver.
      segments.push({ kind: "text", text: token });
    }
    lastEnd = start + token.length;
  }
  if (lastEnd < body.length) {
    segments.push({ kind: "text", text: body.slice(lastEnd) });
  }
  return segments;
}
