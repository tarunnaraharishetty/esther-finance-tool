import { useId, useState } from "react";
import { ChevronDown, ShieldCheck } from "lucide-react";
import {
  componentLabel,
  gradeTone,
  type TrustComponent,
  type TrustScore,
} from "@/lib/trust";
import { cn } from "@/lib/utils";

interface Props {
  trust: TrustScore;
  variant?: "header" | "inline";
}

/**
 * Composite report-quality grade chip.
 *
 * Renders a clickable pill that shows a one-glance ``grade · score``
 * read. Clicking expands the per-component breakdown so the trader can
 * audit *why* the report earned the grade — which signal moved the
 * needle, which was missing, which was n/a.
 *
 * Framing rules codified here:
 *
 *  * ``A+`` / ``A`` use the bull palette — institutional-grade.
 *  * ``B`` / ``C`` use the warn palette — usable but visible caveats.
 *  * ``D`` / ``F`` use the bear palette — surface a degraded-mode
 *    warning even if the user expands nothing.
 *  * Component contributions are real (post-renormalization weights),
 *    not raw weights — what you see is what landed on the score.
 *  * Missing / n/a components are shown but greyed out; their detail
 *    string explains why. We don't hide them.
 */
export function TrustScoreBadge({ trust, variant = "header" }: Props) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const tone = gradeTone(trust.grade);

  return (
    <div className={cn("font-mono text-[10px] uppercase tracking-wider")}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        aria-expanded={open}
        aria-controls={panelId}
        data-testid="trust-score-badge"
        data-grade={trust.grade}
        data-score={Math.round(trust.score)}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 transition-colors",
          tone === "bull" && "border-bull/40 bg-bull/15 text-bull hover:bg-bull/20",
          tone === "warn" && "border-warn/40 bg-warn/15 text-warn hover:bg-warn/20",
          tone === "bear" && "border-bear/40 bg-bear/15 text-bear hover:bg-bear/20",
        )}
        title="Composite report trust grade. Click to see what moved it."
      >
        <ShieldCheck className="h-3 w-3" />
        <span className="font-semibold">{trust.grade}</span>
        <span className="opacity-70">·</span>
        <span>trust {Math.round(trust.score)}/100</span>
        <ChevronDown
          className={cn(
            "h-3 w-3 transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <div
          id={panelId}
          data-testid="trust-score-breakdown"
          className={cn(
            "mt-2 space-y-1 rounded-md border border-border/50 bg-card/70 p-3 normal-case tracking-normal",
            variant === "header" ? "min-w-[280px]" : "w-full",
          )}
        >
          <div className="mb-2 flex items-center justify-between gap-2 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
            <span>Trust composition</span>
            <span>weight · contribution</span>
          </div>
          {trust.components.map((c) => (
            <ComponentRow key={c.name} component={c} />
          ))}
        </div>
      )}
    </div>
  );
}

function ComponentRow({ component: c }: { component: TrustComponent }) {
  const isPresent = c.value !== null;
  const statusTone =
    c.status === "ok"
      ? "bull"
      : c.status === "warn"
        ? "warn"
        : c.status === "missing"
          ? "bear"
          : "muted";
  return (
    <div
      data-testid={`trust-row-${c.name}`}
      data-status={c.status}
      className="grid grid-cols-[1fr_auto] items-start gap-x-3 gap-y-1 border-t border-border/30 pt-2 first:border-t-0 first:pt-0"
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-[11px] text-foreground">
          <StatusDot tone={statusTone} />
          <span>{componentLabel(c.name)}</span>
          {!isPresent && (
            <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
              · {c.status}
            </span>
          )}
        </div>
        <p className="mt-0.5 pl-3.5 text-[10px] leading-snug text-muted-foreground">
          {c.detail}
        </p>
      </div>
      <div className="whitespace-nowrap text-right font-mono text-[10px] text-muted-foreground">
        <div>{formatWeight(c.weight)}</div>
        <div
          className={cn(
            "text-[11px]",
            statusTone === "bull" && "text-bull",
            statusTone === "warn" && "text-warn",
            statusTone === "bear" && "text-bear",
            statusTone === "muted" && "text-muted-foreground",
          )}
        >
          {c.contribution === null ? "—" : `+${c.contribution.toFixed(1)}`}
        </div>
      </div>
    </div>
  );
}

function StatusDot({ tone }: { tone: "bull" | "warn" | "bear" | "muted" }) {
  return (
    <span
      className={cn(
        "inline-block h-1.5 w-1.5 rounded-full",
        tone === "bull" && "bg-bull",
        tone === "warn" && "bg-warn",
        tone === "bear" && "bg-bear",
        tone === "muted" && "bg-muted-foreground/50",
      )}
    />
  );
}

function formatWeight(w: number): string {
  return `w ${w.toFixed(2)}`;
}
