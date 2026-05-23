import type { OutlookEntry } from "@/lib/research";
import { cn } from "@/lib/utils";
import { ProvenanceProse } from "./ProvenanceProse";

interface Props {
  outlook: OutlookEntry[];
}

/**
 * Three-column outlook grid (short / medium / long). Each card shows
 * the bias, confidence bar, and a one-line rationale so the trader
 * can sanity-check the time-horizon framing without expanding text.
 */
export function OutlookTriad({ outlook }: Props) {
  if (outlook.length === 0) {
    return (
      <p className="rounded-md border border-dashed border-border/40 bg-card/30 px-3 py-2 text-xs text-muted-foreground">
        No outlook provided.
      </p>
    );
  }
  return (
    <div className="grid gap-3 md:grid-cols-3">
      {outlook.map((entry) => (
        <OutlookCard key={entry.horizon} entry={entry} />
      ))}
    </div>
  );
}

function OutlookCard({ entry }: { entry: OutlookEntry }) {
  const tone = biasTone(entry.bias);
  const labels: Record<string, string> = {
    short: "Short Term",
    medium: "Medium Term",
    long: "Long Term",
  };
  const subtitle: Record<string, string> = {
    short: "1–5 sessions",
    medium: "1–4 weeks",
    long: "3–12 months",
  };
  return (
    <div
      className={cn(
        "rounded-lg border bg-card/40 p-3",
        tone === "bull" && "border-bull/30",
        tone === "bear" && "border-bear/30",
        tone === "warn" && "border-warn/30",
      )}
    >
      <div className="flex items-baseline justify-between">
        <div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
            {labels[entry.horizon] ?? entry.horizon}
          </div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            {subtitle[entry.horizon] ?? ""}
          </div>
        </div>
        <span
          className={cn(
            "rounded-md border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider",
            tone === "bull" && "border-bull/40 bg-bull/10 text-bull",
            tone === "bear" && "border-bear/40 bg-bear/10 text-bear",
            tone === "warn" && "border-warn/40 bg-warn/10 text-warn",
          )}
        >
          {entry.bias}
        </span>
      </div>
      <div className="mt-2">
        <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>Confidence</span>
          <span className="font-mono tabular-nums">
            {Math.round(entry.confidence * 100)}%
          </span>
        </div>
        <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-muted/60">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              tone === "bull" && "bg-gradient-to-r from-bull/60 to-bull",
              tone === "bear" && "bg-gradient-to-r from-bear/60 to-bear",
              tone === "warn" && "bg-gradient-to-r from-warn/60 to-warn",
            )}
            style={{ width: `${Math.max(4, Math.round(entry.confidence * 100))}%` }}
          />
        </div>
      </div>
      <p className="mt-2 text-[12px] leading-relaxed text-muted-foreground">
        <ProvenanceProse body={entry.detail} provenance={entry.provenance} />
      </p>
    </div>
  );
}

function biasTone(bias: string): "bull" | "bear" | "warn" {
  if (bias === "bullish") return "bull";
  if (bias === "bearish") return "bear";
  return "warn";
}
