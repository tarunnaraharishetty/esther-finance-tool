import { Activity, Compass } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { MarketPulse, PulseEvolution } from "@/lib/types";

interface Props {
  pulse: MarketPulse | null;
  evolution: PulseEvolution | null;
}

/**
 * Pulse card. Categorical badges, gradient breadth bars, regime
 * + pattern chips, and STRONG-signal callouts. Test contract:
 * `data-testid="pulse-strong"` + `data-direction="buy" | "sell"`
 * per chip. Empty state text is "no pulse yet".
 */
export function Pulse({ pulse, evolution }: Props) {
  if (!pulse) {
    return (
      <div className="grid h-full place-items-center py-6 text-sm text-muted-foreground">
        no pulse yet
      </div>
    );
  }

  const sentimentTone = pulseTone(pulse.sentiment);

  return (
    <div className="space-y-3.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant={sentimentTone}>{pulse.sentiment}</Badge>
        <Badge variant="outline">{pulse.conviction}</Badge>
        <Badge variant="outline">{pulse.activity}</Badge>
      </div>

      <p className="text-[13px] leading-relaxed text-muted-foreground">
        {pulse.summary}
      </p>

      <div className="grid grid-cols-3 gap-2">
        <MetricCell label="Long" value={pulse.bullish_count.toString()} tone="bull" />
        <MetricCell label="Short" value={pulse.bearish_count.toString()} tone="bear" />
        <MetricCell
          label="Alerts"
          value={pulse.alert_intensity.toString()}
          tone={pulse.alert_intensity > 0 ? "warn" : undefined}
        />
      </div>

      <div className="space-y-3">
        <BreadthBar
          label="Momentum breadth"
          value={pulse.momentum_breadth}
          tone="bull"
        />
        <BreadthBar
          label="Sentiment breadth"
          value={pulse.sentiment_breadth}
          tone={pulse.sentiment_breadth >= 0.5 ? "bull" : "warn"}
        />
      </div>

      {evolution && evolution.regime !== "indeterminate" && (
        <div className="rounded-lg border border-border/40 bg-card/30 p-3 transition-colors hover:border-border/60">
          <div className="mb-2 flex items-center gap-2">
            <Compass className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
              regime · {evolution.regime}
            </span>
          </div>
          {evolution.patterns.length > 0 && (
            <ul className="flex flex-wrap gap-1.5">
              {evolution.patterns.map((p) => (
                <li key={p.name}>
                  <Badge variant="outline" title={p.detail}>
                    {p.label}
                  </Badge>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {pulse.strongest_symbols.length > 0 && (
        <div data-testid="pulse-strong" className="space-y-2">
          <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-muted-foreground">
            <Activity className="h-3.5 w-3.5" />
            Strong signals
          </div>
          <ul className="flex flex-wrap gap-1.5">
            {pulse.strongest_symbols.map(([symbol, display]) => {
              const direction = display.includes("BUY") ? "buy" : "sell";
              return (
                <li
                  key={symbol}
                  data-direction={direction}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-transform hover:-translate-y-px",
                    direction === "buy"
                      ? "border-bull/40 bg-bull/10 text-bull"
                      : "border-bear/40 bg-bear/10 text-bear",
                  )}
                >
                  <span className="font-mono text-[10px] uppercase tracking-wider opacity-80">
                    {display}
                  </span>
                  <span className="font-mono font-semibold">{symbol}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function pulseTone(sentiment: string): "bull" | "bear" | "warn" | "default" {
  if (sentiment === "bullish") return "bull";
  if (sentiment === "bearish") return "bear";
  if (sentiment === "mixed") return "warn";
  return "default";
}

function MetricCell({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "bull" | "bear" | "warn";
}) {
  return (
    <div className="rounded-md border border-border/40 bg-card/40 px-2.5 py-1.5 transition-colors hover:border-border/60">
      <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "font-mono text-sm font-semibold tabular",
          tone === "bull" && "text-bull",
          tone === "bear" && "text-bear",
          tone === "warn" && "text-warn",
        )}
      >
        {value}
      </div>
    </div>
  );
}

function BreadthBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "bull" | "warn";
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono tabular text-foreground">
          {pct.toFixed(0)}%
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/60">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            tone === "bull"
              ? "bg-gradient-to-r from-bull/70 to-bull"
              : "bg-gradient-to-r from-warn/70 to-warn",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
