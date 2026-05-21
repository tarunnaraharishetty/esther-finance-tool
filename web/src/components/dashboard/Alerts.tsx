import { AlertCircle, AlertTriangle, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Alert } from "@/lib/types";

interface Props {
  alerts: Alert[];
  recentAlerts: Alert[];
  onSelect: (symbol: string) => void;
}

export function Alerts({ alerts, recentAlerts, onSelect }: Props) {
  const hasActive = alerts.length > 0;
  const hasRecent = recentAlerts.length > 0;

  if (!hasActive && !hasRecent) {
    return (
      <div className="grid place-items-center rounded-xl border border-dashed border-border/60 bg-card/40 px-6 py-10 text-center">
        <div className="grid h-10 w-10 place-items-center rounded-full bg-secondary/40 text-muted-foreground">
          <Info className="h-4 w-4" />
        </div>
        <p className="mt-3 text-sm text-muted-foreground">No alerts.</p>
        <p className="text-xs text-muted-foreground/70">The room is quiet.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {hasActive && (
        <section>
          <h3 className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping-soft rounded-full bg-bear/60 opacity-60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-bear" />
            </span>
            Active
          </h3>
          <ul className="space-y-2">
            {alerts.map((a, i) => (
              <AlertRow key={`active-${i}`} alert={a} onSelect={onSelect} />
            ))}
          </ul>
        </section>
      )}
      {hasRecent && (
        <section>
          <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Recent
          </h3>
          <ul className="space-y-2 opacity-80">
            {recentAlerts.map((a, i) => (
              <AlertRow
                key={`recent-${i}`}
                alert={a}
                onSelect={onSelect}
                dim
              />
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function AlertRow({
  alert,
  onSelect,
  dim = false,
}: {
  alert: Alert;
  onSelect: (s: string) => void;
  dim?: boolean;
}) {
  const tone = severityTone(alert.severity);
  const Icon = severityIcon(alert.severity);
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(alert.symbol)}
        className={cn(
          "group flex w-full items-start gap-3 rounded-lg border bg-card/40 px-3 py-2 text-left transition-all duration-150 hover:-translate-y-px hover:border-border hover:bg-card hover:shadow-md",
          tone === "critical" && "border-bear/40",
          tone === "warn" && "border-warn/40",
          tone === "info" && "border-border/40",
          dim && "bg-card/20",
        )}
      >
        <Icon
          className={cn(
            "mt-0.5 h-4 w-4 shrink-0",
            tone === "critical" && "text-bear",
            tone === "warn" && "text-warn",
            tone === "info" && "text-primary",
          )}
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-sm font-semibold">
              {alert.symbol}
            </span>
            <Badge
              variant={
                tone === "critical"
                  ? "bear"
                  : tone === "warn"
                    ? "warn"
                    : "outline"
              }
            >
              {alert.severity}
            </Badge>
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              {alert.rule}
            </span>
          </div>
          <p className="mt-1 text-sm text-foreground/90">{alert.message}</p>
          <span className="font-mono text-[10px] text-muted-foreground">
            {formatAlertTime(alert.fired_at)}
          </span>
        </div>
      </button>
    </li>
  );
}

function severityTone(severity: string): "critical" | "warn" | "info" {
  if (severity === "critical") return "critical";
  if (severity === "warn") return "warn";
  return "info";
}

function severityIcon(severity: string) {
  if (severity === "critical") return AlertCircle;
  if (severity === "warn") return AlertTriangle;
  return Info;
}

function formatAlertTime(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}
