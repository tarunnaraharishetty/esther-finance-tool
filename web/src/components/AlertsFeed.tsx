import type { Alert } from "../lib/types";

interface Props {
  alerts: Alert[];
  recentAlerts: Alert[];
  onSelect: (symbol: string) => void;
}

const SEVERITY_CLASS: Record<string, string> = {
  info: "alerts__severity alerts__severity--info",
  warn: "alerts__severity alerts__severity--warn",
  critical: "alerts__severity alerts__severity--crit",
};

function fmtTime(iso: string): string {
  // ISO 8601 → "HH:MM:SS" in viewer's local timezone. The trader is
  // (almost always) parked on US/Eastern; converting client-side
  // means the panel reads correctly wherever it's loaded without a
  // server-side TZ field.
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour12: false });
}

/**
 * Alerts feed — current-tick alerts first (the "just fired this
 * second" set), then `recent_alerts` for context. Clicking a row
 * surfaces the alert's symbol in the detail panel.
 */
export function AlertsFeed({ alerts, recentAlerts, onSelect }: Props) {
  // Dedupe: an alert that just fired appears in both `alerts` and the
  // tail of `recent_alerts`. Show `alerts` once, then any
  // `recent_alerts` not already in the first set.
  const seen = new Set<string>();
  const ordered: { source: "now" | "recent"; alert: Alert }[] = [];
  for (const a of alerts) {
    const key = `${a.symbol}|${a.rule}|${a.fired_at}`;
    if (!seen.has(key)) {
      seen.add(key);
      ordered.push({ source: "now", alert: a });
    }
  }
  for (const a of recentAlerts) {
    const key = `${a.symbol}|${a.rule}|${a.fired_at}`;
    if (!seen.has(key)) {
      seen.add(key);
      ordered.push({ source: "recent", alert: a });
    }
  }

  return (
    <section className="panel">
      <h2 className="panel__title">
        Alerts
        {alerts.length > 0 && <span className="panel__badge">{alerts.length} new</span>}
      </h2>
      <div className="panel__body panel__body--scroll">
        {ordered.length === 0 ? (
          <div className="panel__body--empty">no alerts this session</div>
        ) : (
          <ul className="alerts">
            {ordered.map(({ source, alert }, i) => (
              <li
                key={`${alert.symbol}-${alert.rule}-${alert.fired_at}-${i}`}
                className={`alerts__row alerts__row--${source}`}
                onClick={() => onSelect(alert.symbol)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(alert.symbol);
                  }
                }}
                tabIndex={0}
              >
                <span className="alerts__time">{fmtTime(alert.fired_at)}</span>
                <span className={SEVERITY_CLASS[alert.severity] ?? "alerts__severity"}>
                  {alert.severity.toUpperCase()}
                </span>
                <span className="alerts__sym">{alert.symbol}</span>
                <span className="alerts__rule">{alert.rule}</span>
                <span className="alerts__msg">{alert.message}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
