import { useEffect, useState } from "react";
import { Clock } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * NYSE clock with a green/red open-or-closed dot. Time is rendered
 * in America/New_York so the trader sees the same wall-clock the
 * tape is keyed off, regardless of their local tz. Updates each
 * second; cheap.
 */
export function MarketClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const { hh, mm, ss, weekday, isOpen } = nyseSnapshot(now);

  return (
    <div
      title={isOpen ? "NYSE — regular hours" : "NYSE — outside regular hours"}
      className="hidden items-center gap-2 rounded-full border border-border/60 bg-secondary/30 px-3 py-1 font-mono text-[11px] md:flex"
    >
      <Clock className="h-3 w-3 text-muted-foreground" />
      <span className="tabular-nums">
        {hh}:{mm}
        <span className="text-muted-foreground">:{ss}</span>
      </span>
      <span className="text-muted-foreground">{weekday}</span>
      <span className="flex items-center gap-1">
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            isOpen ? "bg-bull" : "bg-muted-foreground/60",
          )}
        />
        <span
          className={cn(
            "text-[10px] uppercase tracking-wide",
            isOpen ? "text-bull" : "text-muted-foreground",
          )}
        >
          {isOpen ? "Open" : "Closed"}
        </span>
      </span>
    </div>
  );
}

function nyseSnapshot(date: Date): {
  hh: string;
  mm: string;
  ss: string;
  weekday: string;
  isOpen: boolean;
} {
  // Format in America/New_York. Intl handles DST automatically.
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    weekday: "short",
  });
  const parts = fmt.formatToParts(date);
  const get = (type: string): string =>
    parts.find((p) => p.type === type)?.value ?? "";

  const hh = get("hour");
  const mm = get("minute");
  const ss = get("second");
  const weekday = get("weekday");

  const h = Number(hh);
  const m = Number(mm);
  const minuteOfDay = h * 60 + m;
  const isWeekday = weekday !== "Sat" && weekday !== "Sun";
  const isOpen =
    isWeekday && minuteOfDay >= 9 * 60 + 30 && minuteOfDay < 16 * 60;

  return { hh, mm, ss, weekday, isOpen };
}
