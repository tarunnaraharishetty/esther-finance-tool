import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { NewsFeed } from "@/components/dashboard/NewsFeed";
import { Panel } from "@/components/layout/Panel";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  setActiveSymbol: (s: string) => void;
}

type Sentiment = "all" | "bull" | "bear" | "neutral";

export function NewsPage({ snapshot, setActiveSymbol }: Props) {
  const [query, setQuery] = useState("");
  const [sentiment, setSentiment] = useState<Sentiment>("all");

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    const rows = snapshot.rows.filter((r) => {
      if (q !== "" && !r.symbol.includes(q)) return false;
      if (sentiment === "bull" && r.sentiment_score <= 0.15) return false;
      if (sentiment === "bear" && r.sentiment_score >= -0.15) return false;
      if (
        sentiment === "neutral" &&
        (r.sentiment_score > 0.15 || r.sentiment_score < -0.15)
      ) {
        return false;
      }
      return r.headlines.length > 0;
    });
    return { ...snapshot, rows };
  }, [snapshot, query, sentiment]);

  const totalHeadlines = filtered.rows.reduce(
    (s, r) => s + r.headlines.length,
    0,
  );

  return (
    <Panel
      title="News Feed"
      subtitle={`${totalHeadlines} headlines · ${filtered.rows.length} symbols`}
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="relative max-w-xs flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by symbol…"
            className="h-8 pl-8 text-xs"
          />
        </div>
        <div className="flex items-center gap-0.5 rounded-md border border-border/60 bg-card/40 p-0.5">
          <Chip active={sentiment === "all"} onClick={() => setSentiment("all")}>
            All
          </Chip>
          <Chip
            active={sentiment === "bull"}
            onClick={() => setSentiment("bull")}
            tone="bull"
          >
            Positive
          </Chip>
          <Chip
            active={sentiment === "bear"}
            onClick={() => setSentiment("bear")}
            tone="bear"
          >
            Negative
          </Chip>
          <Chip
            active={sentiment === "neutral"}
            onClick={() => setSentiment("neutral")}
            tone="warn"
          >
            Neutral
          </Chip>
        </div>
      </div>

      <NewsFeed snapshot={filtered} onSelectSymbol={setActiveSymbol} />
    </Panel>
  );
}

function Chip({
  active,
  onClick,
  tone,
  children,
}: {
  active: boolean;
  onClick: () => void;
  tone?: "bull" | "bear" | "warn";
  children: React.ReactNode;
}) {
  const activeTone =
    tone === "bull"
      ? "bg-bull/15 text-bull"
      : tone === "bear"
        ? "bg-bear/15 text-bear"
        : tone === "warn"
          ? "bg-warn/15 text-warn"
          : "bg-secondary text-foreground";
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "rounded-sm px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider transition-colors",
        active
          ? activeTone
          : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}
