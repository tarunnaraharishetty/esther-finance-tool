import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Search } from "lucide-react";
import { Watchlist } from "@/components/dashboard/Watchlist";
import { WatchlistEditor } from "@/components/watchlist/WatchlistEditor";
import { Panel } from "@/components/layout/Panel";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
  /** When set, render the editor above the table. ``null`` means
   *  the viewer is anonymous and editing is disabled. */
  editor?: {
    symbols: string[];
    onAdd: (symbol: string) => Promise<void>;
    onRemove: (symbol: string) => Promise<void>;
    busy: boolean;
    resolved: boolean;
  } | null;
}

type Filter = "all" | "buy" | "sell" | "hold";
type SortKey = "confidence" | "score" | "symbol";

export function WatchlistPage({
  snapshot,
  activeSymbol,
  setActiveSymbol,
  editor,
}: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("confidence");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");

  const counts = useMemo(() => {
    const c = { all: snapshot.rows.length, buy: 0, sell: 0, hold: 0 };
    for (const r of snapshot.rows) {
      if (r.action === "buy") c.buy++;
      else if (r.action === "sell") c.sell++;
      else c.hold++;
    }
    return c;
  }, [snapshot.rows]);

  const rows = useMemo(() => {
    const q = query.trim().toUpperCase();
    let out = snapshot.rows.filter(
      (r) =>
        (filter === "all" || r.action === filter) &&
        (q === "" || r.symbol.includes(q)),
    );
    out = [...out].sort((a, b) => {
      const dir = sortDir === "desc" ? -1 : 1;
      if (sortKey === "confidence") return dir * (a.confidence - b.confidence);
      if (sortKey === "score") return dir * (a.combined_score - b.combined_score);
      return dir * a.symbol.localeCompare(b.symbol);
    });
    return out;
  }, [snapshot.rows, query, filter, sortKey, sortDir]);

  return (
    <>
      {editor && (
        <WatchlistEditor
          symbols={editor.symbols}
          onAdd={editor.onAdd}
          onRemove={editor.onRemove}
          busy={editor.busy}
          resolved={editor.resolved}
        />
      )}
      <Panel
        title="Watchlist"
        subtitle={`${rows.length} of ${counts.all} · ${sortKey} ${sortDir}`}
      >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="relative max-w-xs flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter symbols…"
            className="h-8 pl-8 text-xs"
          />
        </div>

        <div className="flex items-center gap-0.5 rounded-md border border-border/60 bg-card/40 p-0.5">
          <FilterChip
            active={filter === "all"}
            onClick={() => setFilter("all")}
            tone="muted"
          >
            All <span className="ml-1 tabular-nums">{counts.all}</span>
          </FilterChip>
          <FilterChip
            active={filter === "buy"}
            onClick={() => setFilter("buy")}
            tone="bull"
          >
            Buy <span className="ml-1 tabular-nums">{counts.buy}</span>
          </FilterChip>
          <FilterChip
            active={filter === "sell"}
            onClick={() => setFilter("sell")}
            tone="bear"
          >
            Sell <span className="ml-1 tabular-nums">{counts.sell}</span>
          </FilterChip>
          <FilterChip
            active={filter === "hold"}
            onClick={() => setFilter("hold")}
            tone="warn"
          >
            Hold <span className="ml-1 tabular-nums">{counts.hold}</span>
          </FilterChip>
        </div>

        <SortToggle
          sortKey={sortKey}
          sortDir={sortDir}
          onCycle={() => {
            const order: SortKey[] = ["confidence", "score", "symbol"];
            const i = order.indexOf(sortKey);
            setSortKey(order[(i + 1) % order.length]);
          }}
          onFlip={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
        />
      </div>

        <Watchlist
          rows={rows}
          activeSymbol={activeSymbol}
          onSelect={setActiveSymbol}
        />
      </Panel>
    </>
  );
}

function FilterChip({
  active,
  tone,
  onClick,
  children,
}: {
  active: boolean;
  tone: "bull" | "bear" | "warn" | "muted";
  onClick: () => void;
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

function SortToggle({
  sortKey,
  sortDir,
  onCycle,
  onFlip,
}: {
  sortKey: SortKey;
  sortDir: "desc" | "asc";
  onCycle: () => void;
  onFlip: () => void;
}) {
  const Icon = sortDir === "desc" ? ArrowDown : ArrowUp;
  return (
    <div className="ml-auto flex items-center gap-0.5 rounded-md border border-border/60 bg-card/40 p-0.5">
      <button
        type="button"
        onClick={onCycle}
        className="rounded-sm px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground"
      >
        sort · {sortKey}
      </button>
      <button
        type="button"
        onClick={onFlip}
        aria-label={`Sort ${sortDir === "desc" ? "ascending" : "descending"}`}
        className="grid h-6 w-6 place-items-center rounded-sm text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground"
      >
        <Icon className="h-3 w-3" />
      </button>
    </div>
  );
}

