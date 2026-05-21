import { useState } from "react";
import { Bell, Menu, Search, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MarketClock } from "./MarketClock";
import { useUser } from "@/lib/auth";
import type { StreamStatus } from "@/lib/stream";
import { cn } from "@/lib/utils";

interface Props {
  status: StreamStatus;
  isStale: boolean;
  tick: number | null;
  rowCount: number;
  onToggleSidebar: () => void;
  onSearch?: (query: string) => void;
  onOpenPalette?: () => void;
}

export function TopNav({
  status,
  isStale,
  tick,
  rowCount,
  onToggleSidebar,
  onSearch,
  onOpenPalette,
}: Props) {
  const [query, setQuery] = useState("");
  const user = useUser();

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-3 border-b border-border/60 bg-background/60 px-4 backdrop-blur-xl">
      <Button
        variant="ghost"
        size="icon"
        onClick={onToggleSidebar}
        aria-label="Toggle sidebar"
      >
        <Menu className="h-4 w-4" />
      </Button>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (query.trim()) onSearch?.(query.trim().toUpperCase());
        }}
        className="relative max-w-md flex-1"
      >
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => onOpenPalette?.()}
          placeholder="Search ticker · ⌘K for palette"
          className="pl-9 pr-16"
        />
        <button
          type="button"
          onClick={() => onOpenPalette?.()}
          aria-label="Open command palette"
          className="absolute right-2 top-1/2 hidden -translate-y-1/2 select-none items-center gap-1 rounded border border-border/60 bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground md:flex"
        >
          ⌘K
        </button>
      </form>

      <div className="ml-auto flex items-center gap-3">
        <MarketClock />
        <ConnectionPill
          status={status}
          isStale={isStale}
          tick={tick}
          rowCount={rowCount}
        />
        <Button variant="ghost" size="icon" aria-label="Notifications">
          <Bell className="h-4 w-4" />
        </Button>
        <div className="flex items-center gap-2 rounded-full border border-border/60 bg-secondary/40 py-1 pl-1 pr-3 transition-colors hover:border-primary/40">
          <div className="grid h-7 w-7 place-items-center rounded-full bg-gradient-to-br from-primary to-accent shadow-glow">
            <User className="h-3.5 w-3.5 text-primary-foreground" />
          </div>
          <span className="hidden text-xs font-medium md:inline">
            {user.name}
          </span>
        </div>
      </div>
    </header>
  );
}

function ConnectionPill({
  status,
  isStale,
  tick,
  rowCount,
}: {
  status: StreamStatus;
  isStale: boolean;
  tick: number | null;
  rowCount: number;
}) {
  const { label, dotClass, ringClass } = pillTone(status, isStale);
  return (
    <div
      className={cn(
        "hidden items-center gap-2 rounded-full border bg-secondary/30 px-3 py-1 text-xs font-medium md:flex",
        ringClass,
      )}
      title={`Stream: ${status}${isStale ? " · stale" : ""}`}
    >
      <span className="relative flex h-2 w-2">
        <span
          className={cn(
            "absolute inline-flex h-full w-full animate-ping-soft rounded-full opacity-60",
            dotClass,
          )}
        />
        <span
          className={cn(
            "relative inline-flex h-2 w-2 rounded-full",
            dotClass,
          )}
        />
      </span>
      <span>{label}</span>
      {tick !== null && (
        <span className="hidden font-mono text-[10px] tabular text-muted-foreground sm:inline">
          · {rowCount}
        </span>
      )}
    </div>
  );
}

function pillTone(status: StreamStatus, isStale: boolean) {
  if (status === "open" && !isStale)
    return { label: "Live", dotClass: "bg-bull", ringClass: "border-bull/30" };
  if (status === "open" && isStale)
    return { label: "Stale", dotClass: "bg-warn", ringClass: "border-warn/30" };
  if (status === "connecting")
    return {
      label: "Reconnecting",
      dotClass: "bg-warn",
      ringClass: "border-warn/30",
    };
  return { label: "Offline", dotClass: "bg-bear", ringClass: "border-bear/30" };
}
