import { useEffect, useMemo, useState } from "react";
import { AlertOctagon, AlertTriangle, ArrowLeftRight, RefreshCw, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Panel } from "@/components/layout/Panel";
import { ComparisonHero } from "@/components/compare/ComparisonHero";
import { ComparisonSectionCard } from "@/components/compare/ComparisonSectionCard";
import { NarrativePanel } from "@/components/compare/NarrativePanel";
import { useCompareView } from "@/lib/compare";
import { cn } from "@/lib/utils";
import { lookupTicker } from "@/lib/tickers";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  /** Carried in via the global active symbol so the left side defaults
   * to whatever the user was just looking at. The right side
   * auto-picks a contextually useful counterpart (next symbol on
   * the watchlist) but the user can change either freely. */
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
}

/**
 * Compare-two-stocks workstation.
 *
 * Layout: two stacked symbol pickers in the left rail (one per side),
 * main column shows the comparison hero + four sectioned cards
 * (Trust / Valuation / Technical / Risk). Each metric row carries
 * a winner chip pointing at the leading side.
 *
 * Defaults
 * --------
 * Left side defaults to ``activeSymbol`` (whatever the user was
 * viewing on another page). Right side defaults to the next symbol
 * on the watchlist so the comparison renders something useful on
 * first load — the user can swap either at will.
 *
 * What this is NOT
 * ----------------
 * This page does not call the LLM and does not synthesize prose. The
 * verdicts come from :mod:`src.intelligence.comparison`, which is
 * deterministic. The grounded-AI narrative for compare-two-stocks
 * lands in a follow-up commit; the structured per-metric verdict
 * is the load-bearing piece for trust.
 */
export function ComparePage({
  snapshot,
  activeSymbol,
  setActiveSymbol,
}: Props) {
  const [leftSymbol, setLeftSymbol] = useState<string | null>(activeSymbol);
  const [rightSymbol, setRightSymbol] = useState<string | null>(null);
  const [leftFilter, setLeftFilter] = useState("");
  const [rightFilter, setRightFilter] = useState("");

  useEffect(() => {
    // Keep left side in sync when the user changes the active symbol
    // elsewhere in the app (e.g. via the command palette).
    if (activeSymbol !== null && activeSymbol !== leftSymbol) {
      setLeftSymbol(activeSymbol);
    }
  }, [activeSymbol, leftSymbol]);

  useEffect(() => {
    // Right-side default: first watchlist symbol that isn't the left
    // side. Only fires when the right slot is empty — the user can
    // override and we won't clobber the choice.
    if (rightSymbol !== null) return;
    const candidate = snapshot.rows.find(
      (r) => r.symbol !== (leftSymbol ?? ""),
    );
    if (candidate) setRightSymbol(candidate.symbol);
  }, [snapshot.rows, leftSymbol, rightSymbol]);

  const { view, loading, error, refresh } = useCompareView(
    leftSymbol,
    rightSymbol,
  );

  const filteredLeft = useMemo(
    () => filterRows(snapshot.rows, leftFilter),
    [snapshot.rows, leftFilter],
  );
  const filteredRight = useMemo(
    () => filterRows(snapshot.rows, rightFilter),
    [snapshot.rows, rightFilter],
  );

  return (
    <div className="grid gap-4 lg:grid-cols-12">
      <Panel
        className="lg:col-span-3"
        title="Symbols"
        subtitle="Pick two to compare"
      >
        <div className="space-y-4">
          <PickerColumn
            heading="Left side"
            filter={leftFilter}
            onFilter={setLeftFilter}
            rows={filteredLeft}
            active={leftSymbol}
            disabled={rightSymbol}
            onSelect={(s) => {
              setLeftSymbol(s);
              setActiveSymbol(s);
            }}
          />
          <SwapBar
            onSwap={() => {
              setLeftSymbol(rightSymbol);
              setRightSymbol(leftSymbol);
            }}
            disabled={leftSymbol === null || rightSymbol === null}
          />
          <PickerColumn
            heading="Right side"
            filter={rightFilter}
            onFilter={setRightFilter}
            rows={filteredRight}
            active={rightSymbol}
            disabled={leftSymbol}
            onSelect={setRightSymbol}
          />
        </div>
      </Panel>

      <div className="space-y-4 lg:col-span-9">
        {error && (
          <div className="surface flex items-start gap-3 border-bear/40 p-4">
            <AlertOctagon className="mt-0.5 h-4 w-4 shrink-0 text-bear" />
            <div className="flex-1">
              <div className="font-mono text-[10px] uppercase tracking-wider text-bear">
                Comparison failed
              </div>
              <p className="mt-1 text-sm text-muted-foreground">{error}</p>
              <button
                type="button"
                onClick={() => void refresh({ fresh: true })}
                className="mt-2 inline-flex items-center gap-1 rounded-md border border-border/40 bg-card/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wider hover:bg-card"
              >
                <RefreshCw className="h-3 w-3" /> Retry
              </button>
            </div>
          </div>
        )}

        {!view && !error && (leftSymbol === null || rightSymbol === null) && (
          <EmptyState
            leftMissing={leftSymbol === null}
            rightMissing={rightSymbol === null}
          />
        )}

        {!view && !error && leftSymbol !== null && rightSymbol !== null && (
          <CompareSkeleton />
        )}

        {view && (
          <>
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => void refresh({ fresh: true })}
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-md border border-border/40 bg-card/60 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground hover:bg-card hover:text-foreground disabled:opacity-40"
              >
                <RefreshCw
                  className={loading ? "h-3 w-3 animate-spin" : "h-3 w-3"}
                />
                Refresh
              </button>
            </div>

            <ComparisonHero view={view} />

            <NarrativePanel leftSymbol={leftSymbol} rightSymbol={rightSymbol} />

            {view.warnings.length > 0 && (
              <div className="surface flex items-start gap-3 border-warn/40 p-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
                <div className="flex-1">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-warn">
                    Comparison notes
                  </div>
                  <ul className="mt-1 space-y-0.5 text-[12px] text-muted-foreground">
                    {view.warnings.slice(0, 5).map((w, i) => (
                      <li key={i}>· {w}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            <div className="grid gap-3 md:grid-cols-2">
              {view.sections.map((section) => (
                <ComparisonSectionCard
                  key={section.title}
                  section={section}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function filterRows(rows: RecommendationRow[], query: string): RecommendationRow[] {
  const q = query.trim().toUpperCase();
  if (!q) return rows;
  return rows.filter((r) => r.symbol.includes(q));
}

function PickerColumn({
  heading,
  filter,
  onFilter,
  rows,
  active,
  disabled,
  onSelect,
}: {
  heading: string;
  filter: string;
  onFilter: (v: string) => void;
  rows: RecommendationRow[];
  active: string | null;
  /** Symbol already taken by the *other* side; disable that row so
   * the user can't accidentally pick the same symbol on both sides. */
  disabled: string | null;
  onSelect: (s: string) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          {heading}
        </span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {active ?? "—"}
        </span>
      </div>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={filter}
          onChange={(e) => onFilter(e.target.value)}
          placeholder="Filter…"
          className="h-8 pl-8 text-xs"
        />
      </div>
      {rows.length === 0 ? (
        <div className="rounded-md border border-dashed border-border/40 bg-card/30 px-3 py-3 text-center font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          No matches.
        </div>
      ) : (
        <ul className="max-h-56 space-y-1 overflow-auto pr-1">
          {rows.map((r) => (
            <li key={r.symbol}>
              <SymbolButton
                row={r}
                active={r.symbol === active}
                disabled={r.symbol === disabled}
                onSelect={onSelect}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SymbolButton({
  row,
  active,
  disabled,
  onSelect,
}: {
  row: RecommendationRow;
  active: boolean;
  disabled: boolean;
  onSelect: (s: string) => void;
}) {
  const ticker = lookupTicker(row.symbol);
  return (
    <button
      type="button"
      onClick={() => onSelect(row.symbol)}
      disabled={disabled}
      aria-pressed={active}
      className={cn(
        "relative flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left transition-all duration-150",
        active
          ? "border-primary/50 bg-primary/10 shadow-glow"
          : "border-transparent hover:border-border/60 hover:bg-card/60",
        disabled && "cursor-not-allowed opacity-30 hover:border-transparent hover:bg-transparent",
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="font-mono text-sm font-semibold">{row.symbol}</div>
        {ticker && (
          <div className="truncate font-mono text-[10px] text-muted-foreground">
            {ticker.name}
          </div>
        )}
      </div>
    </button>
  );
}

function SwapBar({
  onSwap,
  disabled,
}: {
  onSwap: () => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onSwap}
      disabled={disabled}
      className="flex w-full items-center justify-center gap-1.5 rounded-md border border-border/40 bg-card/30 py-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground transition-colors hover:bg-card/60 hover:text-foreground disabled:opacity-40"
    >
      <ArrowLeftRight className="h-3 w-3" />
      Swap sides
    </button>
  );
}

function EmptyState({
  leftMissing,
  rightMissing,
}: {
  leftMissing: boolean;
  rightMissing: boolean;
}) {
  const missing = [
    leftMissing && "left",
    rightMissing && "right",
  ]
    .filter(Boolean)
    .join(" & ");
  return (
    <section className="surface p-6 text-center text-sm text-muted-foreground">
      Pick a symbol for the <strong className="text-foreground">{missing}</strong> side to start comparing.
    </section>
  );
}

function CompareSkeleton() {
  return (
    <div className="space-y-4">
      <div className="surface-premium p-5">
        <Skeleton className="h-4 w-32 opacity-60" />
        <Skeleton className="mt-3 h-10 w-2/3 opacity-60" />
        <Skeleton className="mt-3 h-20 w-full opacity-50" />
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="surface p-4">
            <Skeleton className="h-4 w-1/3 opacity-60" />
            <Skeleton className="mt-3 h-32 w-full opacity-50" />
          </div>
        ))}
      </div>
    </div>
  );
}
