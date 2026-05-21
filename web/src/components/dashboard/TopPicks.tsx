import { ArrowUpRight, Crown, Flame, Minus, TrendingDown, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { MiniSpark } from "@/components/ui/MiniSpark";
import { ConfidenceMeter } from "./ConfidenceMeter";
import { generateSymbolSummary } from "@/lib/aiSummary";
import { fmtPrice, tierLabel, tierTone } from "@/lib/format";
import { sparklineFor } from "@/lib/sparkline";
import { cn } from "@/lib/utils";
import type {
  DashboardSnapshot,
  RankedOpportunity,
  RecommendationRow,
} from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  onSelect: (symbol: string) => void;
}

interface PickModel {
  rank: number;
  symbol: string;
  tier: RecommendationRow["tier"];
  tone: "bull" | "bear" | "warn";
  action: RecommendationRow["action"];
  confidence: number;
  lastPrice: number;
  rationale: string;
  composite: number | null;
  isHot: boolean;
}

/**
 * AI-ranked top picks. The #1 spot is visually heroified — crown
 * badge, stronger glow, gradient frame, "Flame" indicator when
 * composite ≥ 0.6 — so the trader's eye lands on the highest-
 * conviction opportunity first.
 */
export function TopPicks({ snapshot, onSelect }: Props) {
  const picks = selectPicks(snapshot);
  if (picks.length === 0) {
    return (
      <div className="grid place-items-center rounded-xl border border-dashed border-border/60 bg-card/40 p-8 text-sm text-muted-foreground">
        No qualifying picks yet — waiting on the engine.
      </div>
    );
  }

  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {picks.map((p, i) => (
        <PickCard key={p.symbol} pick={p} onSelect={onSelect} index={i} />
      ))}
    </div>
  );
}

function PickCard({
  pick,
  onSelect,
  index,
}: {
  pick: PickModel;
  onSelect: (s: string) => void;
  index: number;
}) {
  const Icon =
    pick.tone === "bull" ? TrendingUp : pick.tone === "bear" ? TrendingDown : Minus;
  const gradientClass =
    pick.tone === "bull"
      ? "bg-gradient-bull"
      : pick.tone === "bear"
        ? "bg-gradient-bear"
        : "bg-gradient-to-br from-warn/15 to-warn/0";
  const borderClass =
    pick.tone === "bull"
      ? "border-bull/30"
      : pick.tone === "bear"
        ? "border-bear/30"
        : "border-warn/30";
  const isHero = pick.rank === 0;

  return (
    <Card
      role="button"
      tabIndex={0}
      onClick={() => onSelect(pick.symbol)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(pick.symbol);
        }
      }}
      className={cn(
        "group relative cursor-pointer overflow-hidden transition-all duration-300 ease-out animate-slide-up",
        "hover:-translate-y-1.5 hover:scale-[1.012] hover:shadow-2xl",
        isHero && "ring-1 ring-primary/40 shadow-glow",
        borderClass,
        pick.tone === "bull" && "hover:shadow-glow-bull",
        pick.tone === "bear" && "hover:shadow-glow-bear",
      )}
      style={{ animationDelay: `${index * 80}ms`, animationFillMode: "backwards" }}
    >
      <div
        aria-hidden
        className={cn("absolute inset-0 opacity-90", gradientClass)}
      />
      {isHero && (
        <div
          aria-hidden
          className="pointer-events-none absolute -right-12 -top-12 h-32 w-32 rounded-full bg-primary/20 blur-2xl"
        />
      )}

      <div className="relative p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-1.5">
              {isHero && (
                <span className="inline-flex h-4 items-center gap-1 rounded-sm bg-gradient-to-r from-primary to-accent px-1.5 font-mono text-[9px] font-bold uppercase tracking-wider text-primary-foreground shadow-glow">
                  <Crown className="h-2.5 w-2.5" /> #1
                </span>
              )}
              <span className="font-display text-lg font-bold tracking-tightest">
                {pick.symbol}
              </span>
              <Badge variant={pick.tone}>{tierLabel(pick.tier)}</Badge>
              {pick.isHot && (
                <span title="Composite ≥ 60">
                  <Flame className="h-3 w-3 text-warn animate-pulse" />
                </span>
              )}
            </div>
            <div className="mt-0.5 font-mono text-xs text-muted-foreground tabular">
              ${fmtPrice(pick.lastPrice)}
            </div>
          </div>
          <div
            className={cn(
              "grid h-8 w-8 shrink-0 place-items-center rounded-full border transition-transform group-hover:scale-110",
              borderClass,
              pick.tone === "bull" && "bg-bull/15 text-bull",
              pick.tone === "bear" && "bg-bear/15 text-bear",
              pick.tone === "warn" && "bg-warn/15 text-warn",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
          </div>
        </div>

        <div className="mt-2.5">
          <MiniSpark
            data={sparklineFor(pick.symbol, pick.action, pick.lastPrice)}
            tone={pick.tone}
            height={40}
          />
        </div>

        <div className="mt-2.5">
          <ConfidenceMeter
            value={pick.confidence}
            tone={pick.tone}
            label="Confidence"
            size="sm"
            showTicks={false}
          />
        </div>

        <p className="mt-2.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
          {pick.rationale}
        </p>

        <div className="mt-3 flex items-center justify-between">
          {pick.composite !== null ? (
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              composite {(pick.composite * 100).toFixed(0)}
            </span>
          ) : (
            <span />
          )}
          <span className="flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground transition-colors group-hover:text-foreground">
            Detail
            <ArrowUpRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </span>
        </div>
      </div>
    </Card>
  );
}

function selectPicks(snap: DashboardSnapshot): PickModel[] {
  const rowsBySymbol: Record<string, RecommendationRow> = {};
  for (const r of snap.rows) rowsBySymbol[r.symbol] = r;

  const opps: RankedOpportunity[] = snap.ranked_opportunities.slice(0, 3);
  const picks: PickModel[] = [];
  for (const opp of opps) {
    const row = rowsBySymbol[opp.symbol];
    if (!row) continue;
    picks.push({
      rank: picks.length,
      symbol: row.symbol,
      tier: opp.tier,
      tone: tierTone(opp.tier),
      action: row.action,
      confidence: row.confidence,
      lastPrice: row.last_price,
      rationale: generateSymbolSummary(row),
      composite: opp.composite_score,
      isHot: opp.composite_score >= 0.6,
    });
  }
  if (picks.length >= 3) return picks;

  const taken = new Set(picks.map((p) => p.symbol));
  const fallback = [...snap.rows]
    .filter((r) => !taken.has(r.symbol) && r.action !== "hold")
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3 - picks.length);
  for (const row of fallback) {
    picks.push({
      rank: picks.length,
      symbol: row.symbol,
      tier: row.tier,
      tone: tierTone(row.tier),
      action: row.action,
      confidence: row.confidence,
      lastPrice: row.last_price,
      rationale: generateSymbolSummary(row),
      composite: null,
      isHot: false,
    });
  }
  return picks;
}
