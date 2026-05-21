import { useMemo } from "react";
import { ExternalLink, Flame, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { TwitterIcon } from "@/components/ui/TwitterIcon";
import { importanceFor, whyItMatters } from "@/lib/newsImportance";
import { actionTone } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

interface ImportantItem {
  symbol: string;
  headline: string;
  href: string;
  source: string;
  isUrl: boolean;
  score: number;
  reasons: string[];
  row: RecommendationRow;
}

interface Props {
  snapshot: DashboardSnapshot;
  onSelectSymbol: (s: string) => void;
}

/**
 * Pinned "Important News" section at the top of the News Feed.
 *
 * Each headline is scored by `importanceFor`; anything ≥ 0.55 is
 * surfaced here with a "why it matters" rationale derived from the
 * same heuristic. Limited to the top 3 so the section doesn't
 * dominate the panel.
 */
export function ImportantNews({ snapshot, onSelectSymbol }: Props) {
  const items = useMemo(() => {
    const all: ImportantItem[] = [];
    for (const row of snapshot.rows) {
      for (const headline of row.headlines) {
        const imp = importanceFor(headline, row);
        if (imp.score < 0.55) continue;
        const link = resolveLink(headline);
        all.push({
          symbol: row.symbol,
          headline,
          href: link.href,
          source: link.source,
          isUrl: link.isUrl,
          score: imp.score,
          reasons: imp.reasons,
          row,
        });
      }
    }
    all.sort((a, b) => b.score - a.score);
    return all.slice(0, 3);
  }, [snapshot]);

  if (items.length === 0) return null;

  return (
    <section className="mb-4">
      <header className="mb-2 flex items-center gap-2">
        <div className="grid h-6 w-6 place-items-center rounded-md bg-gradient-to-br from-warn to-bear shadow-glow-bear">
          <Flame className="h-3 w-3 text-primary-foreground" />
        </div>
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          Important News
        </span>
        <Badge variant="warn" className="ml-auto">
          {items.length} flagged
        </Badge>
      </header>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={`imp-${i}`} className="animate-fade-in">
            <ImportantCard item={item} onSelectSymbol={onSelectSymbol} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function ImportantCard({
  item,
  onSelectSymbol,
}: {
  item: ImportantItem;
  onSelectSymbol: (s: string) => void;
}) {
  const tone = actionTone(item.row.action);
  const twitterHref = `https://twitter.com/search?q=%24${item.symbol}&f=live&src=cashtag_click`;
  const why = whyItMatters({ score: item.score, reasons: item.reasons });

  return (
    <article
      className={cn(
        "surface-premium overflow-hidden border-l-4",
        tone === "bull" && "border-l-bull",
        tone === "bear" && "border-l-bear",
        tone === "warn" && "border-l-warn",
      )}
    >
      <div className="p-4">
        <div className="flex items-start gap-3">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectSymbol(item.symbol);
            }}
            className="shrink-0 rounded-md border border-border/50 bg-secondary/40 px-2 py-1 font-mono text-[11px] font-semibold tracking-tight transition-all hover:border-primary/40 hover:bg-primary/10 hover:text-primary"
          >
            {item.symbol}
          </button>
          <div className="min-w-0 flex-1">
            <a
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              className="block"
            >
              <h3 className="text-sm font-semibold leading-snug text-foreground transition-colors hover:text-primary">
                {item.headline}
              </h3>
            </a>
            <div className="mt-1.5 flex items-start gap-2 rounded-md bg-card/40 px-2.5 py-1.5">
              <Sparkles className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
              <div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-warn">
                  Why it matters
                </span>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  {why}
                </p>
              </div>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              {item.reasons.slice(0, 3).map((r) => (
                <Badge key={r} variant="warn" className="lowercase tracking-normal">
                  {r}
                </Badge>
              ))}
              <span className="ml-1 inline-flex items-center gap-1 rounded-full border border-border/40 bg-secondary/30 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                {item.source}
              </span>
              <span
                className="inline-flex items-center gap-1 rounded-full border border-border/40 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground"
                title={`Importance score ${Math.round(item.score * 100)}`}
              >
                <Flame className="h-2.5 w-2.5 text-warn" />
                {Math.round(item.score * 100)}
              </span>
              <a
                href={item.href}
                target="_blank"
                rel="noopener noreferrer"
                className="ml-auto inline-flex items-center gap-1 rounded-full border border-border/40 bg-secondary/30 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
              >
                <ExternalLink className="h-3 w-3" />
                {item.isUrl ? "source" : "search"}
              </a>
              <a
                href={twitterHref}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-full border border-border/40 bg-secondary/30 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
                title={`Discuss $${item.symbol} on X`}
              >
                <TwitterIcon className="h-3 w-3" />
                discuss
              </a>
            </div>
          </div>
        </div>
      </div>
    </article>
  );
}

function resolveLink(headline: string): {
  href: string;
  source: string;
  isUrl: boolean;
} {
  const urlMatch = headline.match(/https?:\/\/(\S+)/);
  if (urlMatch) {
    const href = `https://${urlMatch[1]}`;
    const source = extractDomain(urlMatch[1]);
    return { href, source, isUrl: true };
  }
  const q = encodeURIComponent(headline);
  return {
    href: `https://news.google.com/search?q=${q}`,
    source: "Google News",
    isUrl: false,
  };
}

function extractDomain(urlBody: string): string {
  const host = urlBody.split("/")[0];
  const cleaned = host.replace(/^www\./, "");
  const root = cleaned.split(".")[0];
  return root.charAt(0).toUpperCase() + root.slice(1);
}
