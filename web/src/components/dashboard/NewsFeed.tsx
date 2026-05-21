import { Newspaper } from "lucide-react";
import { NewsCard, type NewsCardModel } from "./NewsCard";
import { ImportantNews } from "./ImportantNews";
import { generateSymbolSummary } from "@/lib/aiSummary";
import type { DashboardSnapshot, RecommendationRow } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  onSelectSymbol: (symbol: string) => void;
}

export function NewsFeed({ snapshot, onSelectSymbol }: Props) {
  const items = collectNews(snapshot.rows).slice(0, 12);

  if (items.length === 0) {
    return (
      <div className="grid place-items-center rounded-xl border border-dashed border-border/60 bg-card/40 px-6 py-10 text-center">
        <Newspaper className="mb-3 h-6 w-6 text-muted-foreground/50" />
        <p className="text-sm text-muted-foreground">
          No headlines yet on the watchlist.
        </p>
        <p className="mt-1 text-xs text-muted-foreground/70">
          New articles surface here as the ingestion pipeline scores them.
        </p>
      </div>
    );
  }

  return (
    <div>
      <ImportantNews snapshot={snapshot} onSelectSymbol={onSelectSymbol} />

      <div className="mb-2 flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          All Headlines
        </span>
        <div className="h-px flex-1 bg-border/40" />
      </div>

      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={`${item.symbol}-${i}`} className="animate-fade-in">
            <NewsCard item={item} onSelectSymbol={onSelectSymbol} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function collectNews(rows: RecommendationRow[]): NewsCardModel[] {
  const out: NewsCardModel[] = [];
  let maxLen = 0;
  for (const r of rows) maxLen = Math.max(maxLen, r.headlines.length);
  for (let i = 0; i < maxLen; i++) {
    for (const r of rows) {
      const h = r.headlines[i];
      if (!h) continue;
      const link = resolveLink(h);
      out.push({
        symbol: r.symbol,
        headline: h,
        sentimentScore: r.sentiment_score,
        href: link.href,
        source: link.source,
        isUrl: link.isUrl,
        aiSummary: generateSymbolSummary(r),
      });
    }
  }
  return out;
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
