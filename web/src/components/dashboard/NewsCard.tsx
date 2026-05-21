import { useState } from "react";
import { ChevronDown, ExternalLink, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { TwitterIcon } from "@/components/ui/TwitterIcon";
import { cn } from "@/lib/utils";

export interface NewsCardModel {
  symbol: string;
  headline: string;
  sentimentScore: number;
  href: string;
  source: string;
  isUrl: boolean;
  aiSummary: string;
}

interface Props {
  item: NewsCardModel;
  onSelectSymbol: (s: string) => void;
}

export function NewsCard({ item, onSelectSymbol }: Props) {
  const [expanded, setExpanded] = useState(false);
  const tone = scoreToTone(item.sentimentScore);
  const twitterHref = `https://twitter.com/search?q=%24${item.symbol}&f=live&src=cashtag_click`;

  return (
    <article
      className={cn(
        "group surface surface-hover overflow-hidden transition-all duration-200",
      )}
    >
      <div className="flex items-start gap-3 p-3.5">
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
            <p className="line-clamp-2 text-sm leading-snug text-foreground/95 transition-colors group-hover:text-foreground">
              {item.headline}
            </p>
          </a>

          <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px]">
            <Badge variant={tone}>{toneLabel(tone)}</Badge>
            <span className="inline-flex items-center gap-1 rounded-full border border-border/40 bg-secondary/30 px-2 py-0.5 font-mono uppercase tracking-wide text-muted-foreground">
              {item.source}
            </span>
            <a
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-muted-foreground transition-colors hover:text-primary"
            >
              <ExternalLink className="h-3 w-3" />
              {item.isUrl ? "source" : "search"}
            </a>
            <a
              href={twitterHref}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-muted-foreground transition-colors hover:text-primary"
              title={`Discuss $${item.symbol} on X`}
            >
              <TwitterIcon className="h-3 w-3" />
              discuss
            </a>
            <button
              type="button"
              onClick={() => setExpanded((e) => !e)}
              aria-expanded={expanded}
              className="ml-auto inline-flex items-center gap-1 rounded-full border border-border/40 bg-secondary/30 px-2 py-0.5 font-mono uppercase tracking-wide text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
            >
              <Sparkles className="h-3 w-3" />
              AI summary
              <ChevronDown
                className={cn(
                  "h-3 w-3 transition-transform",
                  expanded && "rotate-180",
                )}
              />
            </button>
          </div>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-border/40 bg-card/40 px-4 py-3 animate-fade-in-fast">
          <div className="flex items-start gap-2">
            <Sparkles className="mt-0.5 h-3 w-3 shrink-0 text-primary" />
            <p className="text-xs leading-relaxed text-muted-foreground">
              {item.aiSummary}
            </p>
          </div>
        </div>
      )}
    </article>
  );
}

function scoreToTone(score: number): "bull" | "bear" | "warn" {
  if (score > 0.15) return "bull";
  if (score < -0.15) return "bear";
  return "warn";
}

function toneLabel(tone: "bull" | "bear" | "warn"): string {
  return tone === "bull" ? "positive" : tone === "bear" ? "negative" : "neutral";
}
