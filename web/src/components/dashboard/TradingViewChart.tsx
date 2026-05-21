import { useEffect, useId, useRef } from "react";
import { tvSymbol } from "@/lib/tickers";
import type { Timeframe } from "@/lib/chartData";

interface Props {
  symbol: string;
  timeframe: Timeframe;
  height?: number;
}

interface TVWidgetOpts {
  container_id: string;
  symbol: string;
  interval: string;
  autosize: boolean;
  theme: "light" | "dark";
  style: string;
  timezone: string;
  toolbar_bg: string;
  hide_side_toolbar: boolean;
  allow_symbol_change: boolean;
  withdateranges: boolean;
  hide_volume: boolean;
  studies: string[];
  locale: string;
}

interface TradingViewGlobal {
  widget: new (opts: TVWidgetOpts) => unknown;
}

declare global {
  interface Window {
    TradingView?: TradingViewGlobal;
  }
}

const TV_SCRIPT_SRC = "https://s3.tradingview.com/tv.js";
const TIMEFRAME_TO_INTERVAL: Record<Timeframe, string> = {
  "1D": "5",
  "1W": "60",
  "1M": "D",
  "1Y": "W",
};

/**
 * TradingView Advanced Chart embed.
 *
 * Loads tv.js once globally (cached after first mount) and
 * instantiates a fresh widget on every (symbol, timeframe) change.
 * Theme + colors are passed in via widget options so it blends
 * with the rest of the dashboard chrome. TradingView's own data
 * powers it — works for any ticker on their universe, not just
 * symbols on the local watchlist.
 */
export function TradingViewChart({ symbol, timeframe, height = 460 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  // useId gives us a stable container id per component instance —
  // multiple TradingViewCharts on the same page wouldn't collide.
  const rawId = useId();
  const containerId = `tv-${rawId.replace(/:/g, "")}`;

  useEffect(() => {
    let cancelled = false;

    const mount = (): void => {
      if (cancelled || !window.TradingView || !containerRef.current) return;
      // The widget renders into a child div whose id matches
      // `container_id`. We clear the host element first so re-mounts
      // don't stack multiple iframes.
      containerRef.current.innerHTML = "";
      const inner = document.createElement("div");
      inner.id = containerId;
      inner.style.height = "100%";
      inner.style.width = "100%";
      containerRef.current.appendChild(inner);
      try {
        new window.TradingView.widget({
          container_id: containerId,
          symbol: tvSymbol(symbol),
          interval: TIMEFRAME_TO_INTERVAL[timeframe],
          autosize: true,
          theme: "dark",
          style: "1",
          timezone: "America/New_York",
          toolbar_bg: "rgba(8,10,16,0.6)",
          hide_side_toolbar: false,
          allow_symbol_change: true,
          withdateranges: true,
          hide_volume: false,
          studies: ["MASimple@tv-basicstudies"],
          locale: "en",
        });
      } catch (err) {
        console.error("TradingView widget init failed", err);
      }
    };

    if (window.TradingView) {
      mount();
    } else {
      let script = document.querySelector<HTMLScriptElement>(
        `script[src="${TV_SCRIPT_SRC}"]`,
      );
      if (!script) {
        script = document.createElement("script");
        script.src = TV_SCRIPT_SRC;
        script.async = true;
        document.head.appendChild(script);
      }
      script.addEventListener("load", mount, { once: true });
    }

    return () => {
      cancelled = true;
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [symbol, timeframe, containerId]);

  return (
    <div className="overflow-hidden rounded-xl border border-border/40 bg-card/30">
      <div
        ref={containerRef}
        style={{ height }}
        className="relative w-full"
        aria-label={`TradingView chart for ${symbol}`}
      >
        {/* tv.js paints into a child div on mount. Until then this
            container stays empty; the parent shows a Suspense
            fallback via lazy() so the user never sees the void. */}
      </div>
    </div>
  );
}
