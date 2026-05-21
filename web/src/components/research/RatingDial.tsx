import { ratingLabel, ratingTone, type Rating } from "@/lib/research";
import { cn } from "@/lib/utils";

interface Props {
  rating: Rating;
  confidence: number; // 0..1
  size?: "sm" | "md" | "lg";
}

/**
 * Institutional-style rating dial. The five tiers map to a 0..100
 * arc — Strong Sell at 0, Strong Buy at 100 — with the dial
 * highlighting the current bucket. Confidence drives the inner ring
 * intensity so a high-conviction HOLD still reads differently from a
 * low-conviction one.
 */
export function RatingDial({ rating, confidence, size = "md" }: Props) {
  const tone = ratingTone(rating);
  const bucket = bucketFor(rating);
  const dim =
    size === "sm" ? "h-20 w-20" : size === "lg" ? "h-32 w-32" : "h-24 w-24";
  const fontDim =
    size === "sm" ? "text-[9px]" : size === "lg" ? "text-[12px]" : "text-[10px]";
  const angle = -90 + (bucket / 100) * 180; // -90deg = left, +90 = right

  return (
    <div className={cn("relative grid place-items-center", dim)}>
      <svg viewBox="0 0 100 60" className="absolute inset-0 h-full w-full">
        <defs>
          <linearGradient id="rating-track" x1="0" x2="100" y1="0" y2="0" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="hsl(var(--bear))" stopOpacity="0.7" />
            <stop offset="50%" stopColor="hsl(var(--warn))" stopOpacity="0.7" />
            <stop offset="100%" stopColor="hsl(var(--bull))" stopOpacity="0.7" />
          </linearGradient>
        </defs>
        <path
          d="M 8 52 A 42 42 0 0 1 92 52"
          fill="none"
          stroke="url(#rating-track)"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <g transform={`rotate(${angle} 50 52)`}>
          <line
            x1="50"
            y1="52"
            x2="50"
            y2="14"
            stroke="hsl(var(--foreground))"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <circle cx="50" cy="52" r="3" fill="hsl(var(--foreground))" />
        </g>
      </svg>
      <div className="relative mt-5 flex flex-col items-center">
        <span
          className={cn(
            "font-mono font-semibold uppercase tracking-wider",
            fontDim,
            tone === "bull" && "text-bull",
            tone === "bear" && "text-bear",
            tone === "warn" && "text-warn",
          )}
        >
          {ratingLabel(rating)}
        </span>
        <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground tabular-nums">
          {Math.round(confidence * 100)}%
        </span>
      </div>
    </div>
  );
}

function bucketFor(rating: Rating): number {
  switch (rating) {
    case "strong_sell":
      return 8;
    case "sell":
      return 28;
    case "hold":
      return 50;
    case "buy":
      return 72;
    case "strong_buy":
      return 92;
  }
}
