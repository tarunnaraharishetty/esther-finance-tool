import { cn } from "@/lib/utils";

interface Props {
  value: number;
  tone: "bull" | "bear" | "warn";
  label?: string;
  size?: "sm" | "md" | "lg";
  showTicks?: boolean;
  className?: string;
}

/**
 * Animated confidence bar with 10 segments. Fills from left, with
 * the leading segment glowing in the tone color. The segmented look
 * reads more "instrumented signal" than a continuous bar.
 */
export function ConfidenceMeter({
  value,
  tone,
  label,
  size = "md",
  showTicks = true,
  className,
}: Props) {
  const clamped = Math.max(0, Math.min(1, value));
  const segments = 10;
  const filled = Math.round(clamped * segments);
  const heightClass =
    size === "sm" ? "h-1.5" : size === "lg" ? "h-3" : "h-2";

  return (
    <div className={className}>
      {label && (
        <div className="mb-1.5 flex items-center justify-between text-[11px] uppercase tracking-wide text-muted-foreground">
          <span>{label}</span>
          <span className="font-mono tabular text-foreground">
            {Math.round(clamped * 100)}%
          </span>
        </div>
      )}
      <div className={cn("flex gap-0.5", heightClass)}>
        {Array.from({ length: segments }).map((_, i) => {
          const isFilled = i < filled;
          const isLeading = i === filled - 1;
          return (
            <div
              key={i}
              className={cn(
                "flex-1 rounded-[2px] transition-all duration-300",
                !isFilled && "bg-muted/60",
                isFilled && tone === "bull" && "bg-bull/80",
                isFilled && tone === "bear" && "bg-bear/80",
                isFilled && tone === "warn" && "bg-warn/80",
                isLeading && tone === "bull" && "shadow-glow-bull",
                isLeading && tone === "bear" && "shadow-glow-bear",
              )}
              style={{ transitionDelay: `${i * 22}ms` }}
            />
          );
        })}
      </div>
      {showTicks && (
        <div className="mt-1 flex justify-between text-[9px] font-mono uppercase tracking-wider text-muted-foreground/60">
          <span>low</span>
          <span>med</span>
          <span>high</span>
        </div>
      )}
    </div>
  );
}
