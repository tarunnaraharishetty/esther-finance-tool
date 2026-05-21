import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface PanelProps {
  title: string;
  subtitle?: string;
  className?: string;
  children: ReactNode;
}

/** Page-level card with header strip + subtitle slot. */
export function Panel({ title, subtitle, className, children }: PanelProps) {
  return (
    <section
      className={cn(
        "surface surface-hover flex h-full flex-col overflow-hidden",
        className,
      )}
    >
      <header className="flex items-baseline justify-between border-b border-border/40 px-4 py-3">
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
        {subtitle && (
          <span className="font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
            {subtitle}
          </span>
        )}
      </header>
      <div className="flex-1 overflow-auto p-4">{children}</div>
    </section>
  );
}

interface SectionProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
}

/**
 * Editorial-style section header. Gradient accent rail at the start
 * + title + uppercase-mono eyebrow + horizontal hair line filling
 * the remaining width. Used for grouping card grids (Top Picks etc.)
 * where the cards already carry visual weight.
 */
export function Section({ title, subtitle, children }: SectionProps) {
  return (
    <section>
      <div className="mb-4 flex items-center gap-3">
        <span className="section-rail" />
        <h2 className="font-display text-base font-semibold tracking-tight">
          {title}
        </h2>
        {subtitle && (
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            {subtitle}
          </span>
        )}
        <div className="h-px flex-1 bg-gradient-to-r from-border/60 to-transparent" />
      </div>
      {children}
    </section>
  );
}
