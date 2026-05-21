import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  title: string;
  eyebrow?: string;
  icon?: React.ComponentType<{ className?: string }>;
  defaultOpen?: boolean;
  /** Renders a chip on the right of the title row. */
  trailing?: ReactNode;
  children: ReactNode;
}

/**
 * Collapsible section card. The visual rhythm of the research page —
 * every block sits in one of these so the report reads like a real
 * dossier rather than a wall of panels.
 */
export function ThesisSection({
  title,
  eyebrow,
  icon: Icon,
  defaultOpen = true,
  trailing,
  children,
}: Props) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="surface overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-3 border-b border-border/40 px-4 py-3 text-left transition-colors hover:bg-card/60"
      >
        {Icon && (
          <div className="grid h-7 w-7 place-items-center rounded-md bg-gradient-to-br from-primary/20 to-accent/10">
            <Icon className="h-3.5 w-3.5 text-primary" />
          </div>
        )}
        <div className="min-w-0 flex-1">
          {eyebrow && (
            <div className="font-mono text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
              {eyebrow}
            </div>
          )}
          <h2 className="font-display text-base font-semibold tracking-tight">
            {title}
          </h2>
        </div>
        {trailing && <div className="shrink-0">{trailing}</div>}
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      {open && <div className="animate-fade-in-fast p-4">{children}</div>}
    </section>
  );
}

/**
 * Common renderer for a `ThesisSection` payload (prose + bullets).
 * Used inside section cards that wrap server-provided text.
 */
export function ProseAndBullets({
  body,
  bullets,
}: {
  body: string;
  bullets: string[];
}) {
  return (
    <div className="space-y-3">
      {body && (
        <div className="space-y-2 text-[13px] leading-relaxed text-foreground/90">
          {body.split(/\n{2,}/).map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </div>
      )}
      {bullets.length > 0 && (
        <ul className="space-y-1.5 border-l border-border/40 pl-3">
          {bullets.map((b, i) => (
            <li key={i} className="text-[13px] leading-relaxed text-muted-foreground">
              {b}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
