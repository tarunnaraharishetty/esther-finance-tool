import { TrendingDown, TrendingUp } from "lucide-react";
import type { BullBearArgument } from "@/lib/research";
import { cn } from "@/lib/utils";
import { ProvenanceProse } from "./ProvenanceProse";

interface Props {
  bull: BullBearArgument[];
  bear: BullBearArgument[];
}

/**
 * Side-by-side bull/bear thesis stacks. Each argument renders as a
 * weighted bar — width = share of the side's total weight — so the
 * eye can compare load-bearing arguments against also-rans without
 * reading every line.
 */
export function BullBearStack({ bull, bear }: Props) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Column title="Bull Thesis" tone="bull" icon={TrendingUp} args={bull} />
      <Column title="Bear Thesis" tone="bear" icon={TrendingDown} args={bear} />
    </div>
  );
}

function Column({
  title,
  tone,
  icon: Icon,
  args,
}: {
  title: string;
  tone: "bull" | "bear";
  icon: React.ComponentType<{ className?: string }>;
  args: BullBearArgument[];
}) {
  return (
    <div
      className={cn(
        "rounded-lg border bg-card/30 p-3.5",
        tone === "bull" ? "border-bull/30" : "border-bear/30",
      )}
    >
      <div className="mb-3 flex items-center gap-2">
        <Icon
          className={cn(
            "h-3.5 w-3.5",
            tone === "bull" ? "text-bull" : "text-bear",
          )}
        />
        <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">
          {title}
        </h3>
        <span className="ml-auto font-mono text-[10px] tabular-nums text-muted-foreground">
          {args.length} arg{args.length === 1 ? "" : "s"}
        </span>
      </div>
      {args.length === 0 ? (
        <p className="rounded-md border border-dashed border-border/40 bg-card/30 px-3 py-2 text-xs text-muted-foreground">
          Nothing on this side from the current signal stack.
        </p>
      ) : (
        <ul className="space-y-2.5">
          {args.map((a) => (
            <li key={a.label}>
              <Argument arg={a} tone={tone} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Argument({
  arg,
  tone,
}: {
  arg: BullBearArgument;
  tone: "bull" | "bear";
}) {
  const pct = Math.max(6, Math.round(arg.weight * 100));
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs font-semibold">{arg.label}</span>
        <span
          className={cn(
            "font-mono text-[10px] tabular-nums",
            tone === "bull" ? "text-bull" : "text-bear",
          )}
        >
          {pct}%
        </span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted/60">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            tone === "bull"
              ? "bg-gradient-to-r from-bull/60 to-bull"
              : "bg-gradient-to-r from-bear/60 to-bear",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1 text-[12px] leading-relaxed text-muted-foreground">
        <ProvenanceProse body={arg.detail} provenance={arg.provenance} />
      </p>
    </div>
  );
}
