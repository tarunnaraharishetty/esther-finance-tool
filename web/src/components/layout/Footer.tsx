import { ShieldCheck } from "lucide-react";

/** Slim status strip — safety posture + version. */
export function Footer() {
  return (
    <footer className="flex shrink-0 items-center justify-between border-t border-border/40 bg-card/40 px-4 py-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground backdrop-blur-md">
      <div className="flex items-center gap-1.5">
        <ShieldCheck className="h-3 w-3 text-bull/70" />
        <span>paper · no execution</span>
      </div>
      <div className="hidden items-center gap-3 sm:flex">
        <span className="hidden md:inline">FinBERT · Alpaca paper-api</span>
        <span>v0.1.0</span>
      </div>
    </footer>
  );
}
