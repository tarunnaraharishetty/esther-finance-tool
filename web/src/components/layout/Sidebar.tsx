import {
  FileText,
  Gauge,
  GitCompareArrows,
  LayoutDashboard,
  LineChart,
  Newspaper,
  Sparkles,
  Star,
  TrendingUp,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type NavKey =
  | "dashboard"
  | "watchlist"
  | "movers"
  | "news"
  | "ai"
  | "charts"
  | "research"
  | "analyzer"
  | "compare";

interface NavItem {
  key: NavKey;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const PRIMARY_NAV: NavItem[] = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "watchlist", label: "Watchlist", icon: Star },
  { key: "movers", label: "Market Movers", icon: TrendingUp },
  { key: "charts", label: "Charts", icon: LineChart },
  { key: "news", label: "News", icon: Newspaper },
  { key: "ai", label: "AI Insights", icon: Sparkles },
  { key: "research", label: "Research", icon: FileText },
  { key: "analyzer", label: "Analyzer", icon: Gauge },
  { key: "compare", label: "Compare", icon: GitCompareArrows },
];

interface Props {
  active: NavKey;
  onSelect: (key: NavKey) => void;
  collapsed: boolean;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  counts?: Partial<Record<NavKey, number>>;
}

export function Sidebar({
  active,
  onSelect,
  collapsed,
  mobileOpen,
  onCloseMobile,
  counts,
}: Props) {
  return (
    <>
      <div
        onClick={onCloseMobile}
        aria-hidden={!mobileOpen}
        className={cn(
          "fixed inset-0 z-30 bg-background/70 backdrop-blur-sm transition-opacity lg:hidden",
          mobileOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex flex-col border-r border-border/60 transition-[transform,width] duration-200",
          // Subtle vertical gradient so the rail reads as a recessed panel
          "bg-gradient-to-b from-card/90 via-card/70 to-card/90 backdrop-blur-xl",
          "lg:static lg:translate-x-0",
          collapsed ? "lg:w-16" : "lg:w-60",
          "w-60",
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
        )}
      >
        <div
          className={cn(
            "flex h-14 items-center justify-between border-b border-border/60 px-5",
            collapsed && "lg:px-0 lg:justify-center",
          )}
        >
          <BrandMark collapsed={collapsed} />
          <button
            onClick={onCloseMobile}
            aria-label="Close sidebar"
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground lg:hidden"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex-1 space-y-0.5 p-3">
          {PRIMARY_NAV.map((item) => (
            <NavButton
              key={item.key}
              item={item}
              active={active === item.key}
              collapsed={collapsed}
              count={counts?.[item.key]}
              onClick={() => {
                onSelect(item.key);
                onCloseMobile();
              }}
            />
          ))}
        </nav>

        {!collapsed && (
          <div className="mx-3 mb-3 flex items-center gap-2 rounded-md border border-border/40 bg-card/40 px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-bull" />
            <span>paper · decision-support</span>
          </div>
        )}
      </aside>
    </>
  );
}

function BrandMark({ collapsed }: { collapsed: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="relative grid h-8 w-8 place-items-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-glow">
        <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
      </div>
      <div
        className={cn(
          "flex flex-col leading-tight",
          collapsed && "lg:hidden",
        )}
      >
        <span className="text-sm font-semibold tracking-tight">Esther</span>
        <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
          terminal · v0.1
        </span>
      </div>
    </div>
  );
}

interface NavButtonProps {
  item: NavItem;
  active: boolean;
  collapsed: boolean;
  onClick: () => void;
  disabled?: boolean;
  count?: number;
}

function NavButton({
  item,
  active,
  collapsed,
  onClick,
  disabled,
  count,
}: NavButtonProps) {
  const Icon = item.icon;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={collapsed ? item.label : undefined}
      className={cn(
        "group relative flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-all",
        active
          ? "bg-secondary text-foreground shadow-sm"
          : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
        disabled && "cursor-not-allowed opacity-40 hover:bg-transparent",
        collapsed && "lg:justify-center lg:px-0",
      )}
    >
      {active && (
        <span className="absolute left-0 h-5 w-0.5 rounded-r-full bg-gradient-to-b from-primary to-accent" />
      )}
      <Icon className="h-4 w-4 shrink-0" />
      <span className={cn("truncate", collapsed && "lg:hidden")}>
        {item.label}
      </span>
      {!collapsed && count !== undefined && count > 0 && (
        <span
          className={cn(
            "ml-auto rounded-full px-1.5 py-0.5 font-mono text-[10px] font-semibold",
            active
              ? "bg-primary/20 text-primary"
              : "bg-muted text-muted-foreground",
          )}
        >
          {count}
        </span>
      )}
    </button>
  );
}
