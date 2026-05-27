import { useState, type ReactNode } from "react";
import { Sidebar, type NavKey } from "./Sidebar";
import { TopNav } from "./TopNav";
import { Footer } from "./Footer";
import { MarketTickerStrip } from "./MarketTickerStrip";
import { VerifyEmailBanner } from "./VerifyEmailBanner";
import type { AuthUser } from "@/lib/auth";
import type { StreamStatus } from "@/lib/stream";

interface Props {
  status: StreamStatus;
  isStale: boolean;
  tick: number | null;
  rowCount: number;
  activeNav: NavKey;
  onNavChange: (key: NavKey) => void;
  onSearch?: (query: string) => void;
  onOpenPalette?: () => void;
  navCounts?: Partial<Record<NavKey, number>>;
  /** When set, header renders the real account menu instead of the
   *  legacy placeholder avatar. Wired from App's session state. */
  authUser?: AuthUser | null;
  onLogout?: () => Promise<void>;
  /** Invoked after the user confirms account deletion in UserMenu's
   *  dialog. App.tsx refreshes session state so the LoginPage takes
   *  over. */
  onAccountDeleted?: () => Promise<void> | void;
  children: ReactNode;
}

export function AppShell({
  status,
  isStale,
  tick,
  rowCount,
  activeNav,
  onNavChange,
  onSearch,
  onOpenPalette,
  navCounts,
  authUser,
  onLogout,
  onAccountDeleted,
  children,
}: Props) {
  const [desktopCollapsed, setDesktopCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleToggle = (): void => {
    if (window.matchMedia("(min-width: 1024px)").matches) {
      setDesktopCollapsed((c) => !c);
    } else {
      setMobileOpen((o) => !o);
    }
  };

  return (
    <div className="flex h-full bg-gradient-to-br from-background via-background to-card">
      <Sidebar
        active={activeNav}
        onSelect={(k) => {
          onNavChange(k);
          setMobileOpen(false);
        }}
        collapsed={desktopCollapsed}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
        counts={navCounts}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopNav
          status={status}
          isStale={isStale}
          tick={tick}
          rowCount={rowCount}
          onToggleSidebar={handleToggle}
          onSearch={onSearch}
          onOpenPalette={onOpenPalette}
          authUser={authUser}
          onLogout={onLogout}
          onAccountDeleted={onAccountDeleted}
        />
        <MarketTickerStrip />
        <main className="flex-1 overflow-auto p-4 lg:p-6">
          {authUser && !authUser.email_verified && (
            <VerifyEmailBanner email={authUser.email} />
          )}
          {children}
        </main>
        <Footer />
      </div>
    </div>
  );
}
