import { AiSummary } from "@/components/dashboard/AiSummary";
import { TopPicks } from "@/components/dashboard/TopPicks";
import { Watchlist } from "@/components/dashboard/Watchlist";
import { Pulse } from "@/components/dashboard/Pulse";
import { MarketMovers } from "@/components/dashboard/MarketMovers";
import { NewsFeed } from "@/components/dashboard/NewsFeed";
import { SymbolDetail } from "@/components/dashboard/SymbolDetail";
import { Alerts } from "@/components/dashboard/Alerts";
import { Panel, Section } from "@/components/layout/Panel";
import type { DashboardSnapshot } from "@/lib/types";

interface Props {
  snapshot: DashboardSnapshot;
  activeSymbol: string | null;
  setActiveSymbol: (s: string) => void;
}

export function DashboardPage({
  snapshot,
  activeSymbol,
  setActiveSymbol,
}: Props) {
  return (
    <div className="space-y-5">
      <AiSummary snapshot={snapshot} />

      <Section
        title="Top Picks"
        subtitle={`${snapshot.ranked_opportunities.length} ranked · composite`}
      >
        <TopPicks snapshot={snapshot} onSelect={setActiveSymbol} />
      </Section>

      <div className="grid gap-3 lg:grid-cols-12">
        <Panel
          className="lg:col-span-8"
          title="Watchlist"
          subtitle={`${snapshot.rows.length} · live`}
        >
          <Watchlist
            rows={snapshot.rows}
            activeSymbol={activeSymbol}
            onSelect={setActiveSymbol}
          />
        </Panel>
        <Panel
          className="lg:col-span-4"
          title="Market Pulse"
          subtitle={snapshot.pulse?.sentiment ?? "—"}
        >
          <Pulse pulse={snapshot.pulse} evolution={snapshot.pulse_evolution} />
        </Panel>
      </div>

      <Panel title={activeSymbol ?? "Symbol"} subtitle="detail · ai · stats">
        <SymbolDetail snapshot={snapshot} symbol={activeSymbol} />
      </Panel>

      <div className="grid gap-3 lg:grid-cols-12">
        <Panel
          className="lg:col-span-4"
          title="Movers"
          subtitle="gainers · losers"
        >
          <MarketMovers snapshot={snapshot} onSelect={setActiveSymbol} />
        </Panel>
        <Panel
          className="lg:col-span-5"
          title="News"
          subtitle={`${snapshot.rows.reduce((s, r) => s + r.headlines.length, 0)} headlines`}
        >
          <NewsFeed snapshot={snapshot} onSelectSymbol={setActiveSymbol} />
        </Panel>
        <Panel
          className="lg:col-span-3"
          title="Alerts"
          subtitle={`${snapshot.alerts.length} active`}
        >
          <Alerts
            alerts={snapshot.alerts}
            recentAlerts={snapshot.recent_alerts}
            onSelect={setActiveSymbol}
          />
        </Panel>
      </div>
    </div>
  );
}
