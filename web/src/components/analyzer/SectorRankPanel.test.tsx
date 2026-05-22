import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { SectorRankPanel } from "./SectorRankPanel";
import type {
  SectorMetricRank,
  SectorRankAvailable,
  SectorRankResponse,
  SectorRankUnavailable,
} from "@/lib/sectorRank";

function metric(over: Partial<SectorMetricRank> = {}): SectorMetricRank {
  return {
    metric: "trust_score",
    label: "Trust score",
    value: 92,
    rank: 1,
    cohort_size: 4,
    percentile: 100,
    direction: "higher_better",
    percent: false,
    ...over,
  };
}

function available(
  over: Partial<SectorRankAvailable> = {},
): SectorRankAvailable {
  return {
    symbol: "AAPL",
    sector: "Information Technology",
    cohort_size: 4,
    cohort_symbols: ["AAPL", "AMD", "GOOGL", "MSFT"],
    metrics: [metric()],
    available: true,
    cache: "miss",
    ...over,
  };
}

function unavailable(
  over: Partial<SectorRankUnavailable> = {},
): SectorRankUnavailable {
  return {
    symbol: "AAPL",
    sector: "Information Technology",
    cohort_size: 0,
    cohort_symbols: [],
    metrics: [],
    available: false,
    cache: "miss",
    reason: "no watchlist peers share this sector (Information Technology)",
    ...over,
  };
}

function mockFetchOk(payload: SectorRankResponse): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => payload,
      statusText: "OK",
    }),
  );
}

function mockFetchStatus(status: number): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      status,
      json: async () => ({ detail: "nope" }),
      statusText: "NOPE",
    }),
  );
}

describe("SectorRankPanel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns null when no symbol is selected", () => {
    const { container } = render(<SectorRankPanel symbol={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when the endpoint errors (analyzer surface owns errors)", async () => {
    mockFetchStatus(500);
    const { container } = render(<SectorRankPanel symbol="AAPL" />);
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("renders the unavailable state with the backend reason", async () => {
    mockFetchOk(unavailable());
    render(<SectorRankPanel symbol="AAPL" />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("sector-rank-panel");
      // Wait past the loading state; render is async via the hook.
      expect(p.getAttribute("data-state")).not.toBe("loading");
      return p;
    });
    expect(panel.getAttribute("data-state")).toBe("unavailable");
    expect(panel).toHaveTextContent(/no watchlist peers share this sector/);
    expect(panel).toHaveTextContent(/Information Technology/);
  });

  it("renders one row per metric with rank + percentile", async () => {
    mockFetchOk(
      available({
        metrics: [
          metric({
            metric: "trust_score",
            label: "Trust score",
            value: 92,
            rank: 1,
            cohort_size: 4,
            percentile: 100,
          }),
          metric({
            metric: "overbought_score",
            label: "Overbought score",
            value: 70,
            rank: 3,
            cohort_size: 4,
            percentile: 33.3,
            direction: "lower_better",
          }),
        ],
      }),
    );
    render(<SectorRankPanel symbol="AAPL" />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("sector-rank-panel");
      // Wait past the loading state; render is async via the hook.
      expect(p.getAttribute("data-state")).not.toBe("loading");
      return p;
    });
    expect(panel.getAttribute("data-state")).toBe("ready");

    const trustRow = screen.getByTestId("sector-rank-row-trust_score");
    expect(trustRow).toHaveTextContent("Trust score");
    expect(trustRow).toHaveTextContent("#1 of 4");
    expect(trustRow).toHaveTextContent(/100th percentile/);

    const overboughtRow = screen.getByTestId("sector-rank-row-overbought_score");
    expect(overboughtRow).toHaveTextContent("Overbought score");
    expect(overboughtRow).toHaveTextContent("#3 of 4");
    expect(overboughtRow).toHaveTextContent("lower · better");
  });

  it("uses bull tone for top-percentile metrics", async () => {
    mockFetchOk(
      available({
        metrics: [
          metric({ percentile: 95 }),
        ],
      }),
    );
    render(<SectorRankPanel symbol="AAPL" />);
    await screen.findByTestId("sector-rank-panel");
    const bar = screen.getByTestId("position-bar");
    expect(bar.getAttribute("data-percentile")).toBe("95");
    // The bar inner div carries the bull tone class.
    const inner = bar.querySelector("div");
    expect(inner?.className ?? "").toContain("bg-bull");
  });

  it("uses warn tone for bottom-percentile metrics", async () => {
    mockFetchOk(
      available({
        metrics: [metric({ percentile: 20 })],
      }),
    );
    render(<SectorRankPanel symbol="AAPL" />);
    await screen.findByTestId("sector-rank-panel");
    const bar = screen.getByTestId("position-bar");
    const inner = bar.querySelector("div");
    expect(inner?.className ?? "").toContain("bg-warn");
  });

  it("renders empty bar when percentile is null (no data)", async () => {
    mockFetchOk(
      available({
        metrics: [
          metric({
            value: null,
            rank: null,
            percentile: null,
            cohort_size: 0,
          }),
        ],
      }),
    );
    render(<SectorRankPanel symbol="AAPL" />);
    await screen.findByTestId("sector-rank-panel");
    const bar = screen.getByTestId("position-bar");
    expect(bar.getAttribute("data-state")).toBe("empty");
  });

  it("surfaces cohort size in the header", async () => {
    mockFetchOk(available({ cohort_size: 12 }));
    render(<SectorRankPanel symbol="AAPL" />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("sector-rank-panel");
      // Wait past the loading state; render is async via the hook.
      expect(p.getAttribute("data-state")).not.toBe("loading");
      return p;
    });
    expect(panel).toHaveTextContent(/cohort · 12/);
  });

  it("shows '—' for missing rank values", async () => {
    mockFetchOk(
      available({
        metrics: [
          metric({ rank: null, cohort_size: 0, value: null, percentile: null }),
        ],
      }),
    );
    render(<SectorRankPanel symbol="AAPL" />);
    const row = await screen.findByTestId("sector-rank-row-trust_score");
    expect(row).toHaveTextContent("—");
  });
});
