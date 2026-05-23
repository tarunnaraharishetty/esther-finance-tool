import { describe, it, expect, vi, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { DriverSpotlight } from "./DriverSpotlight";
import type { SpotlightEntry, SpotlightResponse } from "@/lib/spotlight";
import type { Driver } from "@/lib/movementDrivers";

function driver(over: Partial<Driver> = {}): Driver {
  return {
    kind: "sector",
    label: "Sector outlier · Trust score",
    detail: "#1/24 in Information Technology on trust score.",
    tone: "bull",
    salience: 0.92,
    citations: ["sector"],
    ...over,
  };
}

function entry(over: Partial<SpotlightEntry> = {}): SpotlightEntry {
  return {
    rank: 1,
    symbol: "AAPL",
    driver: driver(),
    ...over,
  };
}

function response(over: Partial<SpotlightResponse> = {}): SpotlightResponse {
  return {
    entries: [entry()],
    considered_symbols: 2,
    surfaced_symbols: 1,
    watchlist_size: 2,
    assembled_symbols: 2,
    cache: "miss",
    ...over,
  };
}

function mockFetchOk(payload: SpotlightResponse): void {
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

describe("DriverSpotlight", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders an empty state when no entries surfaced", async () => {
    mockFetchOk(
      response({
        entries: [],
        surfaced_symbols: 0,
      }),
    );
    render(<DriverSpotlight onSelectSymbol={() => undefined} />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("driver-spotlight");
      expect(p.getAttribute("data-state")).not.toBe("loading");
      return p;
    });
    expect(panel.getAttribute("data-state")).toBe("empty");
    expect(panel).toHaveTextContent(/No notable signals across your watchlist/);
  });

  it("renders the backend reason in the empty state when one is provided", async () => {
    mockFetchOk(
      response({
        entries: [],
        watchlist_size: 0,
        reason: "watchlist is empty",
      }),
    );
    render(<DriverSpotlight onSelectSymbol={() => undefined} />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("driver-spotlight");
      expect(p.getAttribute("data-state")).toBe("empty");
      return p;
    });
    expect(panel).toHaveTextContent(/watchlist is empty/);
  });

  it("renders one card per entry in the ready state", async () => {
    mockFetchOk(
      response({
        entries: [
          entry({ rank: 1, symbol: "NVDA", driver: driver({ salience: 0.92 }) }),
          entry({
            rank: 2,
            symbol: "AAPL",
            driver: driver({
              kind: "sentiment",
              label: "Bullish news sentiment",
              tone: "bull",
              salience: 0.77,
            }),
          }),
          entry({
            rank: 3,
            symbol: "MSFT",
            driver: driver({
              kind: "trust",
              label: "Degraded report trust",
              tone: "warn",
              salience: 0.42,
            }),
          }),
        ],
        surfaced_symbols: 3,
      }),
    );
    render(<DriverSpotlight onSelectSymbol={() => undefined} />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("driver-spotlight");
      expect(p.getAttribute("data-state")).toBe("ready");
      return p;
    });
    expect(panel.getAttribute("data-entry-count")).toBe("3");
    expect(screen.getByTestId("spotlight-card-NVDA")).toBeInTheDocument();
    expect(screen.getByTestId("spotlight-card-AAPL")).toBeInTheDocument();
    expect(screen.getByTestId("spotlight-card-MSFT")).toBeInTheDocument();
  });

  it("applies tone-specific border classes per entry", async () => {
    mockFetchOk(
      response({
        entries: [
          entry({ rank: 1, symbol: "AAPL", driver: driver({ tone: "bull" }) }),
          entry({
            rank: 2,
            symbol: "MSFT",
            driver: driver({ tone: "warn" }),
          }),
        ],
        surfaced_symbols: 2,
      }),
    );
    render(<DriverSpotlight onSelectSymbol={() => undefined} />);
    await waitFor(() => {
      expect(
        screen.getByTestId("driver-spotlight").getAttribute("data-state"),
      ).toBe("ready");
    });
    expect(
      screen.getByTestId("spotlight-card-AAPL").className,
    ).toContain("border-bull");
    expect(
      screen.getByTestId("spotlight-card-MSFT").className,
    ).toContain("border-warn");
  });

  it("invokes onSelectSymbol when an entry is clicked", async () => {
    const onSelect = vi.fn();
    mockFetchOk(
      response({
        entries: [entry({ symbol: "NVDA" })],
        surfaced_symbols: 1,
      }),
    );
    render(<DriverSpotlight onSelectSymbol={onSelect} />);
    await waitFor(() => {
      expect(
        screen.getByTestId("driver-spotlight").getAttribute("data-state"),
      ).toBe("ready");
    });
    fireEvent.click(screen.getByTestId("spotlight-card-NVDA"));
    expect(onSelect).toHaveBeenCalledWith("NVDA");
  });

  it("renders rank + salience on each card", async () => {
    mockFetchOk(
      response({
        entries: [
          entry({ rank: 3, symbol: "AMD", driver: driver({ salience: 0.41 }) }),
        ],
      }),
    );
    render(<DriverSpotlight onSelectSymbol={() => undefined} />);
    await waitFor(() => {
      expect(
        screen.getByTestId("driver-spotlight").getAttribute("data-state"),
      ).toBe("ready");
    });
    const card = screen.getByTestId("spotlight-card-AMD");
    expect(card.getAttribute("data-rank")).toBe("3");
    expect(card).toHaveTextContent(/#3 · sal 41/);
  });

  it("surfaces the surfaced/total tally in the header", async () => {
    mockFetchOk(
      response({
        entries: [entry()],
        surfaced_symbols: 1,
        watchlist_size: 12,
      }),
    );
    render(<DriverSpotlight onSelectSymbol={() => undefined} />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("driver-spotlight");
      expect(p.getAttribute("data-state")).toBe("ready");
      return p;
    });
    expect(panel).toHaveTextContent(/1\/12 symbols surfaced/);
  });

  it("shows the error state with a retry button on fetch failure", async () => {
    mockFetchStatus(500);
    render(<DriverSpotlight onSelectSymbol={() => undefined} />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("driver-spotlight");
      expect(p.getAttribute("data-state")).toBe("error");
      return p;
    });
    expect(panel).toHaveTextContent(/Spotlight failed/);
    expect(
      screen.getByRole("button", { name: /Retry/i }),
    ).toBeInTheDocument();
  });
});
