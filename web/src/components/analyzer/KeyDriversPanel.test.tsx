import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { KeyDriversPanel } from "./KeyDriversPanel";
import type {
  Driver,
  KeyDriversResponse,
} from "@/lib/movementDrivers";

function driver(over: Partial<Driver> = {}): Driver {
  return {
    kind: "technical",
    label: "Overbought technicals",
    detail: "overbought composite reads 82/100.",
    tone: "warn",
    salience: 0.64,
    citations: ["technicals"],
    ...over,
  };
}

function response(over: Partial<KeyDriversResponse> = {}): KeyDriversResponse {
  return {
    symbol: "AAPL",
    drivers: [driver()],
    considered: 1,
    suppressed: 0,
    sources: { analyzer: true, sector: true, snapshot_row: true },
    cache: "miss",
    ...over,
  };
}

function mockFetchOk(payload: KeyDriversResponse): void {
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

describe("KeyDriversPanel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns null when no symbol is selected", () => {
    const { container } = render(<KeyDriversPanel symbol={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing on error (analyzer surface owns errors)", async () => {
    mockFetchStatus(500);
    const { container } = render(<KeyDriversPanel symbol="AAPL" />);
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("renders the empty state when no drivers cleared the threshold", async () => {
    mockFetchOk(response({ drivers: [], suppressed: 2 }));
    render(<KeyDriversPanel symbol="AAPL" />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("key-drivers-panel");
      expect(p.getAttribute("data-state")).not.toBe("loading");
      return p;
    });
    expect(panel.getAttribute("data-state")).toBe("empty");
    expect(panel).toHaveTextContent(/No notable signals stand out for AAPL/);
  });

  it("renders one card per driver in salience order", async () => {
    mockFetchOk(
      response({
        drivers: [
          driver({
            kind: "sector",
            label: "Sector outlier · Trust score",
            salience: 0.92,
            tone: "bull",
          }),
          driver({
            kind: "sentiment",
            label: "Bullish news sentiment",
            salience: 0.77,
            tone: "bull",
          }),
          driver({
            kind: "trust",
            label: "Degraded report trust",
            salience: 0.42,
            tone: "warn",
          }),
        ],
        considered: 3,
        suppressed: 0,
      }),
    );
    render(<KeyDriversPanel symbol="AAPL" />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("key-drivers-panel");
      expect(p.getAttribute("data-state")).toBe("ready");
      return p;
    });
    expect(panel.getAttribute("data-driver-count")).toBe("3");
    expect(screen.getByTestId("driver-card-sector")).toBeInTheDocument();
    expect(screen.getByTestId("driver-card-sentiment")).toBeInTheDocument();
    expect(screen.getByTestId("driver-card-trust")).toBeInTheDocument();
  });

  it("applies the correct tone classes per driver", async () => {
    mockFetchOk(
      response({
        drivers: [
          driver({ kind: "sector", tone: "bull" }),
          driver({ kind: "trust", tone: "warn" }),
        ],
      }),
    );
    render(<KeyDriversPanel symbol="AAPL" />);
    await waitFor(() => {
      expect(
        screen.getByTestId("key-drivers-panel").getAttribute("data-state"),
      ).toBe("ready");
    });
    const sector = screen.getByTestId("driver-card-sector");
    const trust = screen.getByTestId("driver-card-trust");
    expect(sector.getAttribute("data-tone")).toBe("bull");
    expect(sector.className).toContain("border-bull");
    expect(trust.getAttribute("data-tone")).toBe("warn");
    expect(trust.className).toContain("border-warn");
  });

  it("renders citation chips for each driver", async () => {
    mockFetchOk(
      response({
        drivers: [
          driver({
            kind: "sentiment",
            label: "Bullish news sentiment",
            citations: ["sentiment", "news"],
          }),
        ],
      }),
    );
    render(<KeyDriversPanel symbol="AAPL" />);
    await waitFor(() => {
      expect(
        screen.getByTestId("key-drivers-panel").getAttribute("data-state"),
      ).toBe("ready");
    });
    const card = screen.getByTestId("driver-card-sentiment");
    expect(card).toHaveTextContent(/sentiment/);
    expect(card).toHaveTextContent(/news/);
  });

  it("renders salience as both a bar and a number", async () => {
    mockFetchOk(
      response({
        drivers: [driver({ salience: 0.73 })],
      }),
    );
    render(<KeyDriversPanel symbol="AAPL" />);
    await waitFor(() => {
      expect(
        screen.getByTestId("key-drivers-panel").getAttribute("data-state"),
      ).toBe("ready");
    });
    const card = screen.getByTestId("driver-card-technical");
    expect(card.getAttribute("data-salience")).toBe("73");
    expect(card).toHaveTextContent(/salience 73/);
  });

  it("surfaces suppressed count in the header when non-zero", async () => {
    mockFetchOk(
      response({
        drivers: [driver()],
        considered: 1,
        suppressed: 3,
      }),
    );
    render(<KeyDriversPanel symbol="AAPL" />);
    await waitFor(() => {
      expect(
        screen.getByTestId("key-drivers-panel").getAttribute("data-state"),
      ).toBe("ready");
    });
    expect(
      screen.getByTestId("key-drivers-panel"),
    ).toHaveTextContent(/3 low-signal drivers suppressed/);
  });

  it("uses singular wording when one driver is suppressed", async () => {
    mockFetchOk(
      response({
        drivers: [driver()],
        considered: 1,
        suppressed: 1,
      }),
    );
    render(<KeyDriversPanel symbol="AAPL" />);
    const panel = await waitFor(() => {
      const p = screen.getByTestId("key-drivers-panel");
      expect(p.getAttribute("data-state")).toBe("ready");
      return p;
    });
    expect(panel).toHaveTextContent(/1 low-signal driver suppressed/);
    expect(panel.textContent ?? "").not.toMatch(/drivers suppressed/);
  });
});
