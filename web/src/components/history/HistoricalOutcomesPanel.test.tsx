import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { HistoricalOutcomesPanel } from "./HistoricalOutcomesPanel";
import type { HistoricalOutcomes } from "@/lib/history";

function outcomes(over: Partial<HistoricalOutcomes> = {}): HistoricalOutcomes {
  return {
    symbol: "AAPL",
    horizon_days: 5,
    total_observations: 0,
    settled_observations: 0,
    first_observed_at: null,
    last_settled_at: null,
    buckets: [],
    ...over,
  };
}

function mockFetchOk(payload: HistoricalOutcomes): void {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload,
    statusText: "OK",
  }));
}

function mockFetchStatus(status: number): void {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: false,
    status,
    json: async () => ({ detail: "nope" }),
    statusText: "NOPE",
  }));
}

describe("HistoricalOutcomesPanel", () => {
  afterEach(() => {
    // Fake timers interact badly with the hook's async fetch — stick
    // to real time, just stub the fetch boundary.
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders nothing when the endpoint 404s (calibration disabled)", async () => {
    mockFetchStatus(404);
    const { container } = render(
      <HistoricalOutcomesPanel symbol="AAPL" />,
    );
    // After the disabled-store fetch resolves the panel removes itself.
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("renders the empty state when the symbol has no observations yet", async () => {
    mockFetchOk(outcomes());
    render(<HistoricalOutcomesPanel symbol="AAPL" />);
    const panel = await screen.findByTestId("history-panel");
    expect(panel.getAttribute("data-status")).toBe("empty");
    expect(panel).toHaveTextContent(/No observations recorded for AAPL yet/);
  });

  it("renders per-pairing buckets when observations exist", async () => {
    mockFetchOk(
      outcomes({
        total_observations: 12,
        settled_observations: 10,
        buckets: [
          {
            score_name: "pullback_risk",
            outcome_name: "return_negative",
            horizon_days: 5,
            bucket_lo: 70,
            bucket_hi: 80,
            n_observations: 10,
            n_hits: 7,
            hit_rate: 0.7,
            confidence_low: 0.4,
            confidence_high: 0.9,
            bucket_published: true,
          },
        ],
      }),
    );
    render(<HistoricalOutcomesPanel symbol="AAPL" />);
    const panel = await screen.findByTestId("history-panel");
    expect(panel.getAttribute("data-status")).toBe("ready");
    expect(panel).toHaveTextContent(/12 observations · 10 settled · 2 maturing/);
    const pairing = screen.getByTestId("history-pairing-pullback_risk");
    expect(pairing).toHaveTextContent(/Pullback risk → closed lower/);
    const bucket = screen.getByTestId("history-bucket-70");
    expect(bucket.getAttribute("data-status")).toBe("published");
    expect(bucket).toHaveTextContent(/\[70–80\) · 10 obs · 7\/10 \(70%\)/);
    expect(bucket).toHaveTextContent(/CI 40–90%/);
  });

  it("marks under-sampled buckets as insufficient sample", async () => {
    mockFetchOk(
      outcomes({
        total_observations: 3,
        settled_observations: 3,
        buckets: [
          {
            score_name: "pullback_risk",
            outcome_name: "return_negative",
            horizon_days: 5,
            bucket_lo: 70,
            bucket_hi: 80,
            n_observations: 3,
            n_hits: 2,
            hit_rate: 2 / 3,
            confidence_low: 0.2,
            confidence_high: 0.94,
            bucket_published: false,
          },
        ],
      }),
    );
    render(<HistoricalOutcomesPanel symbol="AAPL" />);
    const panel = await screen.findByTestId("history-panel");
    expect(panel.getAttribute("data-status")).toBe("ready");
    const bucket = screen.getByTestId("history-bucket-70");
    expect(bucket.getAttribute("data-status")).toBe("pending");
    expect(bucket).toHaveTextContent(/insufficient sample/);
    // Trust contract: no fabricated percentage in the headline.
    expect(bucket.textContent ?? "").not.toMatch(/\(\d+%\)/);
  });

  it("groups buckets within a pairing and sorts them by bucket_lo", async () => {
    mockFetchOk(
      outcomes({
        total_observations: 18,
        settled_observations: 18,
        buckets: [
          // Intentionally out of order — the component must sort.
          {
            score_name: "pullback_risk",
            outcome_name: "return_negative",
            horizon_days: 5,
            bucket_lo: 80,
            bucket_hi: 90,
            n_observations: 8,
            n_hits: 6,
            hit_rate: 0.75,
            confidence_low: 0.4,
            confidence_high: 0.94,
            bucket_published: true,
          },
          {
            score_name: "pullback_risk",
            outcome_name: "return_negative",
            horizon_days: 5,
            bucket_lo: 50,
            bucket_hi: 60,
            n_observations: 10,
            n_hits: 4,
            hit_rate: 0.4,
            confidence_low: 0.17,
            confidence_high: 0.69,
            bucket_published: true,
          },
        ],
      }),
    );
    render(<HistoricalOutcomesPanel symbol="AAPL" />);
    const pairing = await screen.findByTestId("history-pairing-pullback_risk");
    const buckets = pairing.querySelectorAll(
      "[data-testid^=history-bucket-]",
    );
    // First child must be the [50, 60) bucket (ascending order).
    expect(buckets[0].getAttribute("data-testid")).toBe("history-bucket-50");
    expect(buckets[1].getAttribute("data-testid")).toBe("history-bucket-80");
  });

  it("renders separate blocks for different (score, outcome) pairings", async () => {
    mockFetchOk(
      outcomes({
        total_observations: 20,
        settled_observations: 20,
        buckets: [
          {
            score_name: "pullback_risk",
            outcome_name: "return_negative",
            horizon_days: 5,
            bucket_lo: 70,
            bucket_hi: 80,
            n_observations: 10,
            n_hits: 7,
            hit_rate: 0.7,
            confidence_low: 0.4,
            confidence_high: 0.9,
            bucket_published: true,
          },
          {
            score_name: "rebound_potential",
            outcome_name: "return_positive",
            horizon_days: 5,
            bucket_lo: 10,
            bucket_hi: 20,
            n_observations: 10,
            n_hits: 5,
            hit_rate: 0.5,
            confidence_low: 0.24,
            confidence_high: 0.76,
            bucket_published: true,
          },
        ],
      }),
    );
    render(<HistoricalOutcomesPanel symbol="AAPL" />);
    await screen.findByTestId("history-panel");
    expect(
      screen.getByTestId("history-pairing-pullback_risk"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("history-pairing-rebound_potential"),
    ).toBeInTheDocument();
  });

  it("does not fetch when symbol is null", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    render(<HistoricalOutcomesPanel symbol={null} />);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
