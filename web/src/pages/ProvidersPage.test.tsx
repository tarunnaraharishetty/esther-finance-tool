import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ProvidersPage } from "./ProvidersPage";
import type {
  ProviderFieldAccuracyBlock,
  ProviderRow,
  ProvidersResponse,
  ProviderTrustBreakdown,
} from "@/lib/providers";

function trust(
  over: Partial<ProviderTrustBreakdown> = {},
): ProviderTrustBreakdown {
  return {
    provider: "fmp",
    weight: 1.0,
    accuracy: 0.92,
    uptime: 0.98,
    latency_p95_ms: 142,
    latency_score: 0.95,
    events: 240,
    health_calls: 412,
    cold_start: false,
    ...over,
  };
}

function fieldAccuracy(
  over: Partial<ProviderFieldAccuracyBlock> = {},
): ProviderFieldAccuracyBlock {
  return {
    provider: "fmp",
    window_seconds: 3600,
    total_events: 4,
    total_agreed: 3,
    overall_accuracy: 0.75,
    by_field: [
      {
        provider: "fmp",
        field: "revenue",
        total: 2,
        agreed: 2,
        accuracy: 1.0,
      },
      {
        provider: "fmp",
        field: "net_income",
        total: 2,
        agreed: 1,
        accuracy: 0.5,
      },
    ],
    ...over,
  };
}

function provider(over: Partial<ProviderRow> = {}): ProviderRow {
  return {
    provider: "fmp",
    window_seconds: 3600,
    total: 100,
    ok: 98,
    rate_limited: 1,
    unavailable: 0,
    transient: 1,
    empty: 0,
    skipped: 0,
    success_rate: 0.98,
    p50_latency_ms: 80,
    p95_latency_ms: 142,
    last_seen_at: "2026-05-23T13:00:00+00:00",
    last_error_at: null,
    breakdown: { ok: 98, rate_limited: 1, transient: 1 },
    trust_breakdown: trust(),
    field_accuracy: fieldAccuracy(),
    ...over,
  };
}

function response(over: Partial<ProvidersResponse> = {}): ProvidersResponse {
  return {
    as_of: "2026-05-23T13:00:00+00:00",
    window_seconds: 3600,
    providers: [provider()],
    recent_failures: [],
    ...over,
  };
}

function mockFetchOk(payload: ProvidersResponse): void {
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

describe("ProvidersPage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders the hero + one card per provider in the ready state", async () => {
    mockFetchOk(
      response({
        providers: [
          provider({ provider: "fmp" }),
          provider({
            provider: "finnhub",
            trust_breakdown: trust({ provider: "finnhub", weight: 0.78 }),
            field_accuracy: fieldAccuracy({ provider: "finnhub" }),
          }),
        ],
      }),
    );
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    expect(screen.getByTestId("provider-card-fmp")).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-finnhub")).toBeInTheDocument();
  });

  it("colors the card tone based on trust weight", async () => {
    mockFetchOk(
      response({
        providers: [
          provider({
            provider: "good",
            trust_breakdown: trust({ provider: "good", weight: 1.15 }),
          }),
          provider({
            provider: "bad",
            trust_breakdown: trust({ provider: "bad", weight: 0.55 }),
          }),
          provider({
            provider: "warn",
            trust_breakdown: trust({ provider: "warn", weight: 0.75 }),
          }),
        ],
      }),
    );
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    expect(
      screen.getByTestId("provider-card-good").getAttribute("data-tone"),
    ).toBe("bull");
    expect(
      screen.getByTestId("provider-card-bad").getAttribute("data-tone"),
    ).toBe("bear");
    expect(
      screen.getByTestId("provider-card-warn").getAttribute("data-tone"),
    ).toBe("warn");
  });

  it("renders the weight gauge with formatted value", async () => {
    mockFetchOk(
      response({
        providers: [
          provider({
            trust_breakdown: trust({ weight: 0.873 }),
          }),
        ],
      }),
    );
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    const gauge = screen.getByTestId("weight-gauge");
    expect(gauge.getAttribute("data-weight")).toBe("0.87");
    expect(gauge).toHaveTextContent("0.87");
  });

  it("renders cold-start badge when trust breakdown flag set", async () => {
    mockFetchOk(
      response({
        providers: [
          provider({
            trust_breakdown: trust({
              cold_start: true,
              accuracy: null,
              events: 3,
            }),
          }),
        ],
      }),
    );
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    expect(screen.getByTestId("cold-start-badge")).toBeInTheDocument();
  });

  it("renders per-field accuracy rows when present", async () => {
    mockFetchOk(response());
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    const table = screen.getByTestId("field-accuracy-table");
    expect(table).toBeInTheDocument();
    expect(screen.getByTestId("field-row-revenue")).toHaveTextContent(
      /revenue.*2\/2.*100%/,
    );
    expect(screen.getByTestId("field-row-net_income")).toHaveTextContent(
      /net_income.*1\/2.*50%/,
    );
  });

  it("renders the empty field-accuracy hint when no events landed", async () => {
    mockFetchOk(
      response({
        providers: [
          provider({
            field_accuracy: fieldAccuracy({
              total_events: 0,
              total_agreed: 0,
              by_field: [],
            }),
          }),
        ],
      }),
    );
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    expect(screen.getByTestId("field-accuracy-empty")).toHaveTextContent(
      /No reconciliation events for fmp/,
    );
  });

  it("renders component bars for accuracy, uptime, and latency", async () => {
    mockFetchOk(response());
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    expect(screen.getByTestId("component-accuracy")).toHaveTextContent("92%");
    expect(screen.getByTestId("component-uptime")).toHaveTextContent("98%");
    expect(screen.getByTestId("component-latency")).toHaveTextContent("95%");
    expect(screen.getByTestId("component-latency")).toHaveTextContent("142ms");
  });

  it("renders an em-dash for null component values", async () => {
    mockFetchOk(
      response({
        providers: [
          provider({
            trust_breakdown: trust({ accuracy: null, latency_score: null }),
          }),
        ],
      }),
    );
    render(<ProvidersPage />);
    await screen.findByTestId("providers-hero");
    expect(screen.getByTestId("component-accuracy")).toHaveTextContent("—");
    expect(screen.getByTestId("component-latency")).toHaveTextContent("—");
  });

  it("renders the empty state when no providers returned", async () => {
    mockFetchOk(response({ providers: [] }));
    render(<ProvidersPage />);
    const hero = await screen.findByTestId("providers-hero");
    // Hero still renders even when providers is empty.
    expect(hero).toBeInTheDocument();
    // Empty-state copy surfaces below.
    expect(
      screen.queryByTestId("provider-card-fmp"),
    ).not.toBeInTheDocument();
  });

  it("renders an error state with retry on fetch failure", async () => {
    mockFetchStatus(500);
    render(<ProvidersPage />);
    const retryButton = await waitFor(() =>
      screen.getByRole("button", { name: /Retry/i }),
    );
    expect(retryButton).toBeInTheDocument();
  });
});
