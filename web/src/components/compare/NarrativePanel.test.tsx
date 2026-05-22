import { describe, it, expect, vi, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NarrativePanel } from "./NarrativePanel";
import type {
  ComparisonNarrative,
  NarrativeSection,
} from "@/lib/compareNarrative";
import type { TrustScore } from "@/lib/trust";

function section(over: Partial<NarrativeSection> = {}): NarrativeSection {
  return {
    title: "Stub",
    body: "Stub body sentence.",
    bullets: [],
    ...over,
  };
}

function trust(over: Partial<TrustScore> = {}): TrustScore {
  return {
    score: 95,
    grade: "A+",
    components: [],
    ...over,
  };
}

function narrative(over: Partial<ComparisonNarrative> = {}): ComparisonNarrative {
  return {
    left_symbol: "AAA",
    right_symbol: "BBB",
    tagline: "AAA edges BBB on trust composite.",
    headline: section({ title: "Headline", body: "AAA leads with 92 vs 78." }),
    momentum: section({
      title: "Momentum",
      body: "AAA shows oversold 60 vs 25.",
      bullets: ["Overbought: AAA 40 vs BBB 70 — AAA leads."],
    }),
    valuation: section({ title: "Valuation", body: "AAA confidence 80 vs 60." }),
    risk: section({ title: "Risk", body: "AAA pullback 30 vs 65." }),
    quality: section({ title: "Data Quality", body: "Trust AAA 92 vs BBB 78." }),
    bottom_line: section({
      title: "Bottom Line",
      body: "The split favors AAA broadly.",
    }),
    model: "template@compare-narrative-v1",
    generated_at: "2026-05-22T00:00:00+00:00",
    warnings: [],
    validation: { drop_count: 0, dropped_claims: [] },
    trust_score: trust(),
    mode: "template",
    cache: "miss",
    assembly_warnings: [],
    ...over,
  };
}

function mockFetchOk(payload: ComparisonNarrative): void {
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

function mockFetchStatus(status: number, detail = "nope"): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      status,
      json: async () => ({ detail }),
      statusText: detail,
    }),
  );
}

describe("NarrativePanel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders the empty CTA when no narrative has been requested", () => {
    render(<NarrativePanel leftSymbol="AAA" rightSymbol="BBB" />);
    const panel = screen.getByTestId("narrative-panel");
    expect(panel.getAttribute("data-state")).toBe("empty");
    expect(panel).toHaveTextContent(/Generate a grounded prose comparison/);
    expect(screen.getByRole("button", { name: /Generate/i })).toBeInTheDocument();
  });

  it("returns null when either symbol is missing", () => {
    const { container } = render(
      <NarrativePanel leftSymbol={null} rightSymbol="BBB" />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("returns null when both symbols match (avoid the always-degraded call)", () => {
    const { container } = render(
      <NarrativePanel leftSymbol="AAA" rightSymbol="AAA" />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("fetches and renders the narrative after Generate is clicked", async () => {
    mockFetchOk(narrative());
    render(<NarrativePanel leftSymbol="AAA" rightSymbol="BBB" />);
    fireEvent.click(screen.getByRole("button", { name: /Generate/i }));
    const panel = await waitFor(() => {
      const p = screen.getByTestId("narrative-panel");
      expect(p.getAttribute("data-state")).toBe("ready");
      return p;
    });
    expect(panel).toHaveTextContent(/AAA edges BBB on trust/);
    expect(panel).toHaveTextContent(/AAA leads with 92 vs 78/);
  });

  it("renders the validation chip with the drop count", async () => {
    mockFetchOk(
      narrative({
        validation: {
          drop_count: 2,
          dropped_claims: [
            {
              section: "momentum.body",
              sentence: "AAA reported a 99.9% surge.",
              unsupported_tokens: ["99.9%"],
            },
            {
              section: "valuation.bullets[0]",
              sentence: "Made-up margin: 42.7%",
              unsupported_tokens: ["42.7%"],
            },
          ],
        },
      }),
    );
    render(<NarrativePanel leftSymbol="AAA" rightSymbol="BBB" />);
    fireEvent.click(screen.getByRole("button", { name: /Generate/i }));
    const chip = await screen.findByTestId("narrative-validation-chip");
    expect(chip.getAttribute("data-drop-count")).toBe("2");
    expect(chip).toHaveTextContent(/2 claims dropped/);
    expect(chip.className).toContain("text-warn");
  });

  it("renders one section block per narrative section", async () => {
    mockFetchOk(narrative());
    render(<NarrativePanel leftSymbol="AAA" rightSymbol="BBB" />);
    fireEvent.click(screen.getByRole("button", { name: /Generate/i }));
    await screen.findByTestId("narrative-section-headline");
    expect(screen.getByTestId("narrative-section-momentum")).toBeInTheDocument();
    expect(screen.getByTestId("narrative-section-valuation")).toBeInTheDocument();
    expect(screen.getByTestId("narrative-section-risk")).toBeInTheDocument();
    expect(screen.getByTestId("narrative-section-quality")).toBeInTheDocument();
    expect(
      screen.getByTestId("narrative-section-bottom_line"),
    ).toBeInTheDocument();
  });

  it("shows a 'every claim dropped' note when a section scrubs to empty", async () => {
    mockFetchOk(
      narrative({
        momentum: section({
          title: "Momentum",
          body: "",
          bullets: [],
        }),
      }),
    );
    render(<NarrativePanel leftSymbol="AAA" rightSymbol="BBB" />);
    fireEvent.click(screen.getByRole("button", { name: /Generate/i }));
    const momentum = await screen.findByTestId("narrative-section-momentum");
    expect(momentum).toHaveTextContent(/Every claim in this section dropped/);
  });

  it("surfaces a warning banner when the response carries one", async () => {
    mockFetchOk(
      narrative({
        warning:
          "AI narrative failed (AuthenticationError); rendering deterministic template instead.",
      }),
    );
    render(<NarrativePanel leftSymbol="AAA" rightSymbol="BBB" />);
    fireEvent.click(screen.getByRole("button", { name: /Generate/i }));
    // Wait for the ready state specifically — by default waitFor would
    // resolve with the loading-state panel since both states share the
    // same testid.
    const panel = await waitFor(() => {
      const p = screen.getByTestId("narrative-panel");
      expect(p.getAttribute("data-state")).toBe("ready");
      return p;
    });
    expect(panel).toHaveTextContent(/AI narrative failed/);
  });

  it("renders the error state with a retry button when the fetch fails", async () => {
    mockFetchStatus(503, "ANTHROPIC_API_KEY is not set");
    render(<NarrativePanel leftSymbol="AAA" rightSymbol="BBB" />);
    fireEvent.click(screen.getByRole("button", { name: /Generate/i }));
    const panel = await waitFor(() => {
      const p = screen.getByTestId("narrative-panel");
      expect(p.getAttribute("data-state")).toBe("error");
      return p;
    });
    expect(panel).toHaveTextContent(/ANTHROPIC_API_KEY/);
    expect(screen.getByRole("button", { name: /Retry/i })).toBeInTheDocument();
  });
});
