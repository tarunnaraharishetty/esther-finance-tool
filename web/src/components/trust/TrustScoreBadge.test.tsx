import { describe, it, expect } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { TrustScoreBadge } from "./TrustScoreBadge";
import type { TrustComponent, TrustScore } from "@/lib/trust";

function component(over: Partial<TrustComponent> = {}): TrustComponent {
  return {
    name: "freshness",
    weight: 0.25,
    value: 100,
    contribution: 26.3,
    status: "ok",
    detail: "Fundamentals data tier: fresh.",
    ...over,
  };
}

function trust(over: Partial<TrustScore> = {}): TrustScore {
  return {
    score: 95,
    grade: "A+",
    components: [
      component(),
      component({
        name: "provider_confidence",
        detail: "Provider chain confidence after reconciliation: 1.00.",
      }),
      component({
        name: "analyzer_confidence",
        weight: 0.2,
        contribution: 21.1,
        detail: "Technical sub-score coverage × agreement: 0.92.",
      }),
      component({
        name: "calibration_coverage",
        weight: 0.15,
        contribution: 15.8,
        detail: "75% of analyzer scores have a published bucket.",
      }),
      component({
        name: "scenario_availability",
        weight: 0.1,
        contribution: 10.5,
        detail: "Probabilistic scenario layer attached.",
      }),
      component({
        name: "validation_cleanliness",
        weight: 0.05,
        value: null,
        contribution: null,
        status: "n/a",
        detail: "Analyzer reports are deterministic; no LLM validation step.",
      }),
    ],
    ...over,
  };
}

describe("TrustScoreBadge", () => {
  it("renders the grade and score on the collapsed chip", () => {
    render(<TrustScoreBadge trust={trust()} />);
    const badge = screen.getByTestId("trust-score-badge");
    expect(badge.getAttribute("data-grade")).toBe("A+");
    expect(badge.getAttribute("data-score")).toBe("95");
    expect(badge).toHaveTextContent(/A\+/);
    expect(badge).toHaveTextContent(/trust 95\/100/);
  });

  it("uses the bull palette for A/A+ grades", () => {
    render(<TrustScoreBadge trust={trust({ grade: "A" })} />);
    const badge = screen.getByTestId("trust-score-badge");
    expect(badge.className).toContain("text-bull");
  });

  it("uses the warn palette for B/C grades", () => {
    render(<TrustScoreBadge trust={trust({ grade: "C", score: 72 })} />);
    const badge = screen.getByTestId("trust-score-badge");
    expect(badge.className).toContain("text-warn");
  });

  it("uses the bear palette for D/F grades", () => {
    render(<TrustScoreBadge trust={trust({ grade: "F", score: 42 })} />);
    const badge = screen.getByTestId("trust-score-badge");
    expect(badge.className).toContain("text-bear");
  });

  it("hides the breakdown panel until the user clicks the chip", () => {
    render(<TrustScoreBadge trust={trust()} />);
    expect(screen.queryByTestId("trust-score-breakdown")).toBeNull();

    fireEvent.click(screen.getByTestId("trust-score-badge"));
    expect(screen.getByTestId("trust-score-breakdown")).toBeInTheDocument();
  });

  it("renders one row per component, labeled in human terms", () => {
    render(<TrustScoreBadge trust={trust()} />);
    fireEvent.click(screen.getByTestId("trust-score-badge"));

    expect(screen.getByTestId("trust-row-freshness")).toBeInTheDocument();
    expect(screen.getByTestId("trust-row-provider_confidence")).toBeInTheDocument();
    expect(screen.getByTestId("trust-row-analyzer_confidence")).toBeInTheDocument();
    expect(screen.getByTestId("trust-row-calibration_coverage")).toBeInTheDocument();
    expect(screen.getByTestId("trust-row-scenario_availability")).toBeInTheDocument();
    expect(screen.getByTestId("trust-row-validation_cleanliness")).toBeInTheDocument();

    const breakdown = screen.getByTestId("trust-score-breakdown");
    // Human labels, not raw snake_case keys.
    expect(breakdown).toHaveTextContent("Data freshness");
    expect(breakdown).toHaveTextContent("Provider confidence");
    expect(breakdown).toHaveTextContent("Calibration coverage");
  });

  it("shows '—' for components with no contribution (n/a or missing)", () => {
    render(<TrustScoreBadge trust={trust()} />);
    fireEvent.click(screen.getByTestId("trust-score-badge"));
    const row = screen.getByTestId("trust-row-validation_cleanliness");
    expect(row.getAttribute("data-status")).toBe("n/a");
    expect(row).toHaveTextContent("—");
  });

  it("renders the detail string for each component for tooltipless auditing", () => {
    render(<TrustScoreBadge trust={trust()} />);
    fireEvent.click(screen.getByTestId("trust-score-badge"));
    const row = screen.getByTestId("trust-row-freshness");
    expect(row).toHaveTextContent("Fundamentals data tier: fresh.");
  });

  it("toggles open and closed on repeated clicks", () => {
    render(<TrustScoreBadge trust={trust()} />);
    const badge = screen.getByTestId("trust-score-badge");
    fireEvent.click(badge);
    expect(screen.getByTestId("trust-score-breakdown")).toBeInTheDocument();
    fireEvent.click(badge);
    expect(screen.queryByTestId("trust-score-breakdown")).toBeNull();
  });

  it("marks missing components with their status in the row", () => {
    const t = trust({
      grade: "C",
      score: 70,
      components: [
        component({
          name: "freshness",
          value: null,
          contribution: null,
          status: "missing",
          detail: "No fundamentals freshness envelope available.",
        }),
      ],
    });
    render(<TrustScoreBadge trust={t} />);
    fireEvent.click(screen.getByTestId("trust-score-badge"));
    const row = screen.getByTestId("trust-row-freshness");
    expect(row.getAttribute("data-status")).toBe("missing");
    expect(row).toHaveTextContent(/missing/);
  });
});
