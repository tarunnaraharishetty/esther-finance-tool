import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ComparisonSectionCard } from "./ComparisonSectionCard";
import type { ComparisonSection, MetricComparison } from "@/lib/compare";

function metric(over: Partial<MetricComparison> = {}): MetricComparison {
  return {
    metric: "trust_score",
    label: "Trust score",
    left_value: 92,
    right_value: 78,
    winner: "left",
    direction: "higher_better",
    detail: "x",
    percent: false,
    ...over,
  };
}

function section(over: Partial<ComparisonSection> = {}): ComparisonSection {
  return {
    title: "Trust",
    metrics: [],
    ...over,
  };
}

describe("ComparisonSectionCard", () => {
  it("renders the section title and one row per metric", () => {
    render(
      <ComparisonSectionCard
        section={section({
          title: "Trust",
          metrics: [
            metric({ metric: "trust_score" }),
            metric({ metric: "provider_confidence", label: "Provider" }),
          ],
        })}
      />,
    );
    const card = screen.getByTestId("compare-section-trust");
    expect(card).toHaveTextContent("Trust");
    expect(screen.getByTestId("metric-row-trust_score")).toBeInTheDocument();
    expect(
      screen.getByTestId("metric-row-provider_confidence"),
    ).toBeInTheDocument();
  });

  it("renders the per-side tally in the header (3-1)", () => {
    render(
      <ComparisonSectionCard
        section={section({
          title: "Risk",
          metrics: [
            metric({ winner: "left" }),
            metric({ winner: "left" }),
            metric({ winner: "left" }),
            metric({ winner: "right" }),
          ],
        })}
      />,
    );
    const card = screen.getByTestId("compare-section-risk");
    expect(card).toHaveTextContent("3–1");
  });

  it("includes the tie count in the tally when ties occurred", () => {
    render(
      <ComparisonSectionCard
        section={section({
          title: "Valuation",
          metrics: [
            metric({ winner: "left" }),
            metric({ winner: "tie" }),
          ],
        })}
      />,
    );
    expect(screen.getByTestId("compare-section-valuation")).toHaveTextContent(
      /1 tie/,
    );
  });

  it("renders the metric label per row inside the card", () => {
    render(
      <ComparisonSectionCard
        section={section({
          title: "Technical",
          metrics: [
            metric({
              metric: "overbought_score",
              label: "Overbought",
              winner: "right",
              direction: "lower_better",
            }),
          ],
        })}
      />,
    );
    expect(screen.getByTestId("compare-section-technical")).toHaveTextContent(
      "Overbought",
    );
  });
});
