import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricRow } from "./MetricRow";
import type { MetricComparison } from "@/lib/compare";

function metric(over: Partial<MetricComparison> = {}): MetricComparison {
  return {
    metric: "trust_score",
    label: "Trust score",
    left_value: 92,
    right_value: 78,
    winner: "left",
    direction: "higher_better",
    detail: "Trust composite comparison.",
    percent: false,
    ...over,
  };
}

describe("MetricRow", () => {
  it("renders both values and the metric label", () => {
    render(<MetricRow metric={metric()} />);
    const row = screen.getByTestId("metric-row-trust_score");
    expect(row).toHaveTextContent(/92/);
    expect(row).toHaveTextContent(/78/);
    expect(row).toHaveTextContent(/Trust score/);
  });

  it("data-winner attribute reflects the verdict for downstream styling", () => {
    render(<MetricRow metric={metric({ winner: "right" })} />);
    expect(
      screen.getByTestId("metric-row-trust_score").getAttribute("data-winner"),
    ).toBe("right");
  });

  it("renders a tie chip when the verdict is tie", () => {
    render(<MetricRow metric={metric({ winner: "tie" })} />);
    const chip = screen.getByTestId("winner-chip");
    expect(chip.getAttribute("data-state")).toBe("tie");
    expect(chip).toHaveTextContent(/tie/i);
  });

  it("renders an n/a chip when both sides lacked data", () => {
    render(
      <MetricRow
        metric={metric({
          winner: "n/a",
          left_value: null,
          right_value: null,
        })}
      />,
    );
    const chip = screen.getByTestId("winner-chip");
    expect(chip.getAttribute("data-state")).toBe("na");
    // Missing values render as em-dash.
    const row = screen.getByTestId("metric-row-trust_score");
    expect(row.textContent ?? "").toMatch(/—/);
  });

  it("displays the winning side in the bull tone", () => {
    render(<MetricRow metric={metric({ winner: "left" })} />);
    // Pick out the left value cell — it's the first font-mono cell
    // in the row. We assert via class presence since color tones are
    // the load-bearing UI signal here.
    const row = screen.getByTestId("metric-row-trust_score");
    const valueCells = row.querySelectorAll("div.font-mono.text-base");
    expect(valueCells.length).toBeGreaterThanOrEqual(2);
    expect(valueCells[0].className).toContain("text-bull");
  });

  it("formats percent metrics with a sign + percent suffix", () => {
    render(
      <MetricRow
        metric={metric({
          metric: "upside_to_base_case",
          label: "Upside to base case",
          left_value: 30.0,
          right_value: -5.5,
          percent: true,
        })}
      />,
    );
    const row = screen.getByTestId("metric-row-upside_to_base_case");
    expect(row).toHaveTextContent(/\+30\.0%/);
    expect(row).toHaveTextContent(/-5\.5%/);
  });

  it("renders the direction badge as 'lower · better' for inverted metrics", () => {
    render(
      <MetricRow
        metric={metric({
          metric: "overbought_score",
          label: "Overbought",
          direction: "lower_better",
          left_value: 30,
          right_value: 60,
          winner: "left",  // left lower → wins
        })}
      />,
    );
    expect(screen.getByTestId("direction-badge")).toHaveTextContent(
      /lower · better/,
    );
  });

  it("renders the direction badge as 'higher · better' for normal metrics", () => {
    render(<MetricRow metric={metric()} />);
    expect(screen.getByTestId("direction-badge")).toHaveTextContent(
      /higher · better/,
    );
  });
});
