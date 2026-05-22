import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScenarioRange } from "./ScenarioRange";
import type { ScenarioModel } from "@/lib/analyzer";

function scenarios(over: Partial<ScenarioModel> = {}): ScenarioModel {
  return {
    horizon_days: 30,
    current_price: 100.0,
    annualized_vol: 0.25,
    vol_lookback_days: 30,
    vol_method: "realized_log",
    prob_below_bear: 0.18,
    prob_above_bull: 0.07,
    quantile_20: 94.0,
    quantile_50: 100.0,
    quantile_80: 107.0,
    notes: [
      "Probabilities assume zero drift — they describe today's dispersion, not a forecast.",
      "Lognormal model under-estimates the likelihood of large moves.",
    ],
    ...over,
  };
}

describe("ScenarioRange", () => {
  it("renders the empty state when scenarios is null", () => {
    render(<ScenarioRange scenarios={null} lastPrice={100} />);
    expect(screen.getByTestId("scenario-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("scenario-range")).not.toBeInTheDocument();
  });

  it("renders quantiles + horizon + annualized vol", () => {
    render(<ScenarioRange scenarios={scenarios()} lastPrice={100} />);
    expect(screen.getByTestId("scenario-range")).toBeInTheDocument();
    expect(screen.getByText(/30d horizon/i)).toBeInTheDocument();
    expect(screen.getByText(/25\.0% ann/i)).toBeInTheDocument();
    expect(screen.getByText("$94.00")).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("$107.00")).toBeInTheDocument();
  });

  it("renders both tail probabilities when bear and bull cases exist", () => {
    render(<ScenarioRange scenarios={scenarios()} lastPrice={100} />);
    const tails = screen.getByTestId("scenario-tails");
    expect(tails).toBeInTheDocument();
    expect(screen.getByTestId("scenario-tail-below")).toHaveTextContent("18%");
    expect(screen.getByTestId("scenario-tail-above")).toHaveTextContent("7%");
  });

  it("hides the tail block when both probs are null", () => {
    render(
      <ScenarioRange
        scenarios={scenarios({ prob_below_bear: null, prob_above_bull: null })}
        lastPrice={100}
      />,
    );
    expect(screen.queryByTestId("scenario-tails")).not.toBeInTheDocument();
    // Quantiles must still render — absence of valuation cases doesn't
    // hide the dispersion read.
    expect(screen.getByText("$94.00")).toBeInTheDocument();
  });

  it("renders only the upside tail when bear case is absent", () => {
    render(
      <ScenarioRange
        scenarios={scenarios({ prob_below_bear: null, prob_above_bull: 0.07 })}
        lastPrice={100}
      />,
    );
    expect(screen.getByTestId("scenario-tails")).toBeInTheDocument();
    expect(screen.queryByTestId("scenario-tail-below")).not.toBeInTheDocument();
    expect(screen.getByTestId("scenario-tail-above")).toBeInTheDocument();
  });

  it("renders the model-card notes verbatim", () => {
    const notes = scenarios();
    render(<ScenarioRange scenarios={notes} lastPrice={100} />);
    const block = screen.getByTestId("scenario-notes");
    expect(block).toBeInTheDocument();
    // Both notes from the fixture must appear unmodified.
    for (const n of notes.notes) {
      expect(block).toHaveTextContent(n);
    }
  });

  it("does not render the last-price marker when lastPrice is null", () => {
    render(<ScenarioRange scenarios={scenarios()} lastPrice={null} />);
    const bar = screen.getByTestId("scenario-bar");
    expect(bar.querySelector("[aria-label]")).toBeNull();
  });
});
