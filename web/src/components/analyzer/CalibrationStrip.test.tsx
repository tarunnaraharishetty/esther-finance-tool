import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CalibrationStrip } from "./CalibrationStrip";
import type { CalibrationReading } from "@/lib/analyzer";

function reading(over: Partial<CalibrationReading>): CalibrationReading {
  return {
    score_name: "pullback_risk",
    score_value: 73,
    outcome_name: "return_negative",
    horizon_days: 5,
    bucket_published: false,
    bucket_lo: null,
    bucket_hi: null,
    n_observations: null,
    n_hits: null,
    hit_rate: null,
    confidence_low: null,
    confidence_high: null,
    last_updated: null,
    ...over,
  };
}

describe("CalibrationStrip", () => {
  it("renders nothing when calibrations array is empty", () => {
    const { container } = render(<CalibrationStrip calibrations={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the panel header with the horizon", () => {
    render(<CalibrationStrip calibrations={[reading({})]} />);
    expect(screen.getByTestId("calibration-strip")).toBeInTheDocument();
    expect(screen.getByText(/5d horizon/i)).toBeInTheDocument();
  });

  it("shows 'calibration pending' for an under-sampled reading", () => {
    render(<CalibrationStrip calibrations={[reading({})]} />);
    const row = screen.getByTestId("calibration-row-pullback_risk");
    expect(row.getAttribute("data-status")).toBe("pending");
    expect(row).toHaveTextContent(/calibration pending/i);
    // Trust invariant: never publish a percentage on an under-sampled
    // reading. No "%" character should appear on this row.
    expect(row.textContent ?? "").not.toMatch(/%/);
  });

  it("renders hit rate, Wilson CI, and observation count when published", () => {
    render(
      <CalibrationStrip
        calibrations={[
          reading({
            bucket_published: true,
            bucket_lo: 70,
            bucket_hi: 80,
            n_observations: 412,
            n_hits: 264,
            hit_rate: 0.641,
            confidence_low: 0.594,
            confidence_high: 0.685,
            last_updated: "2026-05-21T03:00:00Z",
          }),
        ]}
      />,
    );
    const row = screen.getByTestId("calibration-row-pullback_risk");
    expect(row.getAttribute("data-status")).toBe("published");
    expect(row).toHaveTextContent(/64%/);
    expect(row).toHaveTextContent(/closed lower/);
    expect(row).toHaveTextContent(/CI 59–69%/);
    expect(row).toHaveTextContent(/412 obs/);
  });

  it("flips the language for return_positive outcomes", () => {
    render(
      <CalibrationStrip
        calibrations={[
          reading({
            score_name: "rebound_potential",
            outcome_name: "return_positive",
            bucket_published: true,
            bucket_lo: 0,
            bucket_hi: 10,
            n_observations: 60,
            n_hits: 35,
            hit_rate: 0.583,
            confidence_low: 0.456,
            confidence_high: 0.7,
            last_updated: "2026-05-21T03:00:00Z",
          }),
        ]}
      />,
    );
    const row = screen.getByTestId("calibration-row-rebound_potential");
    expect(row).toHaveTextContent(/closed higher/);
    // Hit rate > 50% on a positive outcome → bull tone.
    expect(row.innerHTML).toContain("text-bull");
  });

  it("uses muted tone when hit rate is below 50% (no signal)", () => {
    render(
      <CalibrationStrip
        calibrations={[
          reading({
            bucket_published: true,
            bucket_lo: 70,
            bucket_hi: 80,
            n_observations: 200,
            n_hits: 80,
            hit_rate: 0.4,
            confidence_low: 0.33,
            confidence_high: 0.47,
            last_updated: "2026-05-21T03:00:00Z",
          }),
        ]}
      />,
    );
    const row = screen.getByTestId("calibration-row-pullback_risk");
    expect(row.innerHTML).toContain("text-muted-foreground");
  });

  it("renders multiple readings as separate rows", () => {
    render(
      <CalibrationStrip
        calibrations={[
          reading({ score_name: "pullback_risk" }),
          reading({
            score_name: "rebound_potential",
            outcome_name: "return_positive",
          }),
        ]}
      />,
    );
    expect(
      screen.getByTestId("calibration-row-pullback_risk"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("calibration-row-rebound_potential"),
    ).toBeInTheDocument();
  });
});
