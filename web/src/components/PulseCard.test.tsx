import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PulseCard } from "./PulseCard";
import type { MarketPulse, PulseEvolution } from "../lib/types";

// Minimal pulse with no STRONG signals. Drives the "section omitted"
// assertion below.
function pulseWithNoStrong(): MarketPulse {
  return {
    sentiment: "bullish",
    conviction: "moderate",
    activity: "active",
    summary: "3 symbols leaning bullish with moderate conviction.",
    bullish_count: 2,
    bearish_count: 0,
    healthy_count: 3,
    momentum_breadth: 0.66,
    sentiment_breadth: 0.5,
    reversal_intensity: 0,
    alert_intensity: 1,
    strongest_symbols: [],
  };
}

describe("PulseCard", () => {
  it("renders the no-pulse empty state when pulse is null", () => {
    render(<PulseCard pulse={null} evolution={null} />);
    expect(screen.getByText("no pulse yet")).toBeInTheDocument();
  });

  it("omits the STRONG section entirely when strongest_symbols is empty", () => {
    // This is the anti-spam invariant: a quiet tick (no STRONG promotion)
    // must not show the STRONG header or an empty list. Anything else
    // and the panel would scream at the trader every tick.
    render(<PulseCard pulse={pulseWithNoStrong()} evolution={null} />);
    expect(screen.queryByTestId("pulse-strong")).not.toBeInTheDocument();
    expect(screen.queryByText("strong signals")).not.toBeInTheDocument();
  });

  it("renders a buy-colored chip for STRONG BUY entries", () => {
    const pulse: MarketPulse = {
      ...pulseWithNoStrong(),
      strongest_symbols: [["NVDA", "STRONG BUY"]],
    };
    render(<PulseCard pulse={pulse} evolution={null} />);
    const section = screen.getByTestId("pulse-strong");
    expect(section).toBeInTheDocument();
    // The trader sees both the symbol and the tier label verbatim.
    expect(screen.getByText("NVDA")).toBeInTheDocument();
    expect(screen.getByText("STRONG BUY")).toBeInTheDocument();
    // Direction-derived class — green-styling for BUY.
    const chip = section.querySelector(".pulse__strong-chip");
    expect(chip).toHaveClass("pulse__strong-chip--buy");
    expect(chip).not.toHaveClass("pulse__strong-chip--sell");
  });

  it("renders a sell-colored chip for STRONG SELL entries", () => {
    const pulse: MarketPulse = {
      ...pulseWithNoStrong(),
      strongest_symbols: [["TSLA", "STRONG SELL"]],
    };
    render(<PulseCard pulse={pulse} evolution={null} />);
    const chip = screen.getByTestId("pulse-strong").querySelector(".pulse__strong-chip");
    expect(chip).toHaveClass("pulse__strong-chip--sell");
    expect(chip).not.toHaveClass("pulse__strong-chip--buy");
  });

  it("renders every entry when multiple STRONG symbols are present", () => {
    const pulse: MarketPulse = {
      ...pulseWithNoStrong(),
      strongest_symbols: [
        ["NVDA", "STRONG BUY"],
        ["TSLA", "STRONG SELL"],
        ["SPY", "STRONG BUY"],
      ],
    };
    render(<PulseCard pulse={pulse} evolution={null} />);
    const chips = screen
      .getByTestId("pulse-strong")
      .querySelectorAll(".pulse__strong-chip");
    expect(chips.length).toBe(3);
    const buys = screen
      .getByTestId("pulse-strong")
      .querySelectorAll(".pulse__strong-chip--buy");
    const sells = screen
      .getByTestId("pulse-strong")
      .querySelectorAll(".pulse__strong-chip--sell");
    expect(buys.length).toBe(2);
    expect(sells.length).toBe(1);
  });

  it("still renders the regular pulse content alongside STRONG chips", () => {
    // STRONG chips are additive — they don't displace the categorical
    // pills, summary, metrics, or regime block.
    const evolution: PulseEvolution = {
      regime: "risk-on",
      patterns: [],
    };
    const pulse: MarketPulse = {
      ...pulseWithNoStrong(),
      strongest_symbols: [["NVDA", "STRONG BUY"]],
    };
    render(<PulseCard pulse={pulse} evolution={evolution} />);
    expect(screen.getByText(pulse.summary)).toBeInTheDocument();
    expect(screen.getByText("bullish")).toBeInTheDocument();
    expect(screen.getByText(/regime/)).toHaveTextContent("risk-on");
    expect(screen.getByTestId("pulse-strong")).toBeInTheDocument();
  });
});
