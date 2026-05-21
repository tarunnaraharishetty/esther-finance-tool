import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Pulse } from "./Pulse";
import type { MarketPulse, PulseEvolution } from "@/lib/types";

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

describe("Pulse", () => {
  it("renders the no-pulse empty state when pulse is null", () => {
    render(<Pulse pulse={null} evolution={null} />);
    expect(screen.getByText("no pulse yet")).toBeInTheDocument();
  });

  it("omits the STRONG section entirely when strongest_symbols is empty", () => {
    // Anti-spam invariant — preserved from the legacy PulseCard.
    render(<Pulse pulse={pulseWithNoStrong()} evolution={null} />);
    expect(screen.queryByTestId("pulse-strong")).not.toBeInTheDocument();
    expect(screen.queryByText(/strong signals/i)).not.toBeInTheDocument();
  });

  it("renders a buy-directed chip for STRONG BUY entries", () => {
    const pulse: MarketPulse = {
      ...pulseWithNoStrong(),
      strongest_symbols: [["NVDA", "STRONG BUY"]],
    };
    render(<Pulse pulse={pulse} evolution={null} />);
    const section = screen.getByTestId("pulse-strong");
    expect(section).toBeInTheDocument();
    expect(screen.getByText("NVDA")).toBeInTheDocument();
    expect(screen.getByText("STRONG BUY")).toBeInTheDocument();
    const chip = section.querySelector("[data-direction]");
    expect(chip?.getAttribute("data-direction")).toBe("buy");
  });

  it("renders a sell-directed chip for STRONG SELL entries", () => {
    const pulse: MarketPulse = {
      ...pulseWithNoStrong(),
      strongest_symbols: [["TSLA", "STRONG SELL"]],
    };
    render(<Pulse pulse={pulse} evolution={null} />);
    const chip = screen
      .getByTestId("pulse-strong")
      .querySelector("[data-direction]");
    expect(chip?.getAttribute("data-direction")).toBe("sell");
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
    render(<Pulse pulse={pulse} evolution={null} />);
    const chips = screen
      .getByTestId("pulse-strong")
      .querySelectorAll("[data-direction]");
    expect(chips.length).toBe(3);
    const buys = screen
      .getByTestId("pulse-strong")
      .querySelectorAll("[data-direction=buy]");
    const sells = screen
      .getByTestId("pulse-strong")
      .querySelectorAll("[data-direction=sell]");
    expect(buys.length).toBe(2);
    expect(sells.length).toBe(1);
  });

  it("still renders the regular pulse content alongside STRONG chips", () => {
    const evolution: PulseEvolution = { regime: "risk-on", patterns: [] };
    const pulse: MarketPulse = {
      ...pulseWithNoStrong(),
      strongest_symbols: [["NVDA", "STRONG BUY"]],
    };
    render(<Pulse pulse={pulse} evolution={evolution} />);
    expect(screen.getByText(pulse.summary)).toBeInTheDocument();
    expect(screen.getByText("bullish")).toBeInTheDocument();
    expect(screen.getByText(/regime/)).toHaveTextContent("risk-on");
    expect(screen.getByTestId("pulse-strong")).toBeInTheDocument();
  });
});
