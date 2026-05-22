import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { FreshnessBadge } from "./FreshnessBadge";
import type { FundamentalsFreshness } from "@/lib/analyzer";

function freshness(over: Partial<FundamentalsFreshness>): FundamentalsFreshness {
  return {
    as_of: "2026-03-31T00:00:00Z",
    fetched_at: "2026-05-22T00:00:00Z",
    freshness: "fresh",
    data_age_days: 51,
    source_chain: ["fmp"],
    provider_confidence: 1.0,
    divergence_count: 0,
    reconciliation_warning_count: 0,
    ...over,
  };
}

describe("FreshnessBadge", () => {
  it("renders nothing when freshness is null", () => {
    const { container } = render(<FreshnessBadge freshness={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the fresh tier with bull palette and a colored dot", () => {
    render(<FreshnessBadge freshness={freshness({ freshness: "fresh" })} />);
    const chip = screen
      .getByTestId("freshness-badge")
      .querySelector('[data-freshness="fresh"]');
    expect(chip).not.toBeNull();
    expect(chip!.className).toContain("text-bull");
  });

  it.each(["aging", "stale", "expired"] as const)(
    "renders the %s tier with the warn/bear palette",
    (tier) => {
      render(<FreshnessBadge freshness={freshness({ freshness: tier })} />);
      const chip = screen
        .getByTestId("freshness-badge")
        .querySelector(`[data-freshness="${tier}"]`);
      expect(chip).not.toBeNull();
      if (tier === "aging") {
        expect(chip!.className).toContain("text-warn");
      } else {
        expect(chip!.className).toContain("text-bear");
      }
    },
  );

  it("includes the source chain when populated", () => {
    render(
      <FreshnessBadge
        freshness={freshness({ source_chain: ["fmp", "finnhub"] })}
      />,
    );
    expect(screen.getByText(/via\s+fmp\s+→\s+finnhub/)).toBeInTheDocument();
  });

  it("omits the divergence chip when count is zero", () => {
    render(<FreshnessBadge freshness={freshness({ divergence_count: 0 })} />);
    expect(screen.queryByTestId("freshness-divergence")).not.toBeInTheDocument();
  });

  it("renders the divergence chip when count is non-zero", () => {
    render(<FreshnessBadge freshness={freshness({ divergence_count: 2 })} />);
    const chip = screen.getByTestId("freshness-divergence");
    expect(chip).toBeInTheDocument();
    expect(chip).toHaveTextContent(/2/);
    expect(chip).toHaveTextContent(/disagree/i);
  });

  it("formats age as days for under-60-day values", () => {
    render(<FreshnessBadge freshness={freshness({ data_age_days: 51 })} />);
    expect(screen.getByText(/51d/)).toBeInTheDocument();
  });

  it("formats age as months past 60 days", () => {
    render(<FreshnessBadge freshness={freshness({ data_age_days: 180 })} />);
    expect(screen.getByText(/6mo/)).toBeInTheDocument();
  });
});
