import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ValidationBadge } from "./ValidationBadge";
import type { ValidationReport } from "@/lib/research";

function report(over: Partial<ValidationReport> = {}): ValidationReport {
  return { drop_count: 0, dropped_claims: [], ...over };
}

describe("ValidationBadge", () => {
  it("renders the bull tone when zero claims were dropped", () => {
    render(<ValidationBadge validation={report()} />);
    const badge = screen.getByTestId("validation-badge");
    expect(badge.getAttribute("data-drop-count")).toBe("0");
    expect(badge.className).toContain("text-bull");
    expect(badge).toHaveTextContent(/Grounded · 0 claims dropped/);
  });

  it("renders the warn tone and shows the count when drops occurred", () => {
    render(
      <ValidationBadge
        validation={report({
          drop_count: 2,
          dropped_claims: [
            {
              section: "technical_analysis.body",
              sentence: "RSI is at 87.4, in deep oversold territory.",
              unsupported_tokens: ["87.4"],
            },
            {
              section: "metrics[2].value",
              sentence: "Made-up margin: 42.7%",
              unsupported_tokens: ["42.7%"],
            },
          ],
        })}
      />,
    );
    const badge = screen.getByTestId("validation-badge");
    expect(badge.getAttribute("data-drop-count")).toBe("2");
    expect(badge.className).toContain("text-warn");
    expect(badge).toHaveTextContent(/Grounded · 2 claims dropped/);
  });

  it("uses singular 'claim' for exactly one drop", () => {
    render(
      <ValidationBadge
        validation={report({
          drop_count: 1,
          dropped_claims: [
            {
              section: "tagline",
              sentence: "Watch for the 99.9% beat.",
              unsupported_tokens: ["99.9%"],
            },
          ],
        })}
      />,
    );
    const badge = screen.getByTestId("validation-badge");
    expect(badge).toHaveTextContent(/1 claim dropped/);
    expect(badge.textContent ?? "").not.toMatch(/claims dropped/);
  });

  it("surfaces dropped-claim text in the title tooltip for auditing", () => {
    render(
      <ValidationBadge
        validation={report({
          drop_count: 1,
          dropped_claims: [
            {
              section: "technical_analysis.body",
              sentence: "RSI is at 87.4 in oversold territory.",
              unsupported_tokens: ["87.4"],
            },
          ],
        })}
      />,
    );
    const badge = screen.getByTestId("validation-badge");
    expect(badge.getAttribute("title") ?? "").toContain("technical_analysis.body");
    expect(badge.getAttribute("title") ?? "").toContain("87.4");
  });

  it("uses the positive trust-framing message when no drops occurred", () => {
    render(<ValidationBadge validation={report()} />);
    const badge = screen.getByTestId("validation-badge");
    // Tooltip codifies the contract: "every claim anchored" — not just
    // "no drops" — so the trust framing is unambiguous in markup too.
    expect(badge.getAttribute("title") ?? "").toContain("anchored");
  });
});
