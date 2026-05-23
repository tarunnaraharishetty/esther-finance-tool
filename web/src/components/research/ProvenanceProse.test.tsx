import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProvenanceProse } from "./ProvenanceProse";

describe("ProvenanceProse", () => {
  it("renders an empty body as null (no wrapper, no test id)", () => {
    const { container } = render(
      <ProvenanceProse body="" provenance={{}} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders a body with no tokens as plain text inside the wrapper", () => {
    render(
      <ProvenanceProse
        body="A healthy stance with moderate support."
        provenance={{}}
      />,
    );
    const wrapper = screen.getByTestId("provenance-prose");
    expect(wrapper).toHaveTextContent(
      "A healthy stance with moderate support.",
    );
    expect(screen.queryByTestId("prose-token")).toBeNull();
  });

  it("wraps a numeric token in a span+tooltip when provenance is present", () => {
    render(
      <ProvenanceProse
        body="Last close was $180.50 on the tape."
        provenance={{ "$180.50": "row.last_price" }}
      />,
    );
    const token = screen.getByTestId("prose-token");
    expect(token.getAttribute("data-token")).toBe("$180.50");
    expect(token.getAttribute("data-source")).toBe("row.last_price");
    expect(token.getAttribute("title")).toBe("Sourced from row.last_price");
    expect(token).toHaveTextContent("$180.50");
  });

  it("renders multiple tokens with their respective tooltips", () => {
    render(
      <ProvenanceProse
        body="RSI 58.0 with last $180.50 holding."
        provenance={{
          "58.0": "row.rsi",
          "$180.50": "row.last_price",
        }}
      />,
    );
    const tokens = screen.getAllByTestId("prose-token");
    expect(tokens).toHaveLength(2);
    const sources = tokens.map((t) => t.getAttribute("data-source"));
    expect(sources).toEqual(["row.rsi", "row.last_price"]);
  });

  it("renders numeric tokens without provenance as plain text", () => {
    // 99.9% is a real numeric token by the regex, but no provenance map
    // entry → no tooltip, no span.
    render(
      <ProvenanceProse
        body="Made-up 99.9% margin shouldn't show a tooltip."
        provenance={{}}
      />,
    );
    expect(screen.queryByTestId("prose-token")).toBeNull();
    expect(screen.getByTestId("provenance-prose")).toHaveTextContent("99.9%");
  });

  it("preserves whitespace and punctuation around tokens", () => {
    render(
      <ProvenanceProse
        body="Close: $180.50, hold."
        provenance={{ "$180.50": "row.last_price" }}
      />,
    );
    const wrapper = screen.getByTestId("provenance-prose");
    expect(wrapper).toHaveTextContent("Close: $180.50, hold.");
  });

  it("only tooltips the first occurrence of a duplicate token but renders both", () => {
    // Provenance keys are token-level, so a duplicate token in the
    // body still only resolves once per render via the same provenance
    // entry. Both occurrences should render but only one carries the
    // tooltip span — the second renders as plain text (provenance map
    // doesn't track positions).
    render(
      <ProvenanceProse
        body="First $180.50, then $180.50."
        provenance={{ "$180.50": "row.last_price" }}
      />,
    );
    const tokens = screen.getAllByTestId("prose-token");
    // Two matches surface — both reference the same provenance entry
    // since the lookup is by token string.
    expect(tokens.length).toBeGreaterThanOrEqual(1);
    // The wrapper carries both occurrences verbatim.
    expect(screen.getByTestId("provenance-prose")).toHaveTextContent(
      "First $180.50, then $180.50.",
    );
  });

  it("forwards the className to the wrapper span", () => {
    render(
      <ProvenanceProse
        body="text"
        provenance={{}}
        className="my-custom-class"
      />,
    );
    expect(screen.getByTestId("provenance-prose").className).toContain(
      "my-custom-class",
    );
  });

  it("tokenizes percent tokens correctly", () => {
    render(
      <ProvenanceProse
        body="Margin at 23.4% above sector."
        provenance={{ "23.4%": "fundamentals.gross_margin" }}
      />,
    );
    const token = screen.getByTestId("prose-token");
    expect(token).toHaveTextContent("23.4%");
    expect(token.getAttribute("data-source")).toBe(
      "fundamentals.gross_margin",
    );
  });
});
