import { describe, expect, it } from "vitest";
import { fmtScore, parseClaimCitations, scoreTone } from "./analyzer";

describe("parseClaimCitations", () => {
  it("extracts a single tag and strips it from the displayed text", () => {
    const { text, tags } = parseClaimCitations("RSI is 64 [technicals].");
    // The citation regex eats the leading whitespace too, so the
    // rendered prose ends "...64." not "...64 .".
    expect(text).toBe("RSI is 64.");
    expect(tags).toEqual(["technicals"]);
  });

  it("extracts multiple tags from a comma-separated group", () => {
    const { text, tags } = parseClaimCitations(
      "Tape and tape agree [sentiment, volume].",
    );
    expect(text).toBe("Tape and tape agree.");
    expect(tags).toEqual(["sentiment", "volume"]);
  });

  it("ignores unknown tag names rather than crashing the renderer", () => {
    const { tags } = parseClaimCitations("Vibes good [vibes].");
    // Unknown tags are silently dropped client-side — the server-side
    // validator is the actual guard; the client just renders chips.
    expect(tags).toEqual([]);
  });

  it("returns the original text when no citation is present", () => {
    const { text, tags } = parseClaimCitations("No tags here.");
    expect(text).toBe("No tags here.");
    expect(tags).toEqual([]);
  });
});

describe("fmtScore", () => {
  it("rounds finite numbers to integers", () => {
    expect(fmtScore(67.4)).toBe("67");
    expect(fmtScore(0)).toBe("0");
  });

  it("renders an em-dash for null / NaN / undefined", () => {
    expect(fmtScore(null)).toBe("—");
    expect(fmtScore(undefined)).toBe("—");
    expect(fmtScore(Number.NaN)).toBe("—");
  });
});

describe("scoreTone", () => {
  it("maps score buckets to consistent tones", () => {
    expect(scoreTone(85)).toBe("warn");
    expect(scoreTone(55)).toBe("bull");
    expect(scoreTone(25)).toBe("muted");
    expect(scoreTone(5)).toBe("muted");
    expect(scoreTone(null)).toBe("muted");
  });
});
