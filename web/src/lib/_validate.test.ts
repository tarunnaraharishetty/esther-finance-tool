/**
 * Tests for the API-boundary validation helper (B-18).
 *
 * Pins the contract every hook depends on: a schema mismatch surfaces
 * as a clean one-line Error message naming the offending field path,
 * which the existing ``error: string | null`` UI renders verbatim.
 */

import { describe, expect, it } from "vitest";
import { z } from "zod";

import { describeZodIssue, validateJson } from "./_validate";
import {
  ResearchThesisSchema,
  type ResearchThesis,
} from "./research";

/** Build a Response whose ``json()`` resolves to ``payload``. */
function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("validateJson", () => {
  const Tiny = z.object({ a: z.number(), b: z.string() });

  it("returns the parsed object on a matching payload", async () => {
    const res = jsonResponse({ a: 1, b: "hello" });
    const parsed = await validateJson(res, Tiny, "tiny");
    expect(parsed).toEqual({ a: 1, b: "hello" });
  });

  it("throws an Error whose message starts with the where label", async () => {
    const res = jsonResponse({ a: "not-a-number", b: "hi" });
    await expect(validateJson(res, Tiny, "tiny")).rejects.toThrow(
      /^tiny: /,
    );
  });

  it("names the offending field path in the error message", async () => {
    const res = jsonResponse({ a: 1 }); // missing b
    let caught: Error | null = null;
    try {
      await validateJson(res, Tiny, "tiny");
    } catch (e) {
      caught = e instanceof Error ? e : null;
    }
    expect(caught).not.toBeNull();
    // ``b`` is the missing field — the message must mention it so an
    // operator triaging "tiny: ..." in the UI can grep for the path.
    expect(caught!.message).toContain("b");
  });

  it("surfaces a non-JSON response as a JSON parse error", async () => {
    const res = new Response("<html>oops</html>", {
      status: 200,
      headers: { "Content-Type": "text/html" },
    });
    await expect(validateJson(res, Tiny, "tiny")).rejects.toThrow(
      /was not valid JSON/,
    );
  });
});

describe("describeZodIssue", () => {
  it("renders root-level issues with the <root> placeholder", () => {
    const result = z.string().safeParse(123);
    expect(result.success).toBe(false);
    if (!result.success) {
      const line = describeZodIssue(result.error.issues[0]);
      expect(line.startsWith("<root>:")).toBe(true);
    }
  });

  it("joins nested paths with dots", () => {
    const Schema = z.object({ a: z.object({ b: z.string() }) });
    const result = Schema.safeParse({ a: { b: 1 } });
    expect(result.success).toBe(false);
    if (!result.success) {
      const line = describeZodIssue(result.error.issues[0]);
      expect(line.startsWith("a.b:")).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// End-to-end: the real ResearchThesisSchema accepts a good payload and
// rejects the kind of partial-deploy drift B-18 is about.
// ---------------------------------------------------------------------------

function validResearchPayload(): ResearchThesis {
  const section = {
    title: "T",
    body: "B",
    bullets: [],
    provenance: {},
  };
  return {
    symbol: "AAPL",
    generated_at: "2026-05-25T00:00:00+00:00",
    model: "claude-opus-4-7@thesis-v1",
    rating: "buy",
    confidence: 0.72,
    tagline: "Apple looks constructive.",
    company_overview: section,
    bull_thesis: [],
    bear_thesis: [],
    technical_analysis: section,
    fundamental_analysis: section,
    metrics: [],
    sentiment_news: section,
    catalysts: [],
    risk_assessment: section,
    outlook: [],
    explainability: section,
    data_sources: [],
    disclaimers: [],
    cache: "miss",
    cache_age_seconds: 0,
    mode: "llm",
    validation: { drop_count: 0, dropped_claims: [] },
    trust_score: {
      score: 88,
      grade: "B",
      components: [],
    },
  };
}

describe("ResearchThesisSchema", () => {
  it("accepts a well-formed payload", async () => {
    const res = jsonResponse(validResearchPayload());
    const parsed = await validateJson(res, ResearchThesisSchema, "research");
    expect(parsed.symbol).toBe("AAPL");
    expect(parsed.rating).toBe("buy");
  });

  it("rejects a payload missing a required nested field", async () => {
    // Simulate a backend deploy that dropped the validation.drop_count
    // column — a partial deploy where the dataclass was updated but
    // the serializer wasn't. The hook MUST surface this as a clean
    // error rather than crashing later when the badge reads .drop_count.
    const bad = validResearchPayload() as unknown as Record<string, unknown>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (bad.validation as any) = { dropped_claims: [] };
    const res = jsonResponse(bad);
    await expect(
      validateJson(res, ResearchThesisSchema, "research"),
    ).rejects.toThrow(/validation\.drop_count/);
  });

  it("rejects an unknown rating value", async () => {
    const bad = validResearchPayload() as unknown as Record<string, unknown>;
    bad.rating = "absolutely_yes";
    const res = jsonResponse(bad);
    await expect(
      validateJson(res, ResearchThesisSchema, "research"),
    ).rejects.toThrow(/rating/);
  });
});
