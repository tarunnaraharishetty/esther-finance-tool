import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorBoundary } from "./ErrorBoundary";

// A child that throws once, then renders cleanly when re-mounted.
// `shouldThrow` is a module-level toggle so the Retry test can flip
// it between renders.
let shouldThrow = true;
function Bomb({ message = "kaboom" }: { message?: string }) {
  if (shouldThrow) {
    throw new Error(message);
  }
  return <div data-testid="bomb-ok">recovered</div>;
}

describe("ErrorBoundary", () => {
  // Suppress React's internal error logging for the duration of the
  // suite so the test output stays readable. componentDidCatch's
  // explicit console.error is what we exercise below.
  let consoleSpy: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    shouldThrow = true;
    consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    consoleSpy.mockRestore();
  });

  it("renders children when nothing throws", () => {
    shouldThrow = false;
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("bomb-ok")).toBeInTheDocument();
  });

  it("renders the fallback UI when a child throws", () => {
    render(
      <ErrorBoundary>
        <Bomb message="render exploded" />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText(/render exploded/)).toBeInTheDocument();
    // Both action buttons are present.
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reload/i })).toBeInTheDocument();
  });

  it("logs caught errors to console.error", () => {
    render(
      <ErrorBoundary>
        <Bomb message="logged" />
      </ErrorBoundary>,
    );
    // First call carries our prefix, the error, and the React info object.
    const calls: unknown[][] = consoleSpy.mock.calls;
    const found = calls.some(
      (args) =>
        typeof args[0] === "string" &&
        (args[0] as string).includes("ErrorBoundary caught"),
    );
    expect(found).toBe(true);
  });

  it("clears the error and re-renders children on Retry", () => {
    // The Retry button calls setState to clear the error. We pair it
    // with toggling `shouldThrow` to false to simulate "the
    // underlying cause went away" (e.g. a one-off bad SSE payload
    // followed by a clean one).
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();

    shouldThrow = false;
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByTestId("bomb-ok")).toBeInTheDocument();
  });
});
