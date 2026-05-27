import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { VerifyEmailPage } from "./VerifyEmailPage";

describe("VerifyEmailPage", () => {
  it("POSTs the token on mount and renders the success screen", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<VerifyEmailPage token="t-abc" onSuccess={vi.fn()} />);
    await waitFor(() => {
      expect(screen.getByTestId("verify-page").getAttribute("data-state")).toBe(
        "ok",
      );
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/verify/confirm",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ token: "t-abc" }),
      }),
    );
    vi.unstubAllGlobals();
  });

  it("renders the server error message on a 400 (expired / invalid token)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: "Verification link is invalid or has expired.",
        }),
        {
          status: 400,
          statusText: "Bad Request",
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<VerifyEmailPage token="t-bad" onSuccess={vi.fn()} />);
    const err = await screen.findByTestId("verify-error");
    expect(err).toHaveTextContent(/invalid or has expired/);
    vi.unstubAllGlobals();
  });

  it("calls onSuccess when the Continue button is clicked", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const onSuccess = vi.fn();

    render(<VerifyEmailPage token="t-abc" onSuccess={onSuccess} />);
    const cont = await screen.findByTestId("verify-continue");
    fireEvent.click(cont);
    expect(onSuccess).toHaveBeenCalledTimes(1);
    vi.unstubAllGlobals();
  });
});
