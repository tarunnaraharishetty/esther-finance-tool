import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { ResetPasswordPage } from "./ResetPasswordPage";

describe("ResetPasswordPage", () => {
  it("requires the two passwords to match", () => {
    render(<ResetPasswordPage token="t-abc" onSuccess={vi.fn()} />);
    fireEvent.change(screen.getByTestId("new-password-input"), {
      target: { value: "newpw1234" },
    });
    fireEvent.change(screen.getByTestId("confirm-password-input"), {
      target: { value: "different5" },
    });
    fireEvent.click(screen.getByTestId("reset-submit"));
    expect(screen.getByTestId("mismatch-error")).toHaveTextContent(
      /don't match/i,
    );
  });

  it("POSTs token + new password and calls onSuccess", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const onSuccess = vi.fn();

    render(<ResetPasswordPage token="t-abc" onSuccess={onSuccess} />);
    fireEvent.change(screen.getByTestId("new-password-input"), {
      target: { value: "newpw1234" },
    });
    fireEvent.change(screen.getByTestId("confirm-password-input"), {
      target: { value: "newpw1234" },
    });
    fireEvent.click(screen.getByTestId("reset-submit"));

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledTimes(1);
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/password-reset/confirm",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ token: "t-abc", new_password: "newpw1234" }),
      }),
    );
    vi.unstubAllGlobals();
  });

  it("renders the server's error on a 400 (invalid / expired token)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: "Reset link is invalid or has expired. Request a new one.",
        }),
        {
          status: 400,
          statusText: "Bad Request",
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<ResetPasswordPage token="t-bad" onSuccess={vi.fn()} />);
    fireEvent.change(screen.getByTestId("new-password-input"), {
      target: { value: "newpw1234" },
    });
    fireEvent.change(screen.getByTestId("confirm-password-input"), {
      target: { value: "newpw1234" },
    });
    fireEvent.click(screen.getByTestId("reset-submit"));

    const err = await screen.findByTestId("reset-error");
    expect(err).toHaveTextContent(/invalid or has expired/);
    vi.unstubAllGlobals();
  });
});
