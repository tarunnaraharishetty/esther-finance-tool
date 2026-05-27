import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { VerifyEmailBanner } from "./VerifyEmailBanner";

describe("VerifyEmailBanner", () => {
  it("renders the email + a Resend button", () => {
    render(<VerifyEmailBanner email="trader@example.com" onResend={vi.fn()} />);
    expect(screen.getByTestId("verify-email-banner")).toHaveTextContent(
      /Verify your email/,
    );
    expect(screen.getByTestId("verify-email-banner")).toHaveTextContent(
      /trader@example.com/,
    );
    expect(screen.getByTestId("verify-banner-resend")).toBeInTheDocument();
  });

  it("collapses to a Sent badge after a successful resend", async () => {
    const onResend = vi.fn().mockResolvedValue(undefined);
    render(<VerifyEmailBanner email="trader@example.com" onResend={onResend} />);
    fireEvent.click(screen.getByTestId("verify-banner-resend"));
    await waitFor(() => {
      expect(screen.queryByTestId("verify-banner-resend")).toBeNull();
    });
    expect(screen.getByTestId("verify-banner-sent")).toBeInTheDocument();
    expect(onResend).toHaveBeenCalledTimes(1);
  });

  it("renders the server error message on rate-limit / failure", async () => {
    const onResend = vi.fn().mockRejectedValue({
      status: 429,
      message: "Rate limit exceeded. Slow down and try again.",
    });
    render(<VerifyEmailBanner email="trader@example.com" onResend={onResend} />);
    fireEvent.click(screen.getByTestId("verify-banner-resend"));
    const err = await screen.findByTestId("verify-banner-error");
    expect(err).toHaveTextContent(/Rate limit exceeded/);
    // Sent badge must not appear when resend failed — user should be
    // able to retry.
    expect(screen.queryByTestId("verify-banner-sent")).toBeNull();
    expect(screen.queryByTestId("verify-banner-resend")).not.toBeNull();
  });
});
