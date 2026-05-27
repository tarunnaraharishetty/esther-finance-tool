import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { LoginPage } from "./LoginPage";
import type { AuthUser } from "@/lib/auth";

function makeUser(over: Partial<AuthUser> = {}): AuthUser {
  return {
    id: 1,
    email: "a@b.com",
    created_at: "2026-05-24T00:00:00+00:00",
    last_login_at: null,
    ...over,
  };
}

describe("LoginPage", () => {
  it("starts in login mode by default", () => {
    render(
      <LoginPage
        onLogin={vi.fn()}
        onSignup={vi.fn()}
      />,
    );
    expect(screen.getByTestId("login-page").getAttribute("data-mode")).toBe(
      "login",
    );
    expect(screen.getByTestId("submit-button")).toHaveTextContent(/Log in/i);
  });

  it("toggles to signup mode + reveals the password hint", () => {
    render(<LoginPage onLogin={vi.fn()} onSignup={vi.fn()} />);
    fireEvent.click(screen.getByTestId("mode-signup"));
    expect(screen.getByTestId("login-page").getAttribute("data-mode")).toBe(
      "signup",
    );
    expect(screen.getByTestId("submit-button")).toHaveTextContent(
      /Create account/i,
    );
    expect(screen.getByTestId("login-page")).toHaveTextContent(
      /Minimum 8 characters/,
    );
  });

  it("invokes onLogin with the entered credentials on submit", async () => {
    const onLogin = vi.fn().mockResolvedValue(makeUser());
    render(<LoginPage onLogin={onLogin} onSignup={vi.fn()} />);

    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "a@b.com" },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "longenough" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(onLogin).toHaveBeenCalledWith("a@b.com", "longenough");
    });
  });

  it("invokes onSignup when in signup mode", async () => {
    const onSignup = vi.fn().mockResolvedValue(makeUser());
    render(<LoginPage onLogin={vi.fn()} onSignup={onSignup} />);

    fireEvent.click(screen.getByTestId("mode-signup"));
    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "new@user.com" },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "longenough" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(onSignup).toHaveBeenCalledWith("new@user.com", "longenough");
    });
  });

  it("trims whitespace from the email before submitting", async () => {
    const onLogin = vi.fn().mockResolvedValue(makeUser());
    render(<LoginPage onLogin={onLogin} onSignup={vi.fn()} />);

    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "  a@b.com  " },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "longenough" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));

    await waitFor(() => {
      expect(onLogin).toHaveBeenCalledWith("a@b.com", "longenough");
    });
  });

  it("renders the error message when login fails", async () => {
    const onLogin = vi.fn().mockRejectedValue({
      status: 401,
      message: "Invalid email or password.",
    });
    render(<LoginPage onLogin={onLogin} onSignup={vi.fn()} />);

    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "a@b.com" },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "wrongguess" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));

    const err = await screen.findByTestId("login-error");
    expect(err).toHaveTextContent(/Invalid email or password/);
  });

  it("clears the error when switching modes", async () => {
    const onLogin = vi.fn().mockRejectedValue({
      status: 401,
      message: "Invalid email or password.",
    });
    render(<LoginPage onLogin={onLogin} onSignup={vi.fn()} />);

    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "a@b.com" },
    });
    fireEvent.change(screen.getByTestId("password-input"), {
      target: { value: "wrongguess" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));
    await screen.findByTestId("login-error");

    fireEvent.click(screen.getByTestId("mode-signup"));
    expect(screen.queryByTestId("login-error")).toBeNull();
  });

  // ---- Password reset (forgot) mode --------------------------------

  it("exposes a 'Forgot password?' link only on the login tab", () => {
    render(<LoginPage onLogin={vi.fn()} onSignup={vi.fn()} />);
    expect(screen.queryByTestId("forgot-link")).not.toBeNull();
    fireEvent.click(screen.getByTestId("mode-signup"));
    expect(screen.queryByTestId("forgot-link")).toBeNull();
  });

  it("switches to forgot mode and hides the password field", () => {
    render(<LoginPage onLogin={vi.fn()} onSignup={vi.fn()} />);
    fireEvent.click(screen.getByTestId("forgot-link"));
    expect(screen.getByTestId("login-page").getAttribute("data-mode")).toBe(
      "forgot",
    );
    // The mode toggle is replaced by a back-link.
    expect(screen.queryByTestId("password-input")).toBeNull();
    expect(screen.queryByTestId("back-to-login")).not.toBeNull();
    expect(screen.getByTestId("submit-button")).toHaveTextContent(
      /Send reset link/i,
    );
  });

  it("POSTs the email + renders the uniform success message", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          ok: true,
          message: "If that email is registered, a reset link is on its way.",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<LoginPage onLogin={vi.fn()} onSignup={vi.fn()} />);
    fireEvent.click(screen.getByTestId("forgot-link"));
    fireEvent.change(screen.getByTestId("email-input"), {
      target: { value: "anyone@example.com" },
    });
    fireEvent.click(screen.getByTestId("submit-button"));

    const success = await screen.findByTestId("forgot-success");
    expect(success).toHaveTextContent(/reset link is on its way/i);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/password-reset/request",
      expect.objectContaining({
        method: "POST",
        credentials: "same-origin",
        body: JSON.stringify({ email: "anyone@example.com" }),
      }),
    );

    vi.unstubAllGlobals();
  });

  it("renders a flash banner when passed flashMessage", () => {
    render(
      <LoginPage
        onLogin={vi.fn()}
        onSignup={vi.fn()}
        flashMessage="Password updated. Please log in."
      />,
    );
    expect(screen.getByTestId("login-flash")).toHaveTextContent(
      /Password updated\. Please log in\./,
    );
  });
});
