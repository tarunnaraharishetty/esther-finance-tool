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
});
