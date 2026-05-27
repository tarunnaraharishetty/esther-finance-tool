import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { UserMenu } from "./UserMenu";
import type { AuthUser } from "@/lib/auth";

function makeUser(over: Partial<AuthUser> = {}): AuthUser {
  return {
    id: 1,
    email: "trader@example.com",
    created_at: "2026-05-23T00:00:00+00:00",
    last_login_at: "2026-05-24T13:00:00+00:00",
    email_verified: true,
    ...over,
  };
}

describe("UserMenu", () => {
  it("renders the trigger with the display name derived from email", () => {
    render(<UserMenu user={makeUser()} onLogout={vi.fn()} />);
    const trigger = screen.getByTestId("user-menu-trigger");
    // displayName slices on the @, so "trader@example.com" → "trader".
    expect(trigger).toHaveTextContent("trader");
  });

  it("does not render the popover until clicked", () => {
    render(<UserMenu user={makeUser()} onLogout={vi.fn()} />);
    expect(screen.queryByTestId("user-menu-panel")).toBeNull();
    fireEvent.click(screen.getByTestId("user-menu-trigger"));
    expect(screen.getByTestId("user-menu-panel")).toBeInTheDocument();
  });

  it("surfaces the full email in the popover", () => {
    render(<UserMenu user={makeUser()} onLogout={vi.fn()} />);
    fireEvent.click(screen.getByTestId("user-menu-trigger"));
    expect(screen.getByTestId("user-menu-panel")).toHaveTextContent(
      "trader@example.com",
    );
  });

  it("invokes onLogout when the logout button is clicked", async () => {
    const onLogout = vi.fn().mockResolvedValue(undefined);
    render(<UserMenu user={makeUser()} onLogout={onLogout} />);
    fireEvent.click(screen.getByTestId("user-menu-trigger"));
    fireEvent.click(screen.getByTestId("user-menu-logout"));
    await waitFor(() => {
      expect(onLogout).toHaveBeenCalledTimes(1);
    });
  });

  it("renders the last-login timestamp when present", () => {
    render(
      <UserMenu
        user={makeUser({
          last_login_at: new Date(Date.now() - 5 * 60_000).toISOString(),
        })}
        onLogout={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("user-menu-trigger"));
    expect(screen.getByTestId("user-menu-panel")).toHaveTextContent(
      /last login/i,
    );
  });

  it("omits the last-login row when null", () => {
    render(
      <UserMenu
        user={makeUser({ last_login_at: null })}
        onLogout={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("user-menu-trigger"));
    expect(screen.getByTestId("user-menu-panel")).not.toHaveTextContent(
      /last login/i,
    );
  });
});
