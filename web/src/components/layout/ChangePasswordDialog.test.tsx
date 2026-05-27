import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { ChangePasswordDialog } from "./ChangePasswordDialog";

describe("ChangePasswordDialog", () => {
  it("disables the submit button until all three fields are populated", () => {
    render(
      <ChangePasswordDialog
        onClose={vi.fn()}
        onSuccess={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );
    const submit = screen.getByTestId("change-submit") as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
    fireEvent.change(screen.getByTestId("current-password-input"), {
      target: { value: "oldpw1234" },
    });
    expect(submit.disabled).toBe(true);
    fireEvent.change(screen.getByTestId("new-password-input"), {
      target: { value: "newpw5678" },
    });
    expect(submit.disabled).toBe(true);
    fireEvent.change(screen.getByTestId("confirm-new-password-input"), {
      target: { value: "newpw5678" },
    });
    expect(submit.disabled).toBe(false);
  });

  it("flags a local mismatch before sending the request", () => {
    const onConfirm = vi.fn();
    render(
      <ChangePasswordDialog
        onClose={vi.fn()}
        onSuccess={vi.fn()}
        onConfirm={onConfirm}
      />,
    );
    fireEvent.change(screen.getByTestId("current-password-input"), {
      target: { value: "oldpw1234" },
    });
    fireEvent.change(screen.getByTestId("new-password-input"), {
      target: { value: "newpw5678" },
    });
    fireEvent.change(screen.getByTestId("confirm-new-password-input"), {
      target: { value: "different5" },
    });
    fireEvent.click(screen.getByTestId("change-submit"));
    expect(screen.getByTestId("change-mismatch-error")).toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("calls onConfirm with current+new and then onSuccess", async () => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    const onSuccess = vi.fn();
    render(
      <ChangePasswordDialog
        onClose={vi.fn()}
        onSuccess={onSuccess}
        onConfirm={onConfirm}
      />,
    );
    fireEvent.change(screen.getByTestId("current-password-input"), {
      target: { value: "oldpw1234" },
    });
    fireEvent.change(screen.getByTestId("new-password-input"), {
      target: { value: "newpw5678" },
    });
    fireEvent.change(screen.getByTestId("confirm-new-password-input"), {
      target: { value: "newpw5678" },
    });
    fireEvent.click(screen.getByTestId("change-submit"));
    await waitFor(() => {
      expect(onConfirm).toHaveBeenCalledWith("oldpw1234", "newpw5678");
    });
    expect(onSuccess).toHaveBeenCalledTimes(1);
  });

  it("renders the server error on 403 (wrong current password)", async () => {
    const onConfirm = vi.fn().mockRejectedValue({
      status: 403,
      message: "Current password is incorrect.",
    });
    const onSuccess = vi.fn();
    render(
      <ChangePasswordDialog
        onClose={vi.fn()}
        onSuccess={onSuccess}
        onConfirm={onConfirm}
      />,
    );
    fireEvent.change(screen.getByTestId("current-password-input"), {
      target: { value: "wrongguess" },
    });
    fireEvent.change(screen.getByTestId("new-password-input"), {
      target: { value: "newpw5678" },
    });
    fireEvent.change(screen.getByTestId("confirm-new-password-input"), {
      target: { value: "newpw5678" },
    });
    fireEvent.click(screen.getByTestId("change-submit"));
    const err = await screen.findByTestId("change-password-error");
    expect(err).toHaveTextContent(/Current password is incorrect/);
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("closes via backdrop or close button", () => {
    const onClose = vi.fn();
    render(
      <ChangePasswordDialog
        onClose={onClose}
        onSuccess={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("change-password-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("change-password-backdrop"));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
