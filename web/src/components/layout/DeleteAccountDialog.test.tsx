import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { DeleteAccountDialog } from "./DeleteAccountDialog";

describe("DeleteAccountDialog", () => {
  it("disables the Delete button until password + ack are both set", () => {
    render(
      <DeleteAccountDialog
        email="trader@example.com"
        onClose={vi.fn()}
        onSuccess={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );
    const submit = screen.getByTestId("delete-confirm") as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
    // Type password only — still disabled (ack missing).
    fireEvent.change(screen.getByTestId("delete-password-input"), {
      target: { value: "pw1234" },
    });
    expect(submit.disabled).toBe(true);
    // Check ack — now enabled.
    fireEvent.click(screen.getByTestId("delete-ack-checkbox"));
    expect(submit.disabled).toBe(false);
  });

  it("calls onConfirm with the password and then onSuccess", async () => {
    const onConfirm = vi.fn().mockResolvedValue(undefined);
    const onSuccess = vi.fn();
    render(
      <DeleteAccountDialog
        email="trader@example.com"
        onClose={vi.fn()}
        onSuccess={onSuccess}
        onConfirm={onConfirm}
      />,
    );
    fireEvent.change(screen.getByTestId("delete-password-input"), {
      target: { value: "rightpw1234" },
    });
    fireEvent.click(screen.getByTestId("delete-ack-checkbox"));
    fireEvent.click(screen.getByTestId("delete-confirm"));
    await waitFor(() => {
      expect(onConfirm).toHaveBeenCalledWith("rightpw1234");
    });
    expect(onSuccess).toHaveBeenCalledTimes(1);
  });

  it("renders the server error on 403 (wrong password) and does not call onSuccess", async () => {
    const onConfirm = vi.fn().mockRejectedValue({
      status: 403,
      message: "Password is incorrect.",
    });
    const onSuccess = vi.fn();
    render(
      <DeleteAccountDialog
        email="trader@example.com"
        onClose={vi.fn()}
        onSuccess={onSuccess}
        onConfirm={onConfirm}
      />,
    );
    fireEvent.change(screen.getByTestId("delete-password-input"), {
      target: { value: "wrongguess" },
    });
    fireEvent.click(screen.getByTestId("delete-ack-checkbox"));
    fireEvent.click(screen.getByTestId("delete-confirm"));
    const err = await screen.findByTestId("delete-dialog-error");
    expect(err).toHaveTextContent(/Password is incorrect/);
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("calls onClose when the backdrop or close button is clicked", () => {
    const onClose = vi.fn();
    render(
      <DeleteAccountDialog
        email="trader@example.com"
        onClose={onClose}
        onSuccess={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("delete-dialog-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("delete-dialog-backdrop"));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
