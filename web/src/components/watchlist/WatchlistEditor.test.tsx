import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { WatchlistEditor } from "./WatchlistEditor";

describe("WatchlistEditor", () => {
  it("renders one chip per symbol", () => {
    render(
      <WatchlistEditor
        symbols={["AAPL", "MSFT", "NVDA"]}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        resolved
      />,
    );
    expect(screen.getByTestId("watchlist-editor-chip-AAPL")).toBeInTheDocument();
    expect(screen.getByTestId("watchlist-editor-chip-MSFT")).toBeInTheDocument();
    expect(screen.getByTestId("watchlist-editor-chip-NVDA")).toBeInTheDocument();
  });

  it("submitting the add form calls onAdd with the upper-cased symbol", async () => {
    const onAdd = vi.fn().mockResolvedValue(undefined);
    render(
      <WatchlistEditor
        symbols={["AAPL"]}
        onAdd={onAdd}
        onRemove={vi.fn()}
        resolved
      />,
    );
    fireEvent.change(screen.getByTestId("watchlist-editor-input"), {
      target: { value: "nvda" },
    });
    fireEvent.click(screen.getByTestId("watchlist-editor-add"));
    await waitFor(() => {
      expect(onAdd).toHaveBeenCalledWith("NVDA");
    });
  });

  it("clears the input after a successful add", async () => {
    const onAdd = vi.fn().mockResolvedValue(undefined);
    render(
      <WatchlistEditor
        symbols={["AAPL"]}
        onAdd={onAdd}
        onRemove={vi.fn()}
        resolved
      />,
    );
    const input = screen.getByTestId("watchlist-editor-input") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "NVDA" } });
    fireEvent.click(screen.getByTestId("watchlist-editor-add"));
    await waitFor(() => {
      expect(input.value).toBe("");
    });
  });

  it("guards against adding a symbol already on the list (local check)", async () => {
    const onAdd = vi.fn().mockResolvedValue(undefined);
    render(
      <WatchlistEditor
        symbols={["AAPL"]}
        onAdd={onAdd}
        onRemove={vi.fn()}
        resolved
      />,
    );
    fireEvent.change(screen.getByTestId("watchlist-editor-input"), {
      target: { value: "aapl" },
    });
    fireEvent.click(screen.getByTestId("watchlist-editor-add"));
    // Local dup-check should short-circuit before calling onAdd.
    await waitFor(() => {
      expect(screen.getByTestId("watchlist-editor-error")).toBeInTheDocument();
    });
    expect(onAdd).not.toHaveBeenCalled();
  });

  it("clicking the remove button invokes onRemove for that symbol", async () => {
    const onRemove = vi.fn().mockResolvedValue(undefined);
    render(
      <WatchlistEditor
        symbols={["AAPL", "MSFT"]}
        onAdd={vi.fn()}
        onRemove={onRemove}
        resolved
      />,
    );
    fireEvent.click(screen.getByTestId("watchlist-editor-remove-MSFT"));
    await waitFor(() => {
      expect(onRemove).toHaveBeenCalledWith("MSFT");
    });
  });

  it("renders the empty-state with starter symbols when the list is empty + resolved", () => {
    render(
      <WatchlistEditor
        symbols={[]}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        resolved
      />,
    );
    expect(screen.getByTestId("watchlist-editor-empty")).toBeInTheDocument();
    // At least one starter button must be available.
    expect(
      screen.getByTestId("watchlist-editor-starter-AAPL"),
    ).toBeInTheDocument();
  });

  it("does NOT show the empty-state until resolved (avoids first-load flash)", () => {
    render(
      <WatchlistEditor
        symbols={[]}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        resolved={false}
      />,
    );
    expect(screen.queryByTestId("watchlist-editor-empty")).toBeNull();
  });

  it("clicking a starter symbol invokes onAdd", async () => {
    const onAdd = vi.fn().mockResolvedValue(undefined);
    render(
      <WatchlistEditor
        symbols={[]}
        onAdd={onAdd}
        onRemove={vi.fn()}
        resolved
      />,
    );
    fireEvent.click(screen.getByTestId("watchlist-editor-starter-MSFT"));
    await waitFor(() => {
      expect(onAdd).toHaveBeenCalledWith("MSFT");
    });
  });

  it("surfaces an error from onAdd", async () => {
    const onAdd = vi.fn().mockRejectedValue({
      status: 413,
      message: "Watchlist is full.",
    });
    render(
      <WatchlistEditor
        symbols={["AAPL"]}
        onAdd={onAdd}
        onRemove={vi.fn()}
        resolved
      />,
    );
    fireEvent.change(screen.getByTestId("watchlist-editor-input"), {
      target: { value: "NVDA" },
    });
    fireEvent.click(screen.getByTestId("watchlist-editor-add"));
    const err = await screen.findByTestId("watchlist-editor-error");
    expect(err).toHaveTextContent(/Watchlist is full/);
  });

  it("disables interactions when busy=true", () => {
    render(
      <WatchlistEditor
        symbols={["AAPL"]}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        resolved
        busy
      />,
    );
    expect(screen.getByTestId("watchlist-editor-input")).toBeDisabled();
    expect(
      screen.getByTestId("watchlist-editor-remove-AAPL"),
    ).toBeDisabled();
  });
});
