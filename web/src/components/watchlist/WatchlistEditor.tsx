/**
 * Inline watchlist editor — symbol chips + add input + remove buttons.
 *
 * Renders the current user's tracked symbols as removable chips and
 * exposes a tiny add form for new tickers. Mounted above the existing
 * Watchlist table on the WatchlistPage so the user can edit without
 * leaving the page they're already looking at.
 *
 * Optimistic mutations live in the parent (`useUserWatchlist`); this
 * component is purely presentational + form-handling.
 */

import { useState, type FormEvent } from "react";
import { Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Panel } from "@/components/layout/Panel";
import { cn } from "@/lib/utils";

interface Props {
  /** Current symbols on the user's watchlist, in display order. */
  symbols: string[];
  /** Add a symbol. Async so the parent can run a network call.
   *  May throw — surface the error inline. */
  onAdd: (symbol: string) => Promise<void>;
  /** Remove a symbol. Async; idempotent on missing. */
  onRemove: (symbol: string) => Promise<void>;
  /** Disable inputs while the parent is mid-mutation. */
  busy?: boolean;
  /** Whether the parent has resolved the initial fetch.
   *  Used to gate the empty-state message so we don't say
   *  "your watchlist is empty" before the GET settles. */
  resolved: boolean;
}

const STARTER_SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"] as const;

export function WatchlistEditor({
  symbols,
  onAdd,
  onRemove,
  busy = false,
  resolved,
}: Props) {
  const [input, setInput] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    const sym = input.trim().toUpperCase();
    if (!sym) return;
    if (symbols.includes(sym)) {
      setError(`${sym} is already on your watchlist.`);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await onAdd(sym);
      setInput("");
    } catch (e) {
      setError(
        e && typeof e === "object" && "message" in e
          ? String((e as { message: string }).message)
          : "Couldn't add symbol.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleRemove = async (sym: string): Promise<void> => {
    setError(null);
    try {
      await onRemove(sym);
    } catch (e) {
      setError(
        e && typeof e === "object" && "message" in e
          ? String((e as { message: string }).message)
          : "Couldn't remove symbol.",
      );
    }
  };

  const handleStarterClick = async (sym: string): Promise<void> => {
    setError(null);
    try {
      await onAdd(sym);
    } catch (e) {
      setError(
        e && typeof e === "object" && "message" in e
          ? String((e as { message: string }).message)
          : "Couldn't add symbol.",
      );
    }
  };

  const isEmpty = resolved && symbols.length === 0;
  const disabled = busy || submitting;

  return (
    <Panel
      title="Edit watchlist"
      subtitle={`${symbols.length} symbol${symbols.length === 1 ? "" : "s"}`}
      className="mb-4"
    >
      <form
        onSubmit={submit}
        className="mb-3 flex items-center gap-2"
        data-testid="watchlist-editor-form"
      >
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          placeholder="Add a ticker (e.g. NVDA)"
          maxLength={12}
          disabled={disabled}
          className="h-8 max-w-xs text-xs"
          data-testid="watchlist-editor-input"
          aria-label="Add ticker"
        />
        <Button
          type="submit"
          size="sm"
          disabled={disabled || input.trim().length === 0}
          data-testid="watchlist-editor-add"
        >
          <Plus className="h-3.5 w-3.5" />
          Add
        </Button>
      </form>

      {error && (
        <div
          role="alert"
          data-testid="watchlist-editor-error"
          className="mb-3 rounded-md border border-bear/30 bg-bear/10 px-3 py-1.5 text-xs text-bear"
        >
          {error}
        </div>
      )}

      {symbols.length > 0 ? (
        <ul
          className="flex flex-wrap gap-1.5"
          data-testid="watchlist-editor-chips"
        >
          {symbols.map((sym) => (
            <li key={sym}>
              <SymbolChip
                symbol={sym}
                onRemove={() => handleRemove(sym)}
                disabled={disabled}
              />
            </li>
          ))}
        </ul>
      ) : isEmpty ? (
        <div
          data-testid="watchlist-editor-empty"
          className="rounded-md border border-dashed border-border/60 bg-card/30 px-4 py-3 text-xs text-muted-foreground"
        >
          <div className="mb-2">
            Your watchlist is empty. Pick a starter symbol to get going:
          </div>
          <div className="flex flex-wrap gap-1.5">
            {STARTER_SYMBOLS.map((sym) => (
              <button
                key={sym}
                type="button"
                onClick={() => handleStarterClick(sym)}
                disabled={disabled}
                className="rounded-md border border-border/60 bg-card/40 px-2 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-foreground transition-colors hover:border-primary/60 hover:bg-primary/10 hover:text-primary disabled:cursor-not-allowed disabled:opacity-50"
                data-testid={`watchlist-editor-starter-${sym}`}
              >
                + {sym}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="text-xs text-muted-foreground">Loading…</div>
      )}
    </Panel>
  );
}

function SymbolChip({
  symbol,
  onRemove,
  disabled,
}: {
  symbol: string;
  onRemove: () => void;
  disabled: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border border-border/60 bg-card/60 px-2 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-foreground",
        disabled && "opacity-60",
      )}
      data-testid={`watchlist-editor-chip-${symbol}`}
    >
      {symbol}
      <button
        type="button"
        onClick={onRemove}
        disabled={disabled}
        className="grid h-4 w-4 place-items-center rounded text-muted-foreground transition-colors hover:bg-bear/15 hover:text-bear disabled:cursor-not-allowed"
        aria-label={`Remove ${symbol}`}
        data-testid={`watchlist-editor-remove-${symbol}`}
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  );
}
