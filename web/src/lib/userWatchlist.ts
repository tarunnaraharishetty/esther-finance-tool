/**
 * Per-user watchlist client.
 *
 * Mirrors the wire shape produced by `src/api/watchlist.py`:
 *   GET    /api/watchlist                → {symbols, entries}
 *   POST   /api/watchlist/symbols        → {entry}                (201)
 *   DELETE /api/watchlist/symbols/{sym}  → {removed, symbol}
 *
 * All requests carry the session cookie via `credentials: "same-origin"`.
 * The hook applies optimistic updates on add/remove so the UI feels
 * instant; on failure we roll back to the server-confirmed list.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";

import { validateJson } from "@/lib/_validate";
import { csrfHeaders } from "@/lib/csrf";

export interface WatchlistEntry {
  symbol: string;
  added_at: string;
  sort_order: number;
}

// Runtime schema — runtime mirror of WatchlistEntry above and the
// wire shape produced by ``src/api/watchlist.py``. B-18: catches a
// renamed / dropped field at parse time instead of mid-render.
const WatchlistEntrySchema = z.object({
  symbol: z.string(),
  added_at: z.string(),
  sort_order: z.number(),
});

const WatchlistListResponseSchema = z.object({
  entries: z.array(WatchlistEntrySchema),
  // The server also emits ``symbols`` but the hook derives that
  // client-side from entries. We don't require it in the schema so a
  // future server-side prune of the redundant field doesn't fail us.
  symbols: z.array(z.string()).optional(),
});

const WatchlistAddResponseSchema = z.object({
  entry: WatchlistEntrySchema,
});

// Compile-time guard: schema and interface must agree on shape.
type _EntryShapeOk = z.infer<typeof WatchlistEntrySchema> extends WatchlistEntry ? true : false;
const _entryShapeCheck: _EntryShapeOk = true;
void _entryShapeCheck;

export interface WatchlistError {
  status: number;
  message: string;
}

export interface UseUserWatchlist {
  symbols: string[];
  entries: WatchlistEntry[];
  loading: boolean;
  /** ``false`` until the first GET settles. UI shouldn't claim
   *  "your watchlist is empty" until this flips true. */
  resolved: boolean;
  error: WatchlistError | null;
  refresh: () => Promise<void>;
  add: (symbol: string) => Promise<void>;
  remove: (symbol: string) => Promise<void>;
  /** Whether ``symbol`` is currently on the watchlist (case-insensitive). */
  contains: (symbol: string) => boolean;
}

/**
 * Pull + mutate the current user's watchlist.
 *
 * Pass ``enabled=false`` when no user is signed in so the hook
 * doesn't fire requests that would 401. The hook is otherwise inert
 * — symbols stay empty, loading stays false, resolved stays false.
 */
export function useUserWatchlist(enabled: boolean): UseUserWatchlist {
  const [entries, setEntries] = useState<WatchlistEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [resolved, setResolved] = useState<boolean>(false);
  const [error, setError] = useState<WatchlistError | null>(null);
  const reqRef = useRef<number>(0);

  const refresh = useCallback(async (): Promise<void> => {
    if (!enabled) return;
    const reqId = ++reqRef.current;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/watchlist", {
        credentials: "same-origin",
      });
      if (!res.ok) throw await parseError(res);
      const body = await validateJson(
        res,
        WatchlistListResponseSchema,
        "watchlist",
      );
      if (reqRef.current === reqId) {
        setEntries(body.entries);
      }
    } catch (e) {
      if (reqRef.current === reqId) {
        setError(coerceError(e));
      }
    } finally {
      if (reqRef.current === reqId) {
        setLoading(false);
        setResolved(true);
      }
    }
  }, [enabled]);

  const add = useCallback(
    async (symbol: string): Promise<void> => {
      if (!enabled) return;
      const normalized = symbol.trim().toUpperCase();
      if (!normalized) return;
      // Optimistic insert. Track the rollback snapshot so a failed
      // POST restores the prior list rather than silently dropping
      // the optimistic entry.
      const before = entries;
      const nextOrder =
        entries.reduce((m, e) => Math.max(m, e.sort_order), -1) + 1;
      const optimistic: WatchlistEntry = {
        symbol: normalized,
        added_at: new Date().toISOString(),
        sort_order: nextOrder,
      };
      if (entries.some((e) => e.symbol === normalized)) {
        return; // already present — no-op, no flash, no request
      }
      setEntries([...entries, optimistic]);

      try {
        const res = await fetch("/api/watchlist/symbols", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json", ...csrfHeaders() },
          body: JSON.stringify({ symbol: normalized }),
        });
        if (!res.ok) {
          const err = await parseError(res);
          // 409 just means the server already had it — reconcile to
          // the server's truth via refresh instead of erroring loudly.
          if (err.status === 409) {
            await refresh();
            return;
          }
          throw err;
        }
        const body = await validateJson(
          res,
          WatchlistAddResponseSchema,
          "watchlist.add",
        );
        // Replace the optimistic entry with the server-confirmed one
        // so added_at + sort_order match the canonical record.
        setEntries((prev) =>
          prev.map((e) => (e.symbol === normalized ? body.entry : e)),
        );
      } catch (e) {
        setEntries(before);
        setError(coerceError(e));
        throw e;
      }
    },
    [enabled, entries, refresh],
  );

  const remove = useCallback(
    async (symbol: string): Promise<void> => {
      if (!enabled) return;
      const normalized = symbol.trim().toUpperCase();
      if (!normalized) return;
      const before = entries;
      // Optimistic removal — the server is idempotent on missing.
      setEntries(entries.filter((e) => e.symbol !== normalized));

      try {
        const res = await fetch(
          `/api/watchlist/symbols/${encodeURIComponent(normalized)}`,
          {
            method: "DELETE",
            credentials: "same-origin",
            headers: csrfHeaders(),
          },
        );
        if (!res.ok) throw await parseError(res);
      } catch (e) {
        setEntries(before);
        setError(coerceError(e));
        throw e;
      }
    },
    [enabled, entries],
  );

  const contains = useCallback(
    (symbol: string): boolean => {
      const normalized = symbol.trim().toUpperCase();
      if (!normalized) return false;
      return entries.some((e) => e.symbol === normalized);
    },
    [entries],
  );

  useEffect(() => {
    if (!enabled) {
      setEntries([]);
      setLoading(false);
      setResolved(false);
      setError(null);
      return;
    }
    void refresh();
  }, [enabled, refresh]);

  return {
    symbols: entries.map((e) => e.symbol),
    entries,
    loading,
    resolved,
    error,
    refresh,
    add,
    remove,
    contains,
  };
}

async function parseError(res: Response): Promise<WatchlistError> {
  try {
    const body = await res.json();
    const message =
      typeof body?.detail === "string" ? body.detail : res.statusText;
    return { status: res.status, message };
  } catch {
    return { status: res.status, message: res.statusText };
  }
}

function coerceError(e: unknown): WatchlistError {
  if (e && typeof e === "object" && "status" in e && "message" in e) {
    return e as WatchlistError;
  }
  return {
    status: 0,
    message: e instanceof Error ? e.message : String(e),
  };
}
