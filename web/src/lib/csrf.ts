/**
 * CSRF double-submit cookie helper.
 *
 * The server mints a non-HttpOnly ``esther_csrf`` cookie on login /
 * signup. The frontend reads it via ``document.cookie`` and echoes
 * the value back in an ``X-CSRF-Token`` header on every state-
 * changing request. The server's CSRF middleware rejects requests
 * whose cookie + header don't match.
 *
 * A cross-origin attacker can NOT read the cookie via
 * ``document.cookie`` (same-origin policy), so they can't produce a
 * matching header — that's the protection.
 *
 * Keep this module dependency-free and lightweight: every state-
 * changing client (watchlist mutations, future settings POSTs, …)
 * imports it.
 */
const CSRF_COOKIE_NAME = "esther_csrf";
const CSRF_HEADER_NAME = "X-CSRF-Token";

/**
 * Read the current CSRF token from the cookie jar, or ``null`` if
 * no session is active. Returning ``null`` rather than throwing
 * lets callers add the header opportunistically without branching
 * — the request just goes out token-less and the server's
 * middleware returns 403, which the existing error UI surfaces.
 */
export function readCsrfToken(): string | null {
  // SSR / no-document environments (tests) — bail cleanly.
  if (typeof document === "undefined" || !document.cookie) return null;
  for (const piece of document.cookie.split(";")) {
    const [rawKey, ...rest] = piece.split("=");
    if (rawKey.trim() === CSRF_COOKIE_NAME) {
      return decodeURIComponent(rest.join("=").trim());
    }
  }
  return null;
}

/**
 * Build a headers object containing the CSRF header when a token is
 * available. Spread into a ``fetch`` ``headers`` block — when no
 * token is found (logged-out callers, SSR) the spread is empty.
 */
export function csrfHeaders(): Record<string, string> {
  const token = readCsrfToken();
  return token === null ? {} : { [CSRF_HEADER_NAME]: token };
}
