/**
 * Password-reset client.
 *
 * Mirrors the wire shape produced by `src/api/auth.py`:
 *   POST /api/auth/password-reset/request → {ok, message}
 *   POST /api/auth/password-reset/confirm → {ok}
 *
 * Both routes are public (no session cookie needed) and the
 * request route is exempt from CSRF because it lives under
 * `/api/auth/*` which the gate doesn't enforce.
 *
 * Errors:
 * - Request route always returns 200 (no email enumeration).
 * - Confirm returns 400 for invalid / expired / replayed tokens.
 * - Both can return 429 from the per-IP rate limiter.
 */

import { z } from "zod";

import { validateJson } from "@/lib/_validate";

export interface PasswordResetError {
  status: number;
  message: string;
}

// Server returns {ok, message} on the request route; the frontend
// keys off `message` to render the uniform "if registered" prompt.
const RequestResponseSchema = z.object({
  ok: z.boolean(),
  message: z.string(),
});

export interface RequestResetResponse {
  ok: boolean;
  message: string;
}

// Compile-time guard: schema and interface must agree on shape.
type _RequestShapeOk = z.infer<typeof RequestResponseSchema> extends RequestResetResponse
  ? true
  : false;
const _requestShapeCheck: _RequestShapeOk = true;
void _requestShapeCheck;

const ConfirmResponseSchema = z.object({
  ok: z.boolean(),
});

/**
 * Kick off a password reset for ``email``. The server returns the
 * same 200 body whether or not the email is registered — UI shows
 * "if that email is registered, a reset link is on its way" on the
 * happy path, no different rendering for unknown emails.
 */
export async function requestPasswordReset(
  email: string,
): Promise<RequestResetResponse> {
  const res = await fetch("/api/auth/password-reset/request", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) throw await parseError(res);
  return validateJson(res, RequestResponseSchema, "password-reset/request");
}

/**
 * Confirm a reset using the token from the emailed link. On success
 * the caller should route to the login screen with a flash message
 * (the server already revoked every session for the user).
 */
export async function confirmPasswordReset(
  token: string,
  newPassword: string,
): Promise<void> {
  const res = await fetch("/api/auth/password-reset/confirm", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, new_password: newPassword }),
  });
  if (!res.ok) throw await parseError(res);
  await validateJson(res, ConfirmResponseSchema, "password-reset/confirm");
}

async function parseError(res: Response): Promise<PasswordResetError> {
  try {
    const body = await res.json();
    const message =
      typeof body?.detail === "string" ? body.detail : res.statusText;
    return { status: res.status, message };
  } catch {
    return { status: res.status, message: res.statusText };
  }
}
