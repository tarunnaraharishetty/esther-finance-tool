/**
 * Account-management client.
 *
 * Mirrors the wire shape produced by `src/api/auth.py`:
 *   POST   /api/auth/verify/request → {ok, already_verified}
 *   POST   /api/auth/verify/confirm → {ok}
 *   DELETE /api/auth/account        → {ok}
 *
 * Authenticated routes (verify-request, delete-account) carry the
 * CSRF header read from ``document.cookie``. /verify/confirm is
 * public — the token in the URL is the credential.
 */

import { z } from "zod";

import { validateJson } from "@/lib/_validate";
import { csrfHeaders } from "@/lib/csrf";

export interface AccountError {
  status: number;
  message: string;
}

const VerifyRequestResponseSchema = z.object({
  ok: z.boolean(),
  already_verified: z.boolean(),
});

export interface VerifyRequestResponse {
  ok: boolean;
  already_verified: boolean;
}

// Compile-time guard: schema and interface must agree on shape.
type _VerifyShapeOk = z.infer<typeof VerifyRequestResponseSchema> extends VerifyRequestResponse
  ? true
  : false;
const _verifyShapeCheck: _VerifyShapeOk = true;
void _verifyShapeCheck;

const OkResponseSchema = z.object({ ok: z.boolean() });

/** Re-mint + re-send the verification email to the current user. */
export async function requestVerifyEmail(): Promise<VerifyRequestResponse> {
  const res = await fetch("/api/auth/verify/request", {
    method: "POST",
    credentials: "same-origin",
    headers: { ...csrfHeaders(), "Content-Type": "application/json" },
    body: "{}",
  });
  if (!res.ok) throw await parseError(res);
  return validateJson(res, VerifyRequestResponseSchema, "verify/request");
}

/** Redeem a verification token from the emailed link. */
export async function confirmVerifyEmail(token: string): Promise<void> {
  const res = await fetch("/api/auth/verify/confirm", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (!res.ok) throw await parseError(res);
  await validateJson(res, OkResponseSchema, "verify/confirm");
}

/** Rotate the current user's password. Server requires the current
 *  password as re-auth. On success the server rotates the session
 *  cookie too (other devices get logged out, current device stays
 *  via a fresh token). */
export async function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<void> {
  const res = await fetch("/api/auth/password/change", {
    method: "POST",
    credentials: "same-origin",
    headers: { ...csrfHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  if (!res.ok) throw await parseError(res);
  await validateJson(res, OkResponseSchema, "password/change");
}

/** Hard-delete the current account. Requires password re-entry. */
export async function deleteAccount(password: string): Promise<void> {
  const res = await fetch("/api/auth/account", {
    method: "DELETE",
    credentials: "same-origin",
    headers: { ...csrfHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  if (!res.ok) throw await parseError(res);
  await validateJson(res, OkResponseSchema, "account/delete");
}

async function parseError(res: Response): Promise<AccountError> {
  try {
    const body = await res.json();
    const message =
      typeof body?.detail === "string" ? body.detail : res.statusText;
    return { status: res.status, message };
  } catch {
    return { status: res.status, message: res.statusText };
  }
}
