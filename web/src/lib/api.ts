/**
 * Thin typed HTTP client.
 *
 * The SSE stream (`lib/stream.ts`) is the primary data source — this
 * module exists for one-shot lookups (health probes, future search
 * endpoints) and as the single seam where auth headers will be
 * injected when the backend grows them. Keeping all `fetch()` calls
 * routed through here means the auth migration is a one-file change.
 */

import type { DashboardSnapshot } from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    throw new ApiError(`${path} → ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface HealthResponse {
  status: "ok" | "degraded";
  watchlist_size: number;
}

export const api = {
  health: (): Promise<HealthResponse> => request("/health"),
  snapshot: (): Promise<DashboardSnapshot> => request("/snapshot"),
};
