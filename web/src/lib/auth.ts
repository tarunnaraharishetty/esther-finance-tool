import { createContext, useContext } from "react";

/**
 * Placeholder auth surface. The product currently has no real auth —
 * paper-feed-only, single trader. This module reserves the seam so a
 * later phase can drop in OAuth / API-key / session-cookie auth
 * without rewriting every component that wants the current user.
 *
 * When auth lands, swap the default value for a real provider in
 * `main.tsx` and wire `useUser()` through Suspense for the loading
 * state.
 */

export interface User {
  id: string;
  name: string;
  avatarUrl: string | null;
  tier: "free" | "pro" | "enterprise";
}

const ANONYMOUS_USER: User = {
  id: "local",
  name: "Trader",
  avatarUrl: null,
  tier: "pro",
};

export const AuthContext = createContext<User>(ANONYMOUS_USER);

export function useUser(): User {
  return useContext(AuthContext);
}
