/**
 * Auth client + session hook.
 *
 * Mirrors the wire shape produced by `src/api/auth.py`:
 *   GET  /api/auth/me     → {user: AuthUser | null}
 *   POST /api/auth/signup → {user: AuthUser}  (sets session cookie)
 *   POST /api/auth/login  → {user: AuthUser}  (sets session cookie)
 *   POST /api/auth/logout → {ok: true}        (clears session cookie)
 *
 * Session cookies are HttpOnly + SameSite=lax, signed server-side.
 * The browser ships them with `credentials: "same-origin"`; this
 * module always passes that flag so the cookie round-trips.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

export interface AuthUser {
  id: number;
  email: string;
  created_at: string; // ISO-8601
  last_login_at: string | null;
  email_verified: boolean;
}

/** Display-friendly handle derived from email (the part before @). */
export function displayName(user: AuthUser | null): string {
  if (user === null) return "Guest";
  const at = user.email.indexOf("@");
  return at > 0 ? user.email.slice(0, at) : user.email;
}

export interface AuthError {
  status: number;
  message: string;
}

interface UseSession {
  user: AuthUser | null;
  loading: boolean;
  /** ``null`` until the first /me call settles; then either a user
   *  or null thereafter (anonymous is a valid steady state). */
  resolved: boolean;
  error: AuthError | null;
  /** Re-fetch /api/auth/me — useful after a flow we know flipped state. */
  refresh: () => Promise<void>;
  login: (email: string, password: string) => Promise<AuthUser>;
  signup: (email: string, password: string) => Promise<AuthUser>;
  logout: () => Promise<void>;
}

/**
 * Auth session hook. Fetches /api/auth/me on mount; provides
 * imperative login/signup/logout helpers that update local state
 * on success.
 *
 * Designed to be used once at the App root (the result is then
 * pushed through React context for child components to read via
 * :func:`useAuthSession` — but most child components only need
 * :func:`useUser` for the placeholder display.)
 */
export function useSession(): UseSession {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [resolved, setResolved] = useState<boolean>(false);
  const [error, setError] = useState<AuthError | null>(null);
  const reqRef = useRef<number>(0);

  const refresh = useCallback(async (): Promise<void> => {
    const reqId = ++reqRef.current;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/me", {
        credentials: "same-origin",
      });
      if (!res.ok) {
        // /me is never supposed to 4xx for the anonymous case — it
        // returns user=null. If we land here something's wrong with
        // the deployment (e.g. /api/auth not registered). Surface
        // as an error rather than silently treating as logged-out.
        throw { status: res.status, message: res.statusText } as AuthError;
      }
      const body = (await res.json()) as { user: AuthUser | null };
      if (reqRef.current === reqId) {
        setUser(body.user);
      }
    } catch (e) {
      if (reqRef.current === reqId) {
        if (e && typeof e === "object" && "status" in e && "message" in e) {
          setError(e as AuthError);
        } else {
          setError({
            status: 0,
            message: e instanceof Error ? e.message : String(e),
          });
        }
      }
    } finally {
      if (reqRef.current === reqId) {
        setLoading(false);
        setResolved(true);
      }
    }
  }, []);

  const login = useCallback(
    async (email: string, password: string): Promise<AuthUser> => {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) throw await parseError(res);
      const body = (await res.json()) as { user: AuthUser };
      setUser(body.user);
      setResolved(true);
      return body.user;
    },
    [],
  );

  const signup = useCallback(
    async (email: string, password: string): Promise<AuthUser> => {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) throw await parseError(res);
      const body = (await res.json()) as { user: AuthUser };
      setUser(body.user);
      setResolved(true);
      return body.user;
    },
    [],
  );

  const logout = useCallback(async (): Promise<void> => {
    await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "same-origin",
    });
    setUser(null);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return {
    user,
    loading,
    resolved,
    error,
    refresh,
    login,
    signup,
    logout,
  };
}

async function parseError(res: Response): Promise<AuthError> {
  try {
    const body = await res.json();
    const message =
      typeof body?.detail === "string" ? body.detail : res.statusText;
    return { status: res.status, message };
  } catch {
    return { status: res.status, message: res.statusText };
  }
}

// ---------------------------------------------------------------------------
// Compat layer for the placeholder useUser() hook
// ---------------------------------------------------------------------------

/**
 * Light wrapper preserved from the pre-auth scaffolding. Reads from
 * the AuthContext (populated by App) and returns a display-friendly
 * record. Existing components keep using this without caring about
 * the underlying session machinery.
 */
export interface User {
  id: string;
  name: string;
  avatarUrl: string | null;
  tier: "free" | "pro" | "enterprise";
}

const ANONYMOUS_USER: User = {
  id: "guest",
  name: "Guest",
  avatarUrl: null,
  tier: "free",
};

export const AuthContext = createContext<User>(ANONYMOUS_USER);

/** Display-shaped user for layout components. Real session state
 *  lives in :func:`useSession` — this is just for the avatar pill. */
export function useUser(): User {
  return useContext(AuthContext);
}

/** Convert a real :type:`AuthUser` into the display :type:`User`
 *  the existing layout components consume. */
export function asDisplayUser(authUser: AuthUser | null): User {
  if (authUser === null) return ANONYMOUS_USER;
  return {
    id: String(authUser.id),
    name: displayName(authUser),
    avatarUrl: null,
    tier: "pro",
  };
}
