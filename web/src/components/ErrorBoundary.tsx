import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Catches render-time errors anywhere in the React tree and shows a
 * deliberate fallback UI instead of the default React unmount-the-world
 * white-screen behavior.
 *
 * The schema-drift bug class (server adds a field the frontend can't
 * type) is exactly what this guards against — we tightened the types
 * to make those errors compile-time, but defense in depth here covers
 * the leftover surface (transient broker errors, malformed payloads,
 * etc.). Has to be a class component because React still doesn't
 * support hook-based error boundaries.
 */
export class ErrorBoundary extends Component<Props, State> {
  override state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    // Log to console so the dev tools surface a stack and so a future
    // observability hook (e.g. Sentry) has a clear seam to wire into.
    console.error("ErrorBoundary caught:", error, info);
  }

  private readonly handleReload = (): void => {
    window.location.reload();
  };

  private readonly handleRetry = (): void => {
    // Re-render attempt: clear the error and let React try again with
    // a fresh tree. Useful when the underlying cause was transient
    // (e.g. one malformed SSE payload during a deploy).
    this.setState({ hasError: false, error: null });
  };

  override render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="error-boundary" role="alert">
        <div className="error-boundary__inner">
          <h1 className="error-boundary__title">Something went wrong</h1>
          <p className="error-boundary__detail">
            {this.state.error?.message ?? "Unknown rendering error."}
          </p>
          <div className="error-boundary__actions">
            <button type="button" onClick={this.handleRetry}>
              Retry
            </button>
            <button
              type="button"
              onClick={this.handleReload}
              className="error-boundary__primary"
            >
              Reload
            </button>
          </div>
          <p className="error-boundary__hint">
            Check the browser console for a full stack. If this keeps happening
            after a reload, the API may be down — open <code>/api/health</code>
            in a new tab.
          </p>
        </div>
      </div>
    );
  }
}
