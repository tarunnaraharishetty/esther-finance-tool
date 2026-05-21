import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { AlertOctagon, RefreshCw, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Catches render-time errors anywhere in the React tree and shows a
 * deliberate fallback UI instead of unmounting to a white screen.
 * Has to be a class component — React still doesn't ship a
 * hook-based error boundary.
 */
export class ErrorBoundary extends Component<Props, State> {
  override state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("ErrorBoundary caught:", error, info);
  }

  private readonly handleReload = (): void => {
    window.location.reload();
  };

  private readonly handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  override render(): ReactNode {
    if (!this.state.hasError) return this.props.children;

    return (
      <div
        role="alert"
        className="grid min-h-screen place-items-center bg-gradient-to-br from-background via-background to-card p-6"
      >
        <div className="surface-premium max-w-md w-full p-7 animate-slide-up">
          <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-destructive/15 text-destructive">
            <AlertOctagon className="h-5 w-5" />
          </div>
          <h1 className="text-lg font-semibold tracking-tight">
            Something went wrong
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {this.state.error?.message ?? "Unknown rendering error."}
          </p>
          <div className="mt-5 flex gap-2">
            <Button variant="outline" size="sm" onClick={this.handleRetry}>
              <RotateCw className="h-3.5 w-3.5" /> Retry
            </Button>
            <Button size="sm" onClick={this.handleReload}>
              <RefreshCw className="h-3.5 w-3.5" /> Reload
            </Button>
          </div>
          <p className="mt-5 text-[11px] leading-relaxed text-muted-foreground">
            Open the browser console for a stack. If this persists after
            reload, the API may be down — check{" "}
            <code className="font-mono">/api/health</code>.
          </p>
        </div>
      </div>
    );
  }
}
