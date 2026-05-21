import { useEffect, useState } from "react";

/**
 * Reveal `text` character-by-character at ~`charsPerSec` chars / sec.
 *
 * Only animates on first mount with a given text. When text changes
 * we restart from zero so the new copy reveals fresh. Returns the
 * currently visible substring + a `done` flag so callers can fade
 * out cursors / etc. when the reveal completes.
 */
export function useTypewriter(text: string, charsPerSec = 80): {
  visible: string;
  done: boolean;
} {
  const [count, setCount] = useState(0);

  useEffect(() => {
    setCount(0);
    if (text.length === 0) return;
    const intervalMs = Math.max(1, Math.round(1000 / charsPerSec));
    const id = window.setInterval(() => {
      setCount((c) => {
        if (c >= text.length) {
          window.clearInterval(id);
          return c;
        }
        return c + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [text, charsPerSec]);

  return { visible: text.slice(0, count), done: count >= text.length };
}
