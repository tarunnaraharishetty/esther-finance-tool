/**
 * Schema-validation helpers at the API boundary (BUGS.md B-18).
 *
 * Every hook in `web/src/lib/*.ts` used to do
 * ``(await res.json()) as ResearchThesis`` — a bare TypeScript cast
 * with no runtime enforcement. A backend schema drift (renamed field,
 * dropped column, partial deploy) would slip through and crash
 * mid-render with ``Cannot read properties of undefined``.
 *
 * The two helpers here close that gap without inventing a new error UI:
 *
 * * :func:`validateJson` reads JSON off a ``Response`` and runs the
 *   supplied zod schema. On a parse failure it throws a plain
 *   :class:`Error` whose message starts with the surface label
 *   (e.g. ``"research:"``) followed by zod's first issue path — exactly
 *   the shape the existing hooks already render through their
 *   ``error: string | null`` state.
 *
 * * :func:`describeZodIssue` formats one zod issue as a short, single-
 *   line string. Exposed so any caller that wants a custom error UI
 *   can format issues the same way the default validator does.
 *
 * Migration runway
 * ----------------
 * Adopt this in three steps per hook:
 * 1. Declare a ``z.object({...})`` schema mirroring the dataclass on
 *    the backend. Use ``z.infer<typeof Schema>`` as the TypeScript type
 *    so the runtime + compile-time shapes can't diverge.
 * 2. Replace ``(await res.json()) as Foo`` with
 *    ``await validateJson(res, FooSchema, "foo")``.
 * 3. The existing ``catch (e) { setError(e.message) }`` block already
 *    surfaces validation failures — no UI changes needed.
 *
 * See ``research.ts`` and ``userWatchlist.ts`` for the migrated
 * reference implementations.
 */

import { ZodError, type ZodIssue, type ZodTypeAny, type infer as ZInfer } from "zod";

/**
 * Read JSON from ``res`` and validate against ``schema``.
 *
 * Throws an ``Error`` whose message is ``"<where>: <first issue>"`` on
 * either a JSON parse failure or a schema mismatch. The single-line
 * shape keeps the existing error UI legible — hooks already display
 * ``error: string | null`` verbatim in their failure banners.
 *
 * The ``where`` label flows into both the thrown error message AND
 * console diagnostics so an operator triaging "research card failed
 * to render" can grep for the exact surface that drifted.
 */
export async function validateJson<Schema extends ZodTypeAny>(
  res: Response,
  schema: Schema,
  where: string,
): Promise<ZInfer<Schema>> {
  let raw: unknown;
  try {
    raw = await res.json();
  } catch (e) {
    const detail = e instanceof Error ? e.message : String(e);
    throw new Error(`${where}: response was not valid JSON (${detail})`);
  }
  const parsed = schema.safeParse(raw);
  if (parsed.success) {
    return parsed.data;
  }
  // Surface the FIRST issue rather than a multi-line dump — error
  // banners are one line. Operators can console.log(parsed.error) for
  // the full report; we also log it here so dev tools see it.
  const issue = parsed.error.issues[0];
  // eslint-disable-next-line no-console
  console.error(
    `[${where}] schema mismatch — ${parsed.error.issues.length} issue(s):`,
    parsed.error.issues,
  );
  throw new Error(`${where}: ${describeZodIssue(issue)}`);
}

/**
 * Format one zod issue as a short, single-line string.
 *
 * Default zod messages already include the path (e.g. ``"Required at
 * \"metrics.0.value\""``) but the wording varies by issue type. This
 * helper normalises to ``"<path>: <message>"`` so error banners read
 * consistently across surfaces.
 */
export function describeZodIssue(issue: ZodIssue): string {
  const path = issue.path.join(".") || "<root>";
  return `${path}: ${issue.message}`;
}

/**
 * Re-export for hooks that want to catch zod-specific failures with
 * ``e instanceof ZodError``. Most callers want the wrapped Error from
 * :func:`validateJson` instead; this is the escape hatch.
 */
export { ZodError };
