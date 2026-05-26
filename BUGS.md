# BUGS.md — Known defects and risks, brutally honest

Each entry: severity, location, what's wrong, why it matters, suggested
fix. "Severity" is engineering blast-radius, not customer-facing polish.

## CRITICAL — security / data integrity / outage risk

### B-1. `secure_cookies=False` is hardcoded in production path
* **Where:** `src/api/app.py:406`
* **What:** `register_auth_routes(..., secure_cookies=False, ...)` is a
  literal `False`, not a settings-driven value. Inline comment admits
  it: *"should flip ESTHER_SECURE_COOKIES=1 once we wire that."*
* **Why it matters:** Every cookie issued by the app omits the `Secure`
  attribute. Any HTTPS-stripping middlebox / open Wi-Fi can lift the
  session token. There is no production gate.
* **Fix:** Add `secure_cookies: bool = True` to `Settings` (default on),
  and pass `settings.secure_cookies` here. Default off only when
  `app_env=="development"`.

### B-2. Default `session_secret_key` is a static, known string
* **Where:** `src/config/settings.py:166–168`
* **What:** Defaults to the literal `"esther-dev-session-key-change-in-production"`.
  Nothing at startup checks that the value has been overridden.
* **Why it matters:** Any deployment that forgets to set the env var
  signs cookies with a key that is in the public repo. Attacker can
  mint sessions for arbitrary users.
* **Fix:** In `Settings.model_post_init`, raise if
  `app_env != "development"` and the key equals the default (or has
  fewer than, say, 32 random bytes of entropy).

### B-3. `/api/stream` is unauthenticated and unfiltered
* **Where:** `src/api/app.py:251` (route), `src/api/broker.py`
* **What:** SSE snapshot stream has no auth dependency and emits the
  full controller universe to every subscriber. Per-user filtering is
  done in the frontend (`watchlist.py:26–28` calls this out
  explicitly as "v1").
* **Why it matters:** In a multi-tenant deployment, any user can open
  `/api/stream` directly and read every other user's signals.
  Client-side filtering is not a security boundary.
* **Fix:** Add `Depends(require_current_user)` to `/api/stream`. Inside
  the broker, filter snapshots per session's watchlist before sending.

### B-4. All read endpoints are public
* **Where:** `src/api/{analyzer,fundamentals,research,compare,sector,
  movement,spotlight,history,health}.py`
* **What:** `require_current_user` is referenced only in `watchlist.py`.
* **Why it matters:** Every paid AI surface is free to anyone who knows
  the URL. Compute/LLM cost is unbounded. The "auth foundation" memory
  is misleading.
* **Fix:** Require auth on every read route except `/api/health` (for
  load balancer probes). Add per-user rate limiting on
  `/api/research`, `/api/analyzer`, `/api/compare/.../narrative` since
  each of those triggers an Anthropic call.

### B-5. No login or signup rate-limiting
* **Where:** `src/api/auth.py` (handlers), `pyproject.toml`
  (no `slowapi` / `limits` dependency)
* **What:** Unlimited POSTs to `/api/auth/login` and `/api/auth/signup`.
* **Why it matters:** Brute-force, credential stuffing, account
  enumeration via volume. bcrypt cost-12 burns CPU under attack.
* **Fix:** Add `slowapi` or hand-rolled token-bucket in middleware:
  e.g. 10 login attempts / IP / 5 min, 3 signup / IP / hour.

### B-6. Anthropic calls have no timeout
* **Where:** `src/intelligence/llm_summary.py:409`,
  `src/intelligence/llm_research.py:364`,
  `src/intelligence/comparison_narrative.py:547`,
  `src/intelligence/analyzer/llm_explanation.py:~88`
* **What:** `client.messages.create(...)` is called with no `timeout=`.
* **Why it matters:** If the Anthropic API stalls, a request handler
  blocks. Compounds with B-4 (anyone can fire these).
* **Fix:** Pass `timeout=settings.llm_timeout_seconds` (default 30).
  Wrap in `httpx.TimeoutException` handler that returns 504.

## HIGH — wrong-but-not-catastrophic

### B-7. LLM calls don't set temperature; "deterministic grounding" is false
* **Where:** Same files as B-6.
* **What:** SDK default temperature is sampling-on. Repeated calls for
  the same input yield different prose.
* **Why it matters:** Validator catches numeric drift, not prose drift.
  Trust score will fluctuate on identical inputs.
* **Fix:** Pass `temperature=0` (or 0.1) on every call. Document that
  research/compare narratives are deterministic up to validator drops.

### B-8. Summarizer and Compare-Narrator return raw LLM text with no validation — **CLOSED**
* **Where:** `src/intelligence/llm_summary.py:439-451` (shared
  `text_grounding.validate_prose`), `src/api/compare.py:238-248`
  (full-corpus `comparison_narrative_validator.validate_narrative`),
  `src/intelligence/recap.py:493-510` (shared `validate_prose`),
  `src/intelligence/opportunity_brief.py:566-583` (shared `validate_prose`).
* **What was wrong:** Only the research thesis ran through
  `validate_thesis()`. Summary, compare narrative, recap, and OPP brief
  shipped whatever the LLM produced.
* **Resolution:** Every Anthropic call site now drops sentences whose
  numeric tokens are not in a built corpus, and logs the drop count.
  Three implementations remain (research, compare-narrative, shared
  `text_grounding`) because each has surface-specific corpus + provenance
  logic; consolidating them is a follow-on tracked as B-19.

### B-9. Prompt-injection via symbol + headline interpolation — **CLOSED**
* **Coverage (now end-to-end on every LLM prompt boundary):**
  * `sanitize_prompt_value(value, max_chars=...)` in
    `src/intelligence/grounding.py` — strips ASCII control chars,
    caps length (default 600), XML-escapes `& < > " '`. Used for
    headlines + free-text fields.
  * `sanitize_symbol(value)` (new — closes the missing leg of B-9's
    fix) — enforces the strict regex `^[A-Z][A-Z0-9.\-]{0,9}$` per
    BUGS.md. Invalid input returns the ``"UNKNOWN"`` sentinel + a
    logged warning so a malformed symbol can't bring down a batch of
    LLM calls. Non-string inputs (``None``, numerics) are rejected
    upfront — without this guard ``str(None) → "None" → "NONE"``
    would accidentally pass the regex.
  * Wired into every LLM prompt builder: `llm_research.py`,
    `llm_summary.py`, `comparison_narrative.py`, `recap.py` (4
    sites), `opportunity_brief.py`, and `analyzer/prompts.py` (the
    last was the unsanitized hole this pass found and plugged).
* **Coverage tests** in `tests/test_grounding.py` pin the contract:
  * Control characters stripped from headlines (defeats
    ``\n Ignore previous instructions`` injection).
  * XML-dangerous chars entity-encoded so a ``</example>``-shaped
    headline can't break the LLM's XML response parser.
  * Length cap bounds the prompt budget against an attacker-supplied
    100 KB headline.
  * `sanitize_symbol` accepts BRK.B (period), BF-B (dash), 1-10 char
    tickers; rejects too-long, lowercase-start, digit-start, special
    chars (`<script>`, `OR 1=1`, newline-injection); handles
    non-string input defensively.
  * `analyzer/prompts.py` SYMBOL line + top_headlines block both
    flow through the sanitizers.
* **Mitigation, not elimination:** an attacker can still write
  "Ignore previous instructions" inside the cap. The post-hoc claim
  validators (research thesis + compare narrative + summary + recap
  + OPP brief, all wired via B-8) drop unsupported claims from the
  output, closing the loop end-to-end.

### B-10. Reconciliation never honours `Retry-After` — **CLOSED**
* **Where (now end-to-end):**
  * `src/intelligence/fundamentals/http.py:29-63` — RFC 7231 parser
    (delta-seconds + HTTP-date), 24h ceiling.
  * `src/intelligence/fundamentals/base.py:54-72` — `ProviderRateLimited`
    carries `retry_after_seconds`.
  * `src/intelligence/fundamentals/service.py:421-454, 600-637` —
    tracks `max_retry_after_seconds` across the chain, forwards on
    initial enqueue.
  * `src/intelligence/fundamentals/service.py:649-676` —
    `ProviderChainExhausted` carries `retry_after_seconds` so the worker
    reschedule sees it on the second attempt too.
  * `src/data/retry_worker.py` — `TransientRetryFailure` carries the
    hint; `_process` forwards into `record_failure`.
  * `src/data/retry_queue.py:225-298` — `record_failure` uses
    `max(backoff_delay, retry_after_delay)` so a 5-minute server
    cooldown never gets undercut by our 30-second default.
  * `src/api/app.py:_retry_fundamentals` — translates
    `ProviderChainExhausted.retry_after_seconds` into
    `TransientRetryFailure(retry_after_seconds=...)`.
* **Resolution:** RFC 6585 honored on both the initial enqueue and
  every subsequent reschedule. Provider keys can no longer be banned
  by us retrying 30s after the upstream asked for 5min.

### B-11. Provider numeric values are never sanity-bounded — **CLOSED**
* **Where (now end-to-end):**
  * `src/intelligence/fundamentals/sanity.py` — single chokepoint;
    bounds on profile (market_cap, EV, shares), income (EPS, shares,
    cost_of_revenue), balance (debt, assets, cash, investments,
    shares), ratios (P/E, forward P/E, payout, price_to_sales,
    price_to_book, ev_to_revenue, ev_to_ebitda magnitude, peg
    magnitude, current/quick ratio, margins, returns, beta, dividend
    yield), analyst targets (high≥low, recommendation_mean ∈ [1,5],
    number_of_analysts ≥ 0).
  * `src/intelligence/fundamentals/service.py:528-548` — calls the
    sanitizer on every successful provider response and turns each
    rejection into a self-referential `AccuracyEvent` with
    `field="sanity:<path>"`, `agreed=False`, `rel_error=1.0`. Sunk
    alongside cross-provider reconciliation events.
  * `provider_trust` aggregates `agreed=False` events into the rolling
    accuracy signal, so a provider that consistently ships bogus
    values pays an empirical trust-weight penalty on top of the
    one-shot field drop.
* **Resolution:** A negative market cap / EPS > $5K / negative total
  debt / inverted analyst targets etc. all become `None` instead of
  poisoning the valuation ensemble, and the provider's trust weight
  reflects the persistent quality drift.

### B-12. Valuation dispersion can NaN — **CLOSED**
* **Where:** `src/intelligence/analyzer/valuation.py:442-456`.
* **Resolution:** Divisor is `max(abs(base), median_magnitude, 1.0)`
  so a near-zero base case doesn't inflate `std/base` into the millions;
  `raw_dispersion` is then NaN/Inf-checked and clamped into `[0, 1]`;
  `agreement = max(0.0, 1.0 - dispersion)` and the final confidence is
  capped at `[0, 100]`. Regression tests in
  `tests/intelligence/analyzer/test_valuation.py` pin the invariant
  on three scenarios (loss-maker, near-zero base with a huge outlier,
  single surviving estimator) — a future refactor that drops the clamp
  fails immediately.

### B-13. Sector medians are static seed data shipped as production — **CLOSED**
* **Where (now end-to-end):**
  * `src/intelligence/analyzer/sector_medians.py:SectorCohort` —
    bounded per-symbol record cohort (FIFO eviction with
    move-to-end-on-observe protection). Every call to
    `observe(record)` re-runs `refresh_computed_medians(snapshot)`.
  * `src/intelligence/analyzer/sector_medians.py:lookup` — consults
    the computed-overrides table first, falls through to the static
    TOML for any sector that doesn't yet meet the cohort floor.
  * `src/intelligence/analyzer/sector_medians.py:compute_medians_from_cohort` —
    per-sector, per-metric median with `_MIN_COHORT_SIZE=5` floor.
    Sectors below the floor on a given metric keep the static seed
    for that metric only (partial coverage > dropping the sector).
  * `src/intelligence/fundamentals/service.py:_sink_cohort` — feeds
    every successful fetch into the cohort. Best-effort: a refresh
    failure must never break a fundamentals fetch.
  * `src/api/app.py` — constructs one `SectorCohort` per app and
    threads it through `register_fundamentals_routes` + the
    retry-worker's late-built service.
* **Freshness envelope:** Each `SectorMultiples` carries `source` +
  `as_of` + `cohort_size` directly (per-sector envelope is cleaner
  than wrapping the whole table in `DataEnvelope`). The analyzer can
  surface "computed (12 symbols, observed 14:32 UTC)" alongside any
  multiples-based valuation.
* **Resolution:** Multiples-based valuations now anchor on the live
  watchlist's sector reads when enough symbols have been fetched;
  the static TOML serves as a per-metric floor when the cohort is
  thin (a new deployment with 3 tech symbols still gets a P/E read,
  it just stays at the seed value until the 5th observation lands).

### B-14. `signup → seed_default()` is best-effort and silently partial — **CLOSED**
* **Resolution:** Rollback path chosen — the seeded watchlist is part
  of the implicit signup contract, so the cleanest restoration of the
  invariant is "signup either succeeds with a populated watchlist or
  leaves no trace and surfaces 503."
* **Where (now end-to-end):**
  * `src/data/user_store.py` — added `UserStore.delete_user(user_id)`;
    cascades sessions via the existing FK.
  * `src/api/app.py:_seed_user_watchlist` — wraps seed + controller
    add_symbol in a try/except. On failure: drops any partially-seeded
    watchlist rows (no FK to users, must purge explicitly), calls
    `user_store.delete_user`, re-raises.
  * `src/api/auth.py` signup handler — no longer swallows hook
    exceptions; converts them to `HTTPException(503)` with a
    retry-friendly message. No session cookie is issued on the
    failure path.
* **Why rollback over warning:** A "201 + warning + empty watchlist"
  leaves users staring at nothing wondering why; rollback + retry
  restores the invariant cleanly. Transient SQLite locks are the
  common failure mode and a retry against the recovered backend
  produces a clean signup (covered by
  `test_signup_rollback_lets_user_retry_with_same_email`).

### B-15. Snapshot pipeline runs FinBERT inference per article, sequentially — **CLOSED**
* **Where (now end-to-end):**
  * `src/sentiment/analyzer.py` — new `SentimentAnalyzer.score_texts`
    + `score_articles` run a single batched forward pass per
    ``_MAX_BATCH_SIZE = 32`` chunk through the transformers pipeline.
    Empty inputs are skipped without consuming a slot. Output list is
    always the same length and order as the input.
  * `src/strategy/recommendation.py:_score_news` — hot caller now uses
    `analyzer.score_articles(ordered)` instead of looping
    `score_article` per article. With 500 symbols × 20 articles the
    savings are dominant — was ~10k forward passes, now ~313 (10000/32).
  * `score_text` / `score_article` retained for the rare single-input
    callers (CLI helpers, tests).
* **Mock subclasses updated** to override `score_texts`/`score_articles`
  alongside their existing `score_text`/`score_article`:
  `_NeutralAnalyzer` + `_RandomSentiment` in `dashboard/controller.py`,
  four `_Neutral` CLI helpers in `main.py`, `FakeSentimentAnalyzer` +
  the keyword-mapping stub in `tests/test_recommendation.py`. The
  overrides loop per-text/per-article so the FinBERT pipeline is never
  instantiated under test/mock.
* **Coverage:** 8 new tests in `tests/test_sentiment.py` pin: empty
  input returns empty, output matches input order, whitespace inputs
  skip the pipeline, batched chunking at the cap boundary (70 inputs
  → 3 calls of 32/32/6), single-text delegates to batched, articles
  batched in one pass, unknown labels decoded as neutral.
* **Resolution:** The dominant per-tick latency at scale is fixed.
  Adding more symbols / articles to a watchlist no longer scales
  sentiment cost linearly per article.

### B-16. `brief_cache` and `opp_brief_cache` grow without bound — **CLOSED**
* **Where (now end-to-end):**
  * `src/utils/bounded.py` — `bounded_set` (FIFO + delete-then-reinsert
    for "hot key" survival) and `bounded_trim` (one-pass trim on
    hydrate). Already had 7 unit tests.
  * `src/dashboard/controller.py:64` — `_MAX_BRIEF_CACHE_ENTRIES = 200`.
    `record_row_brief` / `record_opp_brief` / `record_intraday_opp_brief`
    all call `bounded_set`; `hydrate_from_snapshot` runs `bounded_trim`
    on every brief cache so an oversized snapshot from an older build
    self-heals on first load.
  * `src/dashboard/app.py` — TUI's mirror caches (`_brief_cache`,
    `_opp_brief_cache`, `_intraday_opp_brief_cache`) now also write
    through `bounded_set` with the same cap. Without this, the TUI's
    in-memory dict could still grow unbounded even though the
    persisted snapshot stayed capped.
* **Regression tests** in `tests/test_session_store.py` pin:
  - `record_*_brief` evicts past the cap (FIFO).
  - Re-recording a brief for the same key refreshes its position so a
    hot symbol survives eviction.
  - Hydrate trims a synthetically oversized snapshot file
    (3× cap) down to the cap.
  - End-to-end save → load → fresh-controller round-trip preserves
    the cap with no persistence-layer growth.
* **Resolution:** Long-running TUI sessions are now bounded at both
  the persistence layer and the in-memory mirror, and the FIFO cap
  self-heals on hydrate so an old oversized snapshot file gets
  normalized on first load.

### B-17. Alpaca live URL is allowed (with a warning) instead of refused
* **Where:** `src/data/alpaca_client.py:39`
* **What:** `serve` / `recommend` / `dashboard` commands check for
  paper URL in `main.py`, but the underlying client *would* run against
  live if instantiated directly elsewhere (e.g., a future code path).
* **Why it matters:** Defence-in-depth. The CLAUDE.md safety rule
  explicitly forbids live broker traffic.
* **Fix:** Raise in `AlpacaClient.__init__` (or `Settings` validator)
  if `alpaca_base_url` is not the paper host.

## MEDIUM — correctness / scaling

### B-18. Frontend casts every API response without runtime validation — **foundation landed; runway documented**
* **Where (now end-to-end on the migrated hooks):**
  * `web/src/lib/_validate.ts` — new `validateJson(res, schema, where)`
    helper. Reads JSON off a Response, runs the supplied zod schema,
    throws an ``Error`` whose single-line message is
    ``"<where>: <field.path>: <reason>"`` — the existing
    ``error: string | null`` UI renders it verbatim.
  * `web/src/lib/research.ts` — ResearchThesisSchema mirrors every
    nested interface (sections, bull/bear, catalysts, outlook,
    metrics, validation, trust_score). Compile-time assertion
    enforces the schema's inferred type matches the hand-written
    interface so drift between the two fails ``tsc``.
  * `web/src/lib/userWatchlist.ts` — WatchlistEntrySchema +
    list/add response schemas. Same compile-time guard.
  * `web/package.json` — added `zod ^3.23.8`.
* **Coverage:** 9 new vitest cases in `web/src/lib/_validate.test.ts`
  — helper accepts good payloads, throws labelled error on schema
  mismatch, names the offending field path, surfaces non-JSON parse
  failures cleanly, the real ResearchThesisSchema rejects a missing
  ``validation.drop_count`` (the exact partial-deploy class B-18
  warns about) AND an unknown rating enum value.
* **Migration runway** for the remaining 12 hooks (`analyzer.ts`,
  `api.ts`, `auth.ts`, `compare.ts`, `compareNarrative.ts`,
  `history.ts`, `movementDrivers.ts`, `providers.ts`, `sectorRank.ts`,
  `spotlight.ts`, `stream.ts`, `utils.ts`): adopt is mechanical — for
  each hook, write a `z.object({...})` mirroring the interface,
  swap the `as Foo` cast for `await validateJson(res, FooSchema, "...")`.
  The existing `catch (e) { setError(e.message) }` block already
  surfaces failures. The pattern is documented in `_validate.ts`
  module docstring + reference impls in research.ts / userWatchlist.ts.
* **Why "foundation + 2 hooks" instead of all 14 at once:** doing
  the same mechanical refactor 14 times in one PR is the kind of
  bulk change that's easy to land partially and hard to review.
  Two representative hooks (the biggest contract + the most-mutated
  surface) prove the pattern; subsequent hooks can adopt one PR each
  with low risk.

### B-19. `ProvenanceProse` TOKEN_RE is duplicated client + server — **CLOSED**
* **Resolution:**
  * **Python:** the three private copies in `research_validator.py`,
    `comparison_narrative_validator.py`, and `text_grounding.py` are
    consolidated into a single canonical `TOKEN_PATTERN` + `SENTENCE_SPLIT`
    in `src/intelligence/text_grounding.py`. The two field-aware
    validators now `import` them; underscore-prefixed aliases (
    `_TOKEN_PATTERN = TOKEN_PATTERN`) keep existing test imports
    working without churn. An invariant test in
    `tests/test_token_re_cross_language_sync.py` asserts all three
    module-level names point at the same compiled object so a future
    contributor copy-pasting a private copy fails CI loudly.
  * **Frontend:** `web/src/components/prose/ProvenanceProse.tsx` still
    carries the JS literal (TS has no verbose-regex mode and we have
    no codegen pipeline). Sync is enforced behaviourally: the new
    cross-language test reads the literal, compiles it via Python
    `re`, and asserts identical match lists against `TOKEN_PATTERN`
    across a 13-input full-coverage fixture (currency, percent,
    multiple, quarter, year, decimal, large integer, mixed, empty).
    Drift between the two regexes — different alternation order,
    missing branch, extra branch — fails CI loudly.
* **Why not codegen:** a build step would add operator burden every
  developer + every CI pipeline has to remember to run. A behavioural
  sync test runs every pytest invocation with no external
  prerequisites. When a new token shape is added, the fixture has to
  grow in lockstep — which is fine, that's the contract.

### B-20. Alpha Vantage quota detection is a substring match
* **Where:** `src/intelligence/fundamentals/alphavantage.py:150`
* **What:** Looks for "rate limit" in `Information` field.
* **Why it matters:** AV has changed this message at least twice in
  the past. A change flips the failure into a `ProviderTransient`
  with garbage JSON.
* **Fix:** Move to multi-pattern (or any non-empty `Information` field
  on otherwise-empty payload).

### B-21. SignalAggregator threshold and weights are hardcoded
* **Where:** `src/strategy/signal_aggregator.py:20`
* **What:** Threshold 0.2; weights default 1.0 if missing from dict.
* **Why it matters:** No A/B path. Tuning needs a code edit + redeploy.
* **Fix:** Move to `Settings`. Allow per-tag weight override.

### B-22. Alerts can fire every tick — **CLOSED**
* **Where (now end-to-end):**
  * `src/intelligence/alert_prioritizer.py:_DEFAULT_COOLDOWNS` — every
    rule cited in BUGS.md ships with a non-zero default cooldown
    (action_changed=60s, confidence_threshold=300s, sentiment_shift=600s,
    tier_changed=300s) plus opportunity_entry=600s and intraday rules.
  * `src/intelligence/alerts.py:ConfidenceThresholdRule` — `band`
    parameter (default 0.02) gates both up-crossing and down-crossing
    so a value oscillating ±0.005 around 0.60 doesn't fire repeatedly.
    Tested by `tests/test_alert_hysteresis.py`.
  * `src/intelligence/alert_prioritizer.py:_apply_cooldown` — honors
    the per-rule cooldown by comparing against `AlertState.last_fired`.
* **Resolution:** Default cooldowns + ConfidenceThresholdRule hysteresis
  cover the jittery-signal case. Behavioural tests in
  `tests/test_alert_prioritizer.py` pin tier-oscillation debouncing
  (no alert per oscillation inside the 300s window) and a regression
  invariant that asserts every default rule has a non-zero cooldown —
  adding a new noisy rule without one trips CI rather than spamming
  users.
* **Why tier_changed / action_changed don't carry hysteresis:** Both
  rules transition between coarse-grained buckets (3 actions, 5
  tiers); the cooldown alone is enough to suppress promotion-gate
  oscillation. Adding a hysteresis band would mute legitimate
  transitions.

### B-23. Retry queue read/write is O(n) per op — **CLOSED**
* **Where (now end-to-end):**
  * `src/data/retry_queue.py` — full rewrite to SQLite + WAL +
    `Migration` framework. Same public surface (``enqueue`` /
    ``due_entries`` / ``record_success`` / ``record_failure`` /
    ``all_entries`` / ``remove`` / ``cleanup`` / ``path``) so every
    caller (``service.py``, ``retry_worker.py``, ``health.py``,
    ``app.py``) is unchanged. Same lazy-init contract as every other
    persistent store (``HealthStore`` / ``UserStore`` /
    ``AccuracyStore``).
  * Schema: one row per symbol with composite indexes on
    ``(status, next_attempt_at)`` and ``(status, last_error_at)`` so
    the ``due_entries`` and ``cleanup`` hot paths are
    proportional to the matched subset, not the total table size.
  * `src/config/settings.py` — default ``retry_queue_path`` moved from
    ``data/retry_queue.json`` → ``data/retry_queue.db``. Deployments
    that pinned the old path via env var will see ``sqlite3`` raise
    "file is not a database" at startup; renaming or removing the
    legacy file restores normal operation.
* **Performance regression test** in
  ``tests/data/test_retry_queue.py``: 5000 permanently-failed rows +
  5 due pending — ``due_entries()`` must return in <500ms. A regression
  to full-scan semantics would scale linearly into the seconds on a
  busy machine, failing the bound loudly.
* **Resolution:** ``enqueue`` is one ``INSERT OR REPLACE`` (O(log n)
  on the symbol primary key); ``due_entries`` is an indexed range
  scan; ``record_success`` / ``remove`` are O(1) targeted deletes;
  ``cleanup`` is index-driven on ``(status, last_error_at)``.
  Thousands of permanently-failed rows no longer slow the live path.

### B-24. SSE broker mutation during iteration
* **Where:** `src/api/broker.py:~165`
* **What:** `_subscribers: set[asyncio.Queue]` is iterated via
  `list()` snapshot, but subscribe/unsubscribe races aren't covered by
  a lock.
* **Why it matters:** No deadlock, but a subscriber added mid-tick may
  miss the first message and look reconnect-broken.
* **Fix:** `asyncio.Lock` around mutations of `_subscribers`.

## LOW — polish / debt

### B-25. SQLite engine init is non-atomic
* `src/data/storage.py:33–46` — `if _engine is None` followed by
  assignment. Harmless under GIL, but flagged for completeness.

### B-26. CLI default `--mock` flag in `Procfile` ships mock pipeline to prod
* `Procfile` — `esther serve --mock --host 0.0.0.0 --port $PORT`.
* Intentional, but verify before flipping the brand to "live data."

### B-27. No `Dockerfile`; deployment is Railway-only
* Only `Procfile`/`nixpacks.toml`/`railway.toml`. Non-Railway hosts
  need extra work.

### B-28. No DB migrations
* Schemas are inline `CREATE TABLE IF NOT EXISTS` strings. Any column
  add needs manual SQL on existing databases.

### B-29. Routing has no URL state
* `web/src/App.tsx` — refresh drops active symbol + nav state.
  Shareable URLs aren't possible.

### B-30. Logout doesn't verify server response
* `web/src/lib/auth.ts` — always clears local state. Idempotent;
  user sees logged-out even if server failed to delete the session.

### B-31. `dashboard/app.py` is ~2.8 k lines
* God-module for the Textual TUI. Tested but hard to navigate.

### B-32. `AiPage`, `AiSummary`, sparklines are templated, branded as AI
* `web/src/lib/aiSummary.ts`, `web/src/lib/sparkline.ts`,
  `web/src/components/PriceChart.tsx:97`. Documented as Phase 4
  placeholders but currently mislead users about what's running.

### B-33. `news_ingestion.py:100` raises NotImplementedError
* Only Alpaca is wired; any other configured provider crashes.
  Defensive, but the call site doesn't guard against a misconfig.

### B-34. Hard-coded calibration constants throughout
* Tier thresholds, opportunity driver weights, trust score weights,
  reconciliation 5 %, confidence position curve `1.0/0.85/0.70/0.55/0.40`,
  trust weight floor/ceiling `0.5/1.2`. None are configurable; none
  have backtest provenance.
