# TODO.md — Prioritised work, brutally honest

Cross-refs are to `BUGS.md` (B-#) and `ROADMAP.md` phases.

## P0 — Critical fixes (must ship before any non-toy deployment)

* [ ] **Gate `secure_cookies` on env.** Add `secure_cookies: bool`
  to `Settings`; default `True`; pass through in `api/app.py:406`.
  *(B-1)*
* [ ] **Refuse to start with default `session_secret_key` in non-dev.**
  Validator in `Settings.model_post_init`. *(B-2)*
* [ ] **Authenticate `/api/stream` and filter per-user.** Add
  `Depends(require_current_user)`; let the broker accept a per-session
  symbol filter; filter snapshots before fan-out. *(B-3)*
* [ ] **Authenticate every read route** except `/api/health`. *(B-4)*
* [ ] **Per-user rate limiting** on `/api/research`, `/api/analyzer`
  (LLM mode), and `/api/compare/.../narrative`. Add `slowapi`. *(B-5)*
* [ ] **Add login/signup rate limiting** at IP + email scope. *(B-5)*
* [ ] **Pass `timeout=` on every Anthropic call**. Define
  `settings.llm_timeout_seconds` (default 30). *(B-6)*
* [ ] **Set `temperature=0`** on summary, research, and compare
  narrative calls. Document the determinism contract. *(B-7)*
* [ ] **Refuse live Alpaca URL inside `AlpacaClient.__init__`**, not
  just at command entry. *(B-17)*
* [x] **Escape symbol and headline strings** before prompt
  interpolation; cap length. `sanitize_prompt_value` (free-text,
  control-strip + XML-escape + length cap) and `sanitize_symbol`
  (strict ticker regex with UNKNOWN sentinel) cover every LLM call
  site including the previously-unsanitized analyzer prompts.
  *(B-9 — closed)*

## P1 — High leverage, before MVP launch

* [x] **Shared `validate_prose(text, corpus)`** wired into
  `LLMSummarizer.summarize()`, `LLMRecapGenerator.generate()`,
  `LLMOpportunityBriefer.brief()`, and `/api/compare/.../narrative`
  (the compare path uses the full-corpus `validate_narrative`). *(B-8 — closed)*
* [x] **Sanity-bound provider numeric values** in `sanity.py` with
  rejection logging to the accuracy ledger via self-referential
  `sanity:*` events. Trust weight reflects persistent quality drift
  via the same agreement aggregator used by reconciliation. *(B-11 — closed)*
* [x] **Clamp dispersion / agreement in valuation ensemble** to avoid
  NaN propagation. Divisor floors at `max(|base|, median(|values|), 1.0)`;
  raw dispersion is NaN-checked and clamped into `[0, 1]`; confidence
  capped at `[0, 100]`. Regression tests added. *(B-12 — closed)*
* [x] **Honour `Retry-After`** end-to-end: parsed in `http.py`,
  carried by `ProviderRateLimited` + `ProviderChainExhausted` +
  `TransientRetryFailure`, used as the reschedule floor in
  `RetryQueue.record_failure` (max(backoff, server hint)). *(B-10 — closed)*
* [~] **Add `zod` schemas** in `web/src/lib/*.ts`. Foundation
  (`_validate.ts` helper + `ResearchThesisSchema` +
  `WatchlistEntrySchema`) landed; 12 remaining hooks adopt the same
  pattern one-by-one (`analyzer`, `api`, `auth`, `compare`,
  `compareNarrative`, `history`, `movementDrivers`, `providers`,
  `sectorRank`, `spotlight`, `stream`, `utils`). *(B-18 — foundation
  landed; runway documented in BUGS.md)*
* [x] **Eviction policy for `brief_cache` / `opp_brief_cache`**
  (FIFO + re-insert-on-refresh, max 200) in `src/utils/bounded.py`,
  wired through both the controller-owned mirror AND the TUI's
  in-memory cache. Hydrate trims oversized snapshots automatically.
  *(B-16 — closed)*
* [x] **Default non-zero cooldown for every alert rule** in
  `_DEFAULT_COOLDOWNS` (60s..900s); `ConfidenceThresholdRule.band`
  provides hysteresis on threshold rules; regression invariant test
  ensures any new default rule ships with a cooldown. *(B-22 — closed)*
* [x] **Transactional signup-seed**: rollback path chosen.
  `_seed_user_watchlist` purges partial watchlist rows + calls
  `UserStore.delete_user`; auth handler returns 503 instead of
  swallowing the exception. *(B-14 — closed)*
* [x] **Replace static `sector_medians.toml` with computed medians**:
  `SectorCohort` observes every successful fundamentals fetch and
  recomputes per-sector medians; `lookup()` serves the computed table
  with `source` + `as_of` + `cohort_size` metadata. Static TOML is
  the per-metric floor when a sector is below the 5-sample cohort
  threshold. *(B-13 — closed)*
* [ ] **CSRF protection** on watchlist mutations (double-submit cookie
  or per-session token). SameSite=lax is not enough for POST/DELETE on
  paid surfaces. *(B-3 context)*
* [ ] **Add `Dockerfile`** + GH Actions CI (lint, mypy, pytest,
  vitest, build). *(B-27)*
* [ ] **Real DB migrations** via Alembic (Python) and explicit migration
  scripts for SQLite files. *(B-28)*

## P2 — MVP-quality features and UX

* [ ] **Replace templated AI surfaces** (`aiSummary.ts`, sparkline mock)
  with real LLM-streamed text or remove the "AI" labeling. *(B-32)*
* [ ] **URL state** for active symbol + nav (deep-linkable). *(B-29)*
* [ ] **Snapshot stream filtered server-side per user watchlist**;
  drop the client-filter pattern entirely. *(B-3)*
* [ ] **Move all magic-number scoring constants** to one
  `intelligence/calibration.toml` with version + author + last-reviewed
  date. *(B-34)*
* [ ] **Settings knobs** for: signal aggregator threshold + weights,
  reconciliation divergence threshold, confidence position curve,
  trust weight floor/ceiling. *(B-21, B-34)*
* [x] **Single source for ProvenanceProse tokenization**:
  `text_grounding.TOKEN_PATTERN` is the canonical Python source
  imported by the two field-aware validators; frontend literal stays
  in TS but a cross-language sync test asserts behavioural parity on
  a full-coverage fixture. *(B-19 — closed)*
* [ ] **AbortController** on every hook fetch in `web/src/lib/*.ts`.
* [ ] **Add `AlertCooldown` view** in the UI so a trader can see why an
  alert *didn't* fire.
* [ ] **`/healthz` endpoint** distinct from `/api/health` for
  load-balancer probes, plus `/readyz` that includes DB and Alpaca
  reachability.

## P3 — High-leverage scaling improvements

* [x] **Batch FinBERT inference**: `SentimentAnalyzer.score_texts` /
  `score_articles` run one forward pass per `_MAX_BATCH_SIZE=32` chunk.
  Hot caller in `_score_news` wired through. *(B-15 — closed; tokenizer
  pinning + device-buffer pre-allocation deferred to a later perf pass
  if profiling shows they help.)*
* [x] **Replace retry queue JSON file with SQLite** with composite
  indexes on `(status, next_attempt_at)` and `(status, last_error_at)`.
  Public API unchanged so every caller (service, worker, health,
  app) is untouched. Default path moved to `retry_queue.db`. Perf
  regression test bounds `due_entries()` at <500ms under 5000+ row
  load. *(B-23 — closed)*
* [ ] **Per-provider rate-limit policies** (token bucket per key) in
  the fundamentals service instead of one global backoff. *(B-10)*
* [ ] **HTTP caching layer** in front of Anthropic (hash of system +
  user + model → cached completion) for non-streaming calls.
* [ ] **Prompt cache hit-rate logging** (cache_read_input_tokens,
  cache_creation_input_tokens) per call site.
* [ ] **Token budget tracker** per request and per user; cut off at
  configurable daily caps.
* [ ] **Split `dashboard/app.py`** into view + controller + screen
  modules. *(B-31)*

## P4 — Calibration / data-quality moat (the actual differentiator)

* [ ] **Walk-forward backtests** for every threshold in
  `analyzer/technical.py`, `analyzer/valuation.py`, `tier.py`,
  `opportunities.py`. Reject thresholds with no out-of-sample
  improvement vs. naive baseline.
* [ ] **Calibration histogram per signal**: predicted P(BUY pays off) vs
  realized. Surface a calibration plot in the Trust Score panel.
* [ ] **Real ground-truth source for the accuracy ledger.** Today the
  ledger measures self-consistency between providers, which is a weak
  proxy. Pull SEC filings as ground truth for `revenue / net_income /
  eps_diluted / total_debt` and back-fill historical agreement.
* [ ] **Track per-provider per-field bias** (signed deviation, not just
  absolute divergence) and surface it in `ProvidersPage`.
* [ ] **Pull in a second equity universe** (e.g., Polygon snapshots)
  to widen the cross-provider check beyond the existing four.
* [ ] **Sector-relative percentile sanity tests** — if a stock's
  P/E is in the 99th percentile of its sector, surface that fact
  before letting valuation suggest it's "fairly valued."

## P5 — CREATE-X / YC application readiness

* [ ] **Live demo on a public URL** with seeded demo account.
* [ ] **Read-the-docs style docs**: 1-page elevator, 3-page architecture,
  10-min screencast.
* [ ] **Logged-out marketing landing page** that's distinct from the
  app shell.
* [ ] **Pricing page + feature gating hooks** (no charging yet,
  just plumbing).
* [ ] **Per-user activity log** for "you ran N research theses, M alerts
  fired" — feeds both the UX and the YC traction story.
* [ ] **Observability stack** (structlog → JSON → Logtail or
  similar). One trace ID per request, propagated through Anthropic and
  provider calls.
* [ ] **Privacy + terms pages** with explicit "we do not trade on your
  behalf" disclaimer matching the project ethos.

## P6 — Institutional / professional-grade

* [ ] **Multi-tenancy with org/role model.** Users belong to orgs;
  watchlists, briefs, briefs cache are org-scoped.
* [ ] **Audit log** of every research generation, alert delivery, and
  watchlist mutation. Tamper-evident (append-only).
* [ ] **SSO (Google / Microsoft) + SCIM provisioning.**
* [ ] **Run book + on-call alerts** (PagerDuty/Opsgenie). RED
  dashboards for provider chain SLOs.
* [ ] **Cost dashboards** for Anthropic, FMP, Finnhub, Alpha Vantage.
  Budget alerts.
* [ ] **Postgres migration** with full schema versioning and
  zero-downtime deploys (replace SQLite for users.db + health.db).
* [ ] **Background workers** (RQ/Celery/dramatiq) for fundamentals
  refresh, news ingest, retry drain, alert fan-out. The current
  in-process design caps horizontal scaling.
* [ ] **Streaming LLM responses** end-to-end (Anthropic stream → SSE
  → client incremental render) so research feels live, not blocking.
* [ ] **Compliance** posture: SOC 2 control mapping, vendor list, data
  retention policy, customer-managed encryption keys.
* [ ] **Backtesting harness** (walk-forward, point-in-time fundamentals,
  realistic slippage) for any quant claim the product makes.
* [ ] **Per-user data isolation tests** that fuzz cookie-tampering and
  session-replay scenarios.

## Out of scope (and should stay that way)

* Order submission of any kind. Decision-support only — per CLAUDE.md.
* Try-to-beat-SPY backtesting horse race. Esther isn't a strategy.
* "Predictive" language anywhere in copy or LLM output. Grounded
  description only.
