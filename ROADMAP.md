# ROADMAP.md — From prototype to professional-grade

This is a sequenced plan, not a wish list. Each phase has a *gate*: you
can't ship the next phase until the current one's gate passes. Phase
ordering deliberately puts security and correctness before features —
the current state of `BUGS.md` (B-1..B-6) makes it irresponsible to
broaden the user base before P0 lands.

Where this overlaps the `NEXT_STEPS.md` claim ("P2/P3 actually shipped
in 3dd08ff"), trust the code. Several scoring features called out in
that file exist; calibration, however, does not.

---

## Phase 0 — Honesty pass (1 week)

**Why:** Three load-bearing claims in CLAUDE.md / memory are stronger
than what the code does. Aligning them is cheap and protects every
later phase.

**Work**
* Add a section to README and CLAUDE.md that distinguishes "validated"
  AI surfaces (research thesis, analyzer explanation) from "templated"
  ones (AiPage, AiSummary, sparklines).
* Move the "no-hallucination guarantee" claim to apply only to the
  research route until B-8 lands.
* Move "deterministic" claim out of any LLM-driven surface until B-7
  lands.
* Sweep `aiSummary.ts`, `sparkline.ts`, `PriceChart.tsx`
  (`chartDataFor` fallback) and either replace with real data or label
  in-product as "preview / synthetic."

**Gate:** No public-facing copy claims a guarantee the code doesn't
enforce.

---

## Phase 1 — Security baseline (2–3 weeks)

**Why:** B-1..B-6 are not theoretical. Any deployment that isn't
single-user-on-laptop is leaking sessions, exposing AI cost to the
internet, and brute-forceable.

**Work** (all P0 from `TODO.md`):
* Gate `secure_cookies`, refuse default `session_secret_key`.
* Authenticate `/api/stream` and every read route. Add per-user filter
  on the broker. Drop the client-side filter pattern.
* `slowapi` rate limits on auth + LLM routes.
* `timeout=` and `temperature=0` on every Anthropic call.
* Refuse live Alpaca URL in `AlpacaClient.__init__`.
* Prompt-injection escaping for symbol + headline.

**Gate:** External pentest checklist:
* No unauthenticated route returns customer-specific data.
* No deployment can run with default secrets.
* No LLM endpoint can be hammered for $$ without auth + rate limit.
* No symbol/headline content from external feeds reaches an Anthropic
  prompt without escaping + length cap.

---

## Phase 2 — MVP correctness (3–4 weeks)

**Why:** Today the system is *operational* but uncalibrated. MVP
shouldn't mean "more features"; it should mean "the features that
exist are trustworthy."

**Work**
* `validate_prose()` for summary + compare narrator (close the
  asymmetry).
* Numeric sanity bounds in fundamentals normalization.
* Clamp valuation dispersion; reject pathological inputs.
* Honour `Retry-After`. Per-provider backoff.
* `zod` schemas on every frontend hook, surfaced via error UI.
* Cache eviction on `brief_cache` / `opp_brief_cache`.
* Transactional signup + seed (no "logged in to empty dashboard").
* Compute live sector medians; retire the static TOML seed.
* `Dockerfile` + CI (lint + mypy + pytest + vitest + build).
* Real migrations (Alembic) on every SQLite file.

**Gate:**
* `pytest` + `vitest` green in CI on each PR.
* Schema migrations applied automatically on deploy.
* `/api/research`, `/api/analyzer`, `/api/compare`, `/api/compare/...narrative`
  all reject invalid output (validator drop > 0 surfaces visibly).
* No request stuck > 30 s.
* Deploy to a non-Railway target works.

---

## Phase 3 — CREATE-X / YC submission ready (3–4 weeks)

**Why:** A demo URL plus a defensible "what's different" pitch. At
this stage the moat to lean on is the Provider Accuracy Ledger and the
Trust Score / per-claim provenance — *if* they're meaningful.

**Work**
* Live demo URL, seeded demo account, scripted screencast.
* Logged-out marketing page distinct from the app shell.
* Distinguish the four trust-related moats in the UI: freshness, provider
  ledger, validator drop counts, calibration coverage. Each gets a tooltip
  explaining what it measures and where it came from.
* Strip remaining templated "AI" surfaces or convert to real LLM
  output. Streaming research panel (SSE) for perceived liveness.
* Configurable scoring constants in `calibration.toml`. Settings knobs
  for aggregator + reconciliation thresholds.
* Per-user activity log + simple traction stats.
* Observability: trace IDs across request → LLM call → provider call,
  surfaced in structlog JSON.
* Privacy + terms pages with the "no trading" disclaimer.

**Gate:**
* Cold-open demo: sign up → seeded watchlist → ask for research on a
  specific symbol → result shows provenance + trust grade + dropped
  claims. < 30 s wall clock.
* You can articulate in one paragraph why a sophisticated trader picks
  Esther over Bloomberg + ChatGPT — and the answer is grounded in the
  per-claim provenance + Provider Accuracy Ledger + Trust Score, not
  in scoring magic numbers.
* Cost dashboards exist; the LLM bill per active user is < $5/month at
  typical usage.

---

## Phase 4 — Calibrated intelligence (6–8 weeks)

**Why:** Right now every scoring threshold (`technical.py`,
`valuation.py`, `tier.py`, `opportunities.py`, `trust_score.py`) is a
hand-set constant. Real moat work happens here.

**Work**
* Walk-forward backtests on each threshold. Replace constants that
  don't survive out-of-sample improvement vs naive baseline.
* Calibration histograms per signal (predicted vs realized hit rate).
* Independence/correlation check on the seven opportunity drivers.
  Replace any pair that double-counts.
* Real ground-truth source for the accuracy ledger (e.g., SEC filings as
  benchmark for revenue/EPS/total_debt). Today's ledger measures
  agreement, not truth.
* Per-provider per-field signed bias (not just absolute divergence).
* DCF that computes WACC from beta + risk-free rate, not from settings.
* Pull a second equity universe (Polygon) to widen reconciliation
  cohort.
* Anomaly detection on provider values (z-score against rolling cohort).

**Gate:**
* Trust Score component weights have a documented derivation, not just
  policy.
* Each technical sub-score has a calibration plot the user can inspect.
* The Provider Accuracy Ledger differentiates "more agreement" from
  "more correct" — and surfaces both.

---

## Phase 5 — Scale + multi-tenant (8–12 weeks)

**Why:** SQLite + in-process workers + global broker hits a ceiling at
roughly the "one team / few hundred symbols / dozens of users" mark.
This phase removes that ceiling.

**Work**
* Postgres for users.db + health.db (esther.db can remain SQLite or
  move too). Zero-downtime migration plan.
* Background workers (RQ / dramatiq) for fundamentals refresh, news
  ingest, retry drain, alert delivery. Move the in-process loop out.
* Org/role model. All per-user objects become org-scoped. Watchlists,
  briefs, alerts, history, accuracy ledger filters.
* Audit log (append-only) of every research generation, alert delivery,
  watchlist mutation, login. Tamper-evident.
* Per-user / per-org cost tracking + budgeting.
* Postgres-backed retry queue (replace JSON file).
* Per-provider rate-limit policies enforced as token buckets.
* End-to-end streaming LLM responses to the SSE stream.

**Gate:**
* 1000 concurrent SSE subscribers per node, < 100 ms p95 fan-out.
* Provider chain pulls scale linearly with worker count.
* Cost per active org is observable and capped.

---

## Phase 6 — Institutional / professional-grade (continuous)

**Why:** "Bloomberg-adjacent" buyers want SSO, audit trails, SLAs.

**Work**
* SSO (Google + Microsoft) + SCIM provisioning.
* Customer-managed encryption keys for sensitive at-rest data.
* SOC 2 control mapping; vendor list; data retention policy.
* RED dashboards per provider with paging on SLO breach.
* Public status page.
* Per-feature A/B harness; ship calibration improvements behind flags.
* Backtest harness with point-in-time fundamentals + realistic slippage
  — required before any quant claim ships in copy.
* Per-user data-isolation fuzz tests (cookie tampering, session
  replay, IDOR on every route).

**Gate:**
* SOC 2 Type II in flight.
* Status page green > 99.5 % per quarter.
* Calibration improvements ship behind flags with rollback.

---

## Anti-roadmap — what NOT to build

* **Order submission, anywhere.** Per CLAUDE.md.
* **Strategy "beat SPY" backtests.** Out of scope, distracts from the
  decision-support framing.
* **More LLM surfaces before the validator asymmetry (B-8) is closed.**
  Adding surfaces compounds the hallucination surface.
* **More providers in the chain before bias/ground-truth is in place
  (Phase 4).** Adding providers without ground truth just widens the
  reconciliation noise floor.
* **Mobile app.** The web app already isn't responsive enough for
  mobile; native is premature.
* **Realtime quant signals at sub-minute resolution.** Esther's value
  is intelligence, not latency.

---

## Honest current state (2026-05-24)

* **Strong**: lazy-init discipline, freshness envelopes, provider chain
  + reconciliation skeleton, per-claim provenance on research, SSE
  client robustness, type safety end-to-end, watchlist auth.
* **Weak**: scoring calibration (every threshold is a guess), AI
  guarantees applied unevenly across surfaces, snapshot isolation is
  client-side, magic-number sector medians, no cost controls.
* **Missing**: rate limiting, secure-cookie gating, Anthropic timeout
  / temperature, real DB migrations, Docker, CI, server-side
  per-user snapshot filter, true ground truth for accuracy ledger.

The right next step is **Phase 0 + Phase 1 in parallel** — honesty pass
on the copy while the security baseline lands. Everything else waits.
