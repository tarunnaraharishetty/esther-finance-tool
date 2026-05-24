"""Application settings loaded from environment / .env via Pydantic.

Single source of truth for configuration. Never read os.environ directly
elsewhere — import :func:`get_settings` instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.data.models import TimeFrame


class AppEnv(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AlpacaDataFeed(StrEnum):
    IEX = "iex"
    SIP = "sip"


class NewsProvider(StrEnum):
    ALPACA = "alpaca"
    NEWSAPI = "newsapi"
    BENZINGA = "benzinga"


class SentimentDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Alpaca ----
    alpaca_api_key: SecretStr
    alpaca_api_secret: SecretStr
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_feed: AlpacaDataFeed = AlpacaDataFeed.IEX

    # ---- News ----
    news_api_key: SecretStr | None = None
    news_provider: NewsProvider = NewsProvider.ALPACA

    # ---- Fundamentals providers ----
    # Per-provider API keys. ``None`` means the provider is unconfigured
    # and the orchestrator silently skips it (logs as ``unavailable``,
    # walks to the next link in the chain). All four are optional — the
    # SEC EDGAR fallback works with just a User-Agent and Yahoo needs
    # nothing, so the chain remains usable when none are set.
    fmp_api_key: SecretStr | None = None
    finnhub_api_key: SecretStr | None = None
    alphavantage_api_key: SecretStr | None = None
    # SEC EDGAR mandates a contact User-Agent on every request (their
    # ToS: "Sample Company Name AdminContact@<sample>.com"). Empty
    # disables the EDGAR provider rather than sending a generic UA.
    sec_edgar_user_agent: str = ""
    # Ordered chain. The orchestrator walks providers in this order and
    # returns the first healthy one. Keep keyed providers ahead of
    # scraped fallbacks; SEC EDGAR sits between paid APIs and Yahoo
    # because it's authoritative but slower and missing ratios.
    fundamentals_provider_order: list[str] = [
        "fmp",
        "finnhub",
        "alpha_vantage",
        "sec_edgar",
        "yahoo",
    ]
    # How long a normalized record is considered fresh in the analyzer
    # cache. 12h is a reasonable default — fundamentals don't move
    # intraday and most providers update within a day of earnings.
    fundamentals_cache_ttl_hours: float = 12.0

    # ---- Retry queue ----
    # Durable JSON-on-disk queue for transient fundamentals fetch
    # failures (rate limits, 5xx). A worker drains it on its own
    # cadence so a 30-second blip doesn't strand a symbol until the
    # next user-initiated retry. Set ``RETRY_QUEUE_PATH=`` (empty) to
    # disable persistence entirely — the route 404s and the service
    # skips enqueue. ``retry_queue_worker_enabled`` is off by default
    # so the local dev TUI and tests don't make background API calls.
    retry_queue_path: Path | None = PROJECT_ROOT / "data" / "retry_queue.json"
    retry_queue_max_attempts: int = 5
    retry_queue_initial_backoff_seconds: float = 30.0
    retry_queue_max_backoff_seconds: float = 1800.0
    retry_queue_worker_interval_seconds: float = 60.0
    retry_queue_worker_enabled: bool = False

    # ---- Calibration ----
    # SignalHistoryCalibrator persists score observations + the
    # materialized hit-rate table to this SQLite file. Set
    # ``CALIBRATION_STORE_PATH=`` (empty) to disable persistence.
    # ``calibration_horizon_days`` is the default forward horizon for
    # outcomes; ``calibration_min_observations`` is the floor below
    # which the UI must not publish a calibrated probability.
    calibration_store_path: Path | None = PROJECT_ROOT / "data" / "calibration.db"
    calibration_horizon_days: int = 5
    calibration_bucket_width: float = 10.0
    calibration_min_observations: int = 30
    # CalibrationMaturationWorker is opt-in; off by default so a local
    # dev session doesn't accumulate background SQLite writes the
    # developer didn't ask for. Production deployments enable.
    calibration_maturation_enabled: bool = False
    calibration_maturation_interval_seconds: float = 3600.0

    # ---- Reconciliation ----
    # When the primary provider in the chain isn't the preferred one
    # (i.e., we fell through to a fallback), make a single follow-up
    # call to a different configured provider and compare high-trust
    # valuation fields (revenue, net_income, eps_diluted, total_debt).
    # Disagreements above ``reconciliation_divergence_threshold`` are
    # surfaced as FieldDivergence rows so the UI can render "providers
    # disagree" and the analyzer can down-weight.
    # Disable in environments where the extra call is too expensive
    # (free-tier API quotas).
    reconciliation_enabled: bool = True
    reconciliation_divergence_threshold: float = 0.05

    # ---- Health observability ----
    # SQLite store of per-provider ProviderHealth rows. Sink writes from
    # FundamentalsService (and later the bars / news paths) land here so
    # the platform can compute rolling SLOs ("FMP success rate last 24h")
    # without re-running every call. Set to ``None`` (env: ``HEALTH_STORE_PATH=``)
    # to disable persistence — useful when running stateless or in tests
    # that don't exercise the /api/health/providers endpoint.
    health_store_path: Path | None = PROJECT_ROOT / "data" / "health.db"

    # ---- Auth / sessions ----
    # User accounts + session tokens live here. Separate from the
    # health/analyzer DBs because user state has different backup +
    # access-control needs than observability. Set to ``None``
    # (env: ``USER_STORE_PATH=``) to disable auth entirely — useful
    # for the single-user dev workflow that pre-dates accounts.
    user_store_path: Path | None = PROJECT_ROOT / "data" / "users.db"
    # Key used to sign session cookies. Dev fallback is a
    # human-readable string — for any deployment you'd flip this in
    # the env (``SESSION_SECRET_KEY=…``). Length isn't enforced;
    # itsdangerous accepts arbitrary bytes.
    session_secret_key: SecretStr = SecretStr(
        "esther-dev-session-key-change-in-production"
    )
    # Session lifetime. 30 days matches the default browser cookie
    # expectation for "stay logged in" — long enough to feel
    # persistent, short enough to bound exposure on a stolen device.
    session_ttl_days: int = 30
    # Cookie name. Stable across deployments; changing this would
    # invalidate every issued session.
    session_cookie_name: str = "esther_session"

    # ---- Analyzer / valuation ----
    # Sector-median multiples (P/E, EV/EBITDA, P/S, PEG) used by the
    # multiple-based valuation models. Shipped seed lives at
    # ``config/sector_medians.toml``; override with the
    # ``SECTOR_MEDIANS_PATH`` env var to point at a custom file (e.g.
    # backfilled from market data).
    sector_medians_path: Path = PROJECT_ROOT / "config" / "sector_medians.toml"
    # DCF defaults. Models can override per-call, but most paths use
    # the same five-year projection + perpetual terminal we use here.
    dcf_default_discount_rate: float = 0.10  # 10% WACC proxy
    dcf_default_terminal_growth: float = 0.025  # 2.5% perpetual growth
    dcf_projection_years: int = 5

    # ---- Database ----
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'esther.db'}"

    # ---- Runtime ----
    app_env: AppEnv = AppEnv.DEV
    log_level: LogLevel = LogLevel.INFO
    log_json: bool = False

    # ---- Sentiment model ----
    sentiment_model: str = "ProsusAI/finbert"
    sentiment_device: SentimentDevice = SentimentDevice.CPU

    # ---- News quality weighting ----
    # Per-article weight = recency_decay * source_reputation. Both
    # factors are bounded [0, 1]; the recency floor keeps stale
    # articles contributing a non-zero share.
    news_recency_half_life_hours: float = 24.0
    news_recency_floor_weight: float = 0.05

    # ---- Session persistence ----
    # JSON snapshot of intelligence-layer trackers (signal history,
    # OPP membership, pulse history, alert state). Written atomically
    # at the end of each tick and read at controller startup so the
    # dashboard resumes mid-day instead of starting cold.
    session_state_path: Path = PROJECT_ROOT / "data" / "session_state.json"

    # ---- Multi-timeframe ----
    # Opt-in secondary intraday read alongside the daily pipeline.
    # When enabled the controller fetches a second bar window per
    # symbol per tick at ``intraday_timeframe`` and surfaces an
    # alignment chip in the dashboard. Default off — doubles Alpaca
    # bar-fetch volume when on.
    intraday_enabled: bool = False
    intraday_timeframe: TimeFrame = TimeFrame.MIN_15
    intraday_lookback_days: int = 5

    # ---- Dashboard cadence ----
    # Base refresh interval + adaptive-burst follow-up. Burst fires
    # one extra tick at ``dashboard_burst_seconds`` when a snapshot
    # produces fresh alerts or any healthy row's action flips.
    dashboard_refresh_seconds: float = 5.0
    dashboard_burst_seconds: float = 1.5

    # ---- Dashboard columns ----
    # Visible columns in the watchlist DataTable, in display order.
    # Override via the ``DASHBOARD_COLUMNS`` env var (JSON list, e.g.
    # ``DASHBOARD_COLUMNS='["SYM","ACTION","CONF","PRICE","NEWS"]'``)
    # when the full 11-column grid is too dense. Field validator
    # rejects unknown names so a typo fails fast at startup.
    dashboard_columns: list[str] = [
        "SYM",
        "ACTION",
        "CONF",
        "BAR",
        "TECH",
        "SENT",
        "RSI",
        "MACD",
        "BBAND",
        "PRICE",
        "NEWS",
    ]

    # ---- AI summaries (Anthropic) ----
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-4-7"
    llm_max_tokens: int = 1024

    # ---- HTTP API ----
    # Origins allowed to hit the API cross-origin. Default is the Vite
    # dev-server (5173) so `npm run dev` works out of the box. In
    # production, when FastAPI serves the bundled React app from
    # ``web/dist/`` on the same origin, browsers never make CORS
    # requests at all — these origins are irrelevant unless you host
    # the frontend separately. Override with ``CORS_ORIGINS=`` (empty
    # = block all cross-origin) or a comma/JSON list.
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ---- Paths ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    logs_dir: Path = PROJECT_ROOT / "logs"
    config_dir: Path = PROJECT_ROOT / "config"

    @field_validator("alpaca_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: object) -> object:
        """Accept ``CORS_ORIGINS=http://a,http://b`` in addition to a JSON list.

        Deploy hosts (Railway / Render) usually pass env vars as plain
        strings; comma-separated is the friendlier form. JSON-list also
        works for parity with the rest of the settings layer.
        """
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return v  # let pydantic parse the JSON list
            return [piece.strip() for piece in stripped.split(",") if piece.strip()]
        return v

    @field_validator("health_store_path", mode="before")
    @classmethod
    def _empty_health_path_is_none(cls, v: object) -> object:
        """Treat ``HEALTH_STORE_PATH=`` (empty string) as disabled.

        Lets deploy hosts opt out of provider-health persistence by
        unsetting the variable (Railway/Render treat unset and empty
        identically). Otherwise pydantic-settings parses empty as
        ``Path("")`` which would silently land DB files in the CWD.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("retry_queue_path", mode="before")
    @classmethod
    def _empty_retry_queue_path_is_none(cls, v: object) -> object:
        """Same opt-out convention as the health store path."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("calibration_store_path", mode="before")
    @classmethod
    def _empty_calibration_path_is_none(cls, v: object) -> object:
        """Same opt-out convention as the health store path."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("fundamentals_provider_order", mode="before")
    @classmethod
    def _split_provider_order(cls, v: object) -> object:
        """Accept ``FUNDAMENTALS_PROVIDER_ORDER=fmp,finnhub`` or JSON list."""
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return v
            return [piece.strip() for piece in stripped.split(",") if piece.strip()]
        return v

    @field_validator("fundamentals_provider_order")
    @classmethod
    def _validate_provider_order(cls, v: list[str]) -> list[str]:
        """Reject unknown provider names so typos fail fast at startup."""
        from src.intelligence.fundamentals.models import ProviderName

        valid_names = {name.value for name in ProviderName}
        unknown = [name for name in v if name not in valid_names]
        if unknown:
            joined = ", ".join(sorted(valid_names))
            raise ValueError(
                f"fundamentals_provider_order contains unknown name(s) {unknown}. "
                f"Valid names: {joined}"
            )
        return v

    @field_validator("dashboard_columns")
    @classmethod
    def _validate_dashboard_columns(cls, v: list[str]) -> list[str]:
        """Reject unknown column names so a typo fails fast at startup.

        Empty list is allowed — the dashboard renders the table with
        no columns (unusual but legal). The canonical name set is
        defined in :mod:`src.dashboard.app`; importing lazily keeps
        this module free of dashboard imports.
        """
        from src.dashboard.app import _COLUMN_NAMES

        unknown = [name for name in v if name not in _COLUMN_NAMES]
        if unknown:
            valid = ", ".join(sorted(_COLUMN_NAMES))
            raise ValueError(
                f"dashboard_columns contains unknown name(s) {unknown}. Valid names: {valid}"
            )
        return v

    @property
    def is_paper_trading(self) -> bool:
        """Sanity-check helper. Live URL must be an explicit, deliberate flip."""
        return "paper" in self.alpaca_base_url.lower()

    @property
    def is_prod(self) -> bool:
        return self.app_env == AppEnv.PROD


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Call once at startup; reuse the singleton."""
    return Settings()  # type: ignore[call-arg]
