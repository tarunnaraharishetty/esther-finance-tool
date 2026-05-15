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
