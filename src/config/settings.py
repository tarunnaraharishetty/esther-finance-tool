"""Application settings loaded from environment / .env via Pydantic.

Single source of truth for configuration. Never read os.environ directly
elsewhere — import :func:`get_settings` instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ---- Risk ----
    max_position_pct: float = Field(0.05, gt=0, le=1)
    max_daily_drawdown_pct: float = Field(0.03, gt=0, le=1)
    default_stop_loss_pct: float = Field(0.02, gt=0, le=1)
    default_take_profit_pct: float = Field(0.04, gt=0, le=1)

    # ---- Sentiment model ----
    sentiment_model: str = "ProsusAI/finbert"
    sentiment_device: SentimentDevice = SentimentDevice.CPU

    # ---- Paths ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    logs_dir: Path = PROJECT_ROOT / "logs"
    config_dir: Path = PROJECT_ROOT / "config"

    @field_validator("alpaca_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

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
