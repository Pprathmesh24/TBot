from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskConfig(BaseSettings):
    max_consecutive_losses: int = 3
    position_size_percent: float = 1.0
    stop_loss_percent: float = 1.0
    take_profit_percent: float = 2.0
    daily_loss_cap_percent: float = 3.0
    cooldown_hours: int = 24


class OandaConfig(BaseSettings):
    account_id: str = ""
    api_token: str = ""
    environment: str = "practice"  # "practice" | "live"


class TBotConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    lookback_period: int = 1000
    min_pattern_separation: int = 3
    analysis_batch_size: int = 100
    min_confidence: float = 0.6       # signals below this are dropped in Phase 5
    enable_alerts: bool = True
    save_reports: bool = True
    report_directory: Path = Path("./reports")
    db_path: Path = Path("./data/tbot.sqlite")

    slack_webhook_url: str = ""   # set SLACK_WEBHOOK_URL in .env
    fred_api_key:      str = ""   # set FRED_API_KEY in .env

    risk: RiskConfig = Field(default_factory=RiskConfig)
    oanda: OandaConfig = Field(default_factory=OandaConfig)


# Module-level singleton — import this everywhere: from tbot.config import cfg
cfg = TBotConfig()
