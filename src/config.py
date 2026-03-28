import os
import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Network
    wss_url: str

    # LLM (OpenAI-compatible proxy — no real key needed)
    base_url: str
    our_team_name: str = ""  # e.g. "team1" or "team2" — set before match day
    model_researcher: str = "gemini-3.0-flash"
    model_debater: str = "gemini-3.1-pro"
    temperature_researcher: float = 0.3
    temperature_debater: float = 0.75

    # Tavily — comma-separated pool, one chosen at random per search
    tavily_api_keys: str  # raw string; parsed in property below

    # LangSmith
    langchain_tracing_v2: str = "false"
    langchain_project: str = "agent-slam-2026"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""

    # Logging
    log_level: str = "INFO"

    @property
    def tavily_keys_list(self) -> list[str]:
        """Return the list of Tavily API keys parsed from the comma-separated env var."""
        return [k.strip() for k in self.tavily_api_keys.split(",") if k.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
