"""
Entry point — starts the WebSocket client.
Usage:  uv run python -m src.main
"""

import asyncio

from src.config import configure_logging, get_settings
from src.wss_client import DebateClient


def main() -> None:
    configure_logging()
    settings = get_settings()

    import logging
    logger = logging.getLogger(__name__)
    logger.info("AgentSlam 2026 — starting agent")
    logger.info("Connecting to: %s", settings.wss_url[:60] + "...")
    logger.info("Models: researcher=%s  debater=%s", settings.model_researcher, settings.model_debater)

    client = DebateClient()
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
