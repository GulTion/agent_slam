"""
Async WebSocket client for the AgentSlam competition platform.

Connection flow:
  1. Connect to WSS_URL (JWT token already embedded in the URL; no handshake needed)
  2. Receive 'welcome' → log and wait
  3. Receive 'match-state' → extract topic/stance/turn/teams
  4. Receive 'debate-message' (opponent) → record in history
  5. Detect our turn → invoke LangGraph → send 'debate-message' payload
  6. Handle pause/resume, finish, and error messages gracefully
  7. Reconnect automatically on transient disconnects (up to MAX_RETRIES)
"""

import asyncio
import json
import logging
import random
import time
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from src.agent.graph import run_debate_turn
from src.config import get_settings

logger = logging.getLogger(__name__)

MAX_RECONNECT_DURATION_SECONDS = 90
RESPONSE_DEADLINE_SECONDS = 110  # server SLA = 120s; we keep a buffer


class DebateState:
    """In-memory holder for all match-related state."""

    def __init__(self) -> None:
        self.topic: str = ""
        self.stance: str = ""          # "PRO" or "CON"
        self.our_team: str = ""        # e.g. "team1" or "team2"
        self.opponent_team: str = ""
        self.current_turn: str = ""    # which team's turn it is
        self.match_status: str = ""    # started | paused | completed
        self.message_history: list[str] = []
        self.finish_time_ms: Optional[int] = None
        self.is_paused: bool = False
        self.is_processing_turn: bool = False

    def time_remaining_seconds(self) -> float:
        if self.finish_time_ms is None:
            return float("inf")
        return max(0.0, (self.finish_time_ms - time.time() * 1000) / 1000)

    def is_our_turn(self) -> bool:
        return (
            self.current_turn == self.our_team
            and self.match_status == "started"
            and not self.is_paused
        )


class DebateClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.state = DebateState()

    # ──────────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Connect and keep reconnecting on transient failures within a 90s cutoff."""
        attempt = 0
        disconnect_time = None

        def on_connect():
            nonlocal attempt, disconnect_time
            if attempt > 0:
                logger.info("Reconnection successful!")
            attempt = 0
            disconnect_time = None

        while True:
            try:
                await self._connect_and_listen(on_connect)
                break  # clean exit (match finished)
            except ConnectionClosedOK:
                logger.info("WebSocket closed cleanly. Match likely finished.")
                break
            except (ConnectionClosedError, Exception) as exc:
                if disconnect_time is None:
                    disconnect_time = time.time()

                attempt += 1
                if isinstance(exc, ConnectionClosedError):
                    logger.warning("Connection lost (attempt %d): %s", attempt, exc)
                else:
                    logger.error("Unexpected error (attempt %d): %s", attempt, exc, exc_info=True)

                elapsed = time.time() - disconnect_time
                if elapsed >= MAX_RECONNECT_DURATION_SECONDS:
                    logger.error("Failed to reconnect within %d seconds. Giving up to avoid disqualification.", MAX_RECONNECT_DURATION_SECONDS)
                    break

                # Exponential backoff: 2^attempt, capped at 10s, with jitter
                backoff = min(10.0, 2 ** attempt) + random.uniform(0, 1)

                # Make sure we don't sleep past the hard cutoff
                time_left = MAX_RECONNECT_DURATION_SECONDS - (time.time() - disconnect_time)
                sleep_time = max(0.1, min(backoff, time_left))

                logger.info("Waiting %.1fs before next reconnect attempt...", sleep_time)
                await asyncio.sleep(sleep_time)

    # ──────────────────────────────────────────────────────────────────────────
    # Core listen loop
    # ──────────────────────────────────────────────────────────────────────────

    async def _connect_and_listen(self, on_connect=None) -> None:
        url = self.settings.wss_url
        logger.info("Connecting to %s ...", url)
        async with websockets.connect(url, ping_interval=20, ping_timeout=30) as ws:
            logger.info("Connected.")
            if on_connect:
                on_connect()
            async for raw_msg in ws:
                await self._handle_message(ws, raw_msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Message dispatch
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_message(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Received non-JSON message: %r", raw)
            return

        msg_type: str = msg.get("type", "")
        sender: str = msg.get("from", "system")
        data: dict = msg.get("data", {})

        logger.debug("← [%s] from=%s  data=%s", msg_type, sender, data)

        match msg_type:
            case "welcome":
                logger.info("Server welcome: %s", data.get("message", ""))

            case "user-joined":
                logger.info("User joined: %s", data.get("message", ""))

            case "match-update":
                # Contains finishTime and a status message; store finish timestamp
                self.state.finish_time_ms = data.get("finishTime")
                logger.info(
                    "Match update — finishTime=%s  msg=%s",
                    self.state.finish_time_ms,
                    data.get("message", ""),
                )

            case "match-state":
                await self._on_match_state(ws, data)

            case "debate-message":
                self._on_debate_message(sender, data.get("message", ""))

            case "previous-message":
                # Re-hydrate history if we reconnected mid-match
                convos = data.get("conversations", [])
                self.state.message_history = [
                    f"[{c['team']}]: {c['message']}" for c in convos
                ]
                logger.info(
                    "Restored %d previous messages.", len(self.state.message_history)
                )

            case "match-paused":
                self.state.is_paused = True
                logger.info(
                    "Match paused. Time remaining: %sms", data.get("timeRemaining")
                )

            case "match-resumed":
                self.state.is_paused = False
                self.state.finish_time_ms = data.get("finishTime")
                logger.info("Match resumed: %s", data.get("message", ""))
                # Check immediately if it's now our turn
                await self._maybe_respond(ws)

            case "match-finish":
                logger.info("Match finished! %s", data.get("message", ""))

            case "info":
                logger.info("Server info: %s", data.get("message", ""))

            case "error":
                logger.warning("Server error: %s", data.get("message", ""))

            case "sandbox-message":
                logger.info("Sandbox echo: %s", data.get("message", ""))

            case _:
                logger.debug("Unhandled message type: %s", msg_type)

    # ──────────────────────────────────────────────────────────────────────────
    # Match-state handler
    # ──────────────────────────────────────────────────────────────────────────

    async def _on_match_state(self, ws, data: dict) -> None:
        """
        Update all state fields from a match-state message then decide if
        we should respond.
        """
        prev_turn = self.state.current_turn

        self.state.topic = data.get("topic", self.state.topic)
        self.state.match_status = data.get("status", self.state.match_status)
        self.state.current_turn = data.get("turn", "")
        self.state.finish_time_ms = data.get("finishTime", self.state.finish_time_ms)

        # Determine our team from pros/cons assignment (set once)
        if not self.state.our_team:
            team1 = data.get("team1", "")
            team2 = data.get("team2", "")
            pros_team = data.get("pros", "")
            cons_team = data.get("cons", "")
            # We need to figure out which team is ours.
            # The server sends our turn when it's our go — use that as signal.
            # For now, detect on first turn assignment.
            # This will be finalised on first turn (see _detect_our_team).
            self._detect_our_team(data)

        logger.info(
            "match-state | topic=%r | turn=%s | status=%s | remaining=%.0fs",
            self.state.topic,
            self.state.current_turn,
            self.state.match_status,
            self.state.time_remaining_seconds(),
        )

        await self._maybe_respond(ws)

    def _detect_our_team(self, data: dict) -> None:
        """
        Determine which slot (team1/team2) belongs to us.

        Priority:
          1. OUR_TEAM_NAME in .env  — most reliable, set before match day.
          2. If not set, defer: mark our team as unknown and only snap it
             in when we recognise both slots from match-state.
             NOTE: without OUR_TEAM_NAME the agent will refuse to respond
             until the user sets the value. Log a clear warning.
        """
        if self.state.our_team:
            return

        configured_name = self.settings.our_team_name.strip().lower()
        if configured_name:
            # Use the explicitly configured name
            self.state.our_team = configured_name
        else:
            logger.warning(
                "OUR_TEAM_NAME is not set in .env! Cannot determine which team we are. "
                "Please set OUR_TEAM_NAME=team1 (or team2) in .env before the match."
            )
            return  # stay silent until configured

        pros = data.get("pros", "").lower()
        cons = data.get("cons", "").lower()
        self.state.stance = "PRO" if self.state.our_team == pros else "CON"
        self.state.opponent_team = cons if self.state.stance == "PRO" else pros
        logger.info(
            "Team locked — our_team=%s | stance=%s | opponent=%s",
            self.state.our_team,
            self.state.stance,
            self.state.opponent_team,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Debate-message handler
    # ──────────────────────────────────────────────────────────────────────────

    def _on_debate_message(self, sender: str, message: str) -> None:
        entry = f"[{sender}]: {message}"
        self.state.message_history.append(entry)
        logger.info("Debate message recorded from %s (%d chars)", sender, len(message))

    # ──────────────────────────────────────────────────────────────────────────
    # Turn response
    # ──────────────────────────────────────────────────────────────────────────

    async def _maybe_respond(self, ws) -> None:
        """Trigger the LangGraph agent and send a response if it's our turn."""
        if not self.state.is_our_turn() or getattr(self.state, "is_processing_turn", False):
            return

        remaining = self.state.time_remaining_seconds()
        if remaining < 10:
            logger.warning("Too little time remaining (%.1fs); skipping turn.", remaining)
            return

        self.state.is_processing_turn = True
        try:
            logger.info("It's our turn. Invoking LangGraph pipeline...")

            # Grab opponent's last message from history
            opponent_message = ""
            for entry in reversed(self.state.message_history):
                if not entry.startswith(f"[{self.state.our_team}]"):
                    opponent_message = entry.split("]: ", 1)[-1] if "]: " in entry else entry
                    break

            try:
                # Run graph in a thread so we don't block the async event loop
                argument = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_debate_turn,
                        self.state.topic,
                        self.state.stance,
                        self.state.our_team,
                        opponent_message,
                        list(self.state.message_history),
                        int(remaining),
                    ),
                    timeout=RESPONSE_DEADLINE_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error("LangGraph timed out after %ds!", RESPONSE_DEADLINE_SECONDS)
                argument = (
                    "The weight of evidence is clear: our position stands on verified data "
                    "and logical consistency, which the opponent has failed to rebut."
                )

            # Append our own response to history
            self.state.message_history.append(f"[{self.state.our_team}]: {argument}")

            payload = json.dumps(
                {"type": "debate-message", "data": {"message": argument}},
                ensure_ascii=False,
            )

            logger.info("→ Sending argument (%d chars)...", len(argument))
            await ws.send(payload)
            logger.debug("Sent: %s", argument[:200])
        finally:
            self.state.is_processing_turn = False
