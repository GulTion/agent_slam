"""
Tests for the WebSocket client (wss_client.py).

Uses an in-process mock WebSocket server to simulate competition server messages.
Run: uv run pytest tests/test_wss_client.py -v
"""

import asyncio
import json
import re
import pytest
import pytest_asyncio
import websockets
from unittest.mock import AsyncMock, MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_msg(**kwargs) -> str:
    return json.dumps({"from": "system", **kwargs})


WELCOME_MSG = _make_msg(type="welcome", data={"message": "Welcome Tester to AgentSlam!"})

MATCH_STATE_OUR_TURN = _make_msg(
    type="match-state",
    data={
        "team1": "team1",
        "team2": "team2",
        "topic": "Renewable energy can fully replace fossil fuels by 2040",
        "pros": "team1",
        "cons": "team2",
        "turn": "team1",
        "status": "started",
        "finishTime": 9_999_999_999_000,  # far future
        "round": "Round 1",
    },
)

MATCH_STATE_OPPONENT_TURN = _make_msg(
    type="match-state",
    data={
        "team1": "team1",
        "team2": "team2",
        "topic": "Renewable energy can fully replace fossil fuels by 2040",
        "pros": "team1",
        "cons": "team2",
        "turn": "team2",
        "status": "started",
        "finishTime": 9_999_999_999_000,
        "round": "Round 1",
    },
)

OPPONENT_DEBATE_MSG = _make_msg(
    type="debate-message",
    **{"from": "team2"},
    data={"message": "Fossil fuels remain essential due to grid reliability needs."},
)

MATCH_FINISH_MSG = _make_msg(type="match-finish", data={"message": "The match has ended!"})


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — DebateState
# ──────────────────────────────────────────────────────────────────────────────

class TestDebateState:

    def test_is_our_turn_true(self):
        from src.wss_client import DebateState
        s = DebateState()
        s.our_team = "team1"
        s.current_turn = "team1"
        s.match_status = "started"
        s.is_paused = False
        assert s.is_our_turn() is True

    def test_is_our_turn_false_opponent(self):
        from src.wss_client import DebateState
        s = DebateState()
        s.our_team = "team1"
        s.current_turn = "team2"
        s.match_status = "started"
        s.is_paused = False
        assert s.is_our_turn() is False

    def test_is_our_turn_false_paused(self):
        from src.wss_client import DebateState
        s = DebateState()
        s.our_team = "team1"
        s.current_turn = "team1"
        s.match_status = "started"
        s.is_paused = True
        assert s.is_our_turn() is False

    def test_is_our_turn_false_not_started(self):
        from src.wss_client import DebateState
        s = DebateState()
        s.our_team = "team1"
        s.current_turn = "team1"
        s.match_status = "paused"
        s.is_paused = False
        assert s.is_our_turn() is False


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — payload validation
# ──────────────────────────────────────────────────────────────────────────────

class TestPayloadValidation:

    def test_outgoing_payload_valid_json(self):
        """Ensure the payload we construct is valid JSON with correct structure."""
        argument = "Renewable energy is rapidly scaling (source: https://iea.org/2024). Cost of solar dropped 90% in a decade (source: https://irena.org/solar)."
        payload = json.dumps(
            {"type": "debate-message", "data": {"message": argument}},
            ensure_ascii=False,
        )
        parsed = json.loads(payload)
        assert parsed["type"] == "debate-message"
        assert parsed["data"]["message"] == argument

    def test_outgoing_payload_under_3000_chars(self):
        """Payload message must not exceed 3000 characters."""
        argument = "X" * 2900
        payload = json.dumps({"type": "debate-message", "data": {"message": argument}})
        parsed = json.loads(payload)
        assert len(parsed["data"]["message"]) <= 3000

    def test_citation_format_regex(self):
        """Verify that the regex used in tests correctly identifies citations."""
        good = "Emissions fell (source: https://un.org/emissions). Costs dropped (source: https://irena.org/costs)."
        matches = re.findall(r'\(source:\s*https?://\S+\)', good)
        assert len(matches) == 2

    def test_no_filler_openings(self):
        """Simple guardrail check for filler patterns."""
        fillers = ["certainly ", "here is ", "as an ai", "great question", "of course,"]
        good_start = "Solar energy now accounts for 12% of global power generation"
        for f in fillers:
            assert not good_start.lower().startswith(f), f"Filler detected: {f}"


# ──────────────────────────────────────────────────────────────────────────────
# Integration — mock WebSocket server
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestWebSocketClientIntegration:

    @patch("src.wss_client.MAX_RECONNECT_DURATION_SECONDS", 0)
    @patch("src.wss_client.run_debate_turn")
    async def test_client_sends_debate_message_on_our_turn(self, mock_run_turn):
        """
        Spin up a mock server that sends a match-state with our turn.
        Verify the client responds with a correctly formatted debate-message.
        """
        mock_run_turn.return_value = (
            "Renewable energy investment surged 300% in 5 years (source: https://iea.org/2024). "
            "Storage technology matured, resolving grid reliability concerns (source: https://irena.org/storage). "
            "The transition is technologically and economically viable."
        )

        sent_messages: list[dict] = []

        async def mock_server(websocket):
            # Send welcome then match-state with our turn
            await websocket.send(WELCOME_MSG)
            await asyncio.sleep(0.05)
            await websocket.send(MATCH_STATE_OUR_TURN)
            # Collect client's response
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=10)
                sent_messages.append(json.loads(raw))
            except asyncio.TimeoutError:
                pass
            await websocket.send(MATCH_FINISH_MSG)

        async with websockets.serve(mock_server, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            mock_wss_url = f"ws://127.0.0.1:{port}"

            from src.config import Settings
            from src.wss_client import DebateClient
            client = DebateClient()
            # Inject settings directly to avoid lru_cache stale state
            client.settings = Settings(
                wss_url=mock_wss_url,
                base_url="http://127.0.0.1:9999/v1",
                our_team_name="team1",
                model_researcher="gemini-3.0-flash",
                model_debater="gemini-3.1-pro",
                tavily_api_keys="tvly-dummy",
                langchain_tracing_v2="false",
                langchain_api_key="dummy",
            )
            await asyncio.wait_for(client.run(), timeout=15)

        assert len(sent_messages) == 1, "Client should send exactly one message"
        msg = sent_messages[0]
        assert msg["type"] == "debate-message"
        assert "message" in msg["data"]
        assert len(msg["data"]["message"]) <= 3000

    @patch("src.wss_client.MAX_RECONNECT_DURATION_SECONDS", 0)
    @patch("src.wss_client.run_debate_turn")
    async def test_client_does_not_respond_on_opponent_turn(self, mock_run_turn):
        """Client must NOT send any message when it is the opponent's turn."""
        mock_run_turn.return_value = "Should not be called."
        sent_messages: list[dict] = []

        async def mock_server(websocket):
            await websocket.send(WELCOME_MSG)
            await asyncio.sleep(0.05)
            await websocket.send(MATCH_STATE_OPPONENT_TURN)
            await asyncio.sleep(0.5)
            # Try collecting any unsolicited message
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=1)
                sent_messages.append(json.loads(raw))
            except asyncio.TimeoutError:
                pass
            await websocket.send(MATCH_FINISH_MSG)

        async with websockets.serve(mock_server, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            mock_wss_url = f"ws://127.0.0.1:{port}"

            with patch.dict("os.environ", {
                "WSS_URL": mock_wss_url,
                "BASE_URL": "http://127.0.0.1:9999/v1",
                "OUR_TEAM_NAME": "team1",
                "MODEL_RESEARCHER": "gemini-3.0-flash",
                "MODEL_DEBATER": "gemini-3.1-pro",
                "TAVILY_API_KEYS": "tvly-dummy",
                "LANGCHAIN_TRACING_V2": "false",
                "LANGCHAIN_API_KEY": "dummy",
            }):
                from src.config import get_settings
                get_settings.cache_clear()
                from src.wss_client import DebateClient
                client = DebateClient()
                client.settings.wss_url = mock_wss_url
                await asyncio.wait_for(client.run(), timeout=10)

        assert len(sent_messages) == 0, "Client must NOT respond when it's not our turn"
        mock_run_turn.assert_not_called()

    @patch("src.wss_client.MAX_RECONNECT_DURATION_SECONDS", 0)
    @patch("src.wss_client.run_debate_turn")
    async def test_client_restores_history_on_reconnect(self, mock_run_turn):
        """Client should restore message history from previous-message event."""
        mock_run_turn.return_value = "Our argument (source: https://a.com) strong (source: https://b.com)."

        captured_state: dict = {}

        async def mock_server(websocket):
            await websocket.send(WELCOME_MSG)
            # Send previous-message (simulating mid-match reconnect)
            prev = json.dumps({
                "type": "previous-message",
                "from": "system",
                "data": {
                    "message": "Match is already live! Here are the previous conversations.",
                    "conversations": [
                        {"team": "team2", "message": "Old opponent argument.", "timestamp": "2026-03-28T10:00:00Z"},
                        {"team": "team1", "message": "Our old argument.", "timestamp": "2026-03-28T10:01:00Z"},
                    ],
                },
            })
            await websocket.send(prev)
            await asyncio.sleep(0.05)
            await websocket.send(MATCH_STATE_OUR_TURN)
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=10)
                captured_state["msg"] = json.loads(raw)
            except asyncio.TimeoutError:
                pass
            await websocket.send(MATCH_FINISH_MSG)

        async with websockets.serve(mock_server, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            mock_wss_url = f"ws://127.0.0.1:{port}"

            from src.config import Settings
            from src.wss_client import DebateClient
            client = DebateClient()
            client.settings = Settings(
                wss_url=mock_wss_url,
                base_url="http://127.0.0.1:9999/v1",
                our_team_name="team1",
                model_researcher="gemini-3.0-flash",
                model_debater="gemini-3.1-pro",
                tavily_api_keys="tvly-dummy",
                langchain_tracing_v2="false",
                langchain_api_key="dummy",
            )
            await asyncio.wait_for(client.run(), timeout=15)

        # The run_debate_turn should have been called with old history restored
        call_args = mock_run_turn.call_args
        assert call_args is not None
        history = call_args.kwargs.get("message_history") or call_args.args[4]
        assert len(history) >= 2
