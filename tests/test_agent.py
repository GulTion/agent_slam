"""
Tests for the LangGraph agent (graph.py).

All external calls (LLM + Tavily) are mocked so no real network calls are made.
Run: uv run pytest tests/test_agent.py -v
"""
import re
import pytest
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture: minimal valid GraphState
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    return {
        "topic": "AI regulation is necessary for public safety",
        "stance": "PRO",
        "our_team": "team1",
        "opponent_message": "AI self-regulates through market forces. Government intervention stifles innovation (source: https://nber.org/paper/01).",
        "message_history": [
            "[team2]: AI self-regulates through market forces. Government intervention stifles innovation (source: https://nber.org/paper/01).",
        ],
        "research_queries": [],
        "search_results": [],
        "research_context": "",
        "final_argument": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# researcher_node tests
# ──────────────────────────────────────────────────────────────────────────────

class TestResearcherNode:

    def _make_ai_message_with_tool_call(self, queries: list[str]):
        """Build a mock AIMessage that looks like a proper tool call response."""
        msg = MagicMock()
        msg.tool_calls = [
            {
                "name": "tavily_search",
                "args": {"queries": queries},
                "id": "call_abc123",
            }
        ]
        msg.content = ""
        return msg

    @patch("src.agent.graph._build_llm")
    def test_researcher_returns_queries(self, mock_build_llm, base_state):
        """Researcher node should extract queries from LLM tool_calls."""
        from src.agent.graph import researcher_node

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_with_tools.invoke.return_value = self._make_ai_message_with_tool_call(
            ["AI regulation effectiveness statistics 2024", "AI market failure examples"]
        )
        mock_build_llm.return_value = mock_llm

        result = researcher_node(base_state)

        assert "research_queries" in result
        assert len(result["research_queries"]) == 2
        assert "AI regulation effectiveness statistics 2024" in result["research_queries"]

    @patch("src.agent.graph._build_llm")
    def test_researcher_fallback_on_missing_tool_call(self, mock_build_llm, base_state):
        """If the LLM returns no tool_calls, a fallback query should be generated."""
        from src.agent.graph import researcher_node

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        fallback_msg = MagicMock()
        fallback_msg.tool_calls = []
        fallback_msg.content = "I cannot find any queries."
        mock_llm_with_tools.invoke.return_value = fallback_msg
        mock_build_llm.return_value = mock_llm

        result = researcher_node(base_state)

        assert "research_queries" in result
        assert len(result["research_queries"]) >= 1
        # Fallback must contain the topic keywords
        assert "AI regulation" in result["research_queries"][0] or "evidence" in result["research_queries"][0]


# ──────────────────────────────────────────────────────────────────────────────
# search_node tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSearchNode:

    @patch("src.agent.graph._run_tavily_search")
    def test_search_node_builds_context(self, mock_tavily, base_state):
        """search_node should format the Tavily results into a context string."""
        from src.agent.graph import search_node

        mock_tavily.return_value = [
            {"url": "https://example.com/report", "content": "AI regulation reduces bias by 30%.", "query": "AI regulation statistics"},
            {"url": "https://stanford.edu/ai", "content": "AI incidents increased 40% without oversight.", "query": "AI market failure"},
        ]
        base_state["research_queries"] = ["AI regulation statistics", "AI market failure"]

        result = search_node(base_state)

        assert "search_results" in result
        assert "research_context" in result
        assert "https://example.com/report" in result["research_context"]
        assert "https://stanford.edu/ai" in result["research_context"]
        assert len(result["search_results"]) == 2

    @patch("src.agent.graph._run_tavily_search")
    def test_search_node_handles_empty_results(self, mock_tavily, base_state):
        """search_node should handle zero results gracefully."""
        from src.agent.graph import search_node

        mock_tavily.return_value = []
        base_state["research_queries"] = ["obscure topic"]

        result = search_node(base_state)

        assert result["research_context"] == "No search results were retrieved."
        assert result["search_results"] == []


# ──────────────────────────────────────────────────────────────────────────────
# debater_node tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDebaterNode:

    def _mock_debater_response(self, content: str):
        msg = MagicMock()
        msg.content = content
        return msg

    @patch("src.agent.graph._build_llm")
    def test_debater_returns_under_3000_chars(self, mock_build_llm, base_state):
        """Debater output must never exceed MAX_RESPONSE_CHARS."""
        from src.agent.graph import debater_node, MAX_RESPONSE_CHARS

        long_stub = "A" * 4000
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_debater_response(long_stub)
        mock_build_llm.return_value = mock_llm

        base_state["research_context"] = "URL: https://a.com\nSummary: Some fact.\n"
        result = debater_node(base_state)

        assert len(result["final_argument"]) <= MAX_RESPONSE_CHARS

    @patch("src.agent.graph._build_llm")
    def test_debater_two_citations_present(self, mock_build_llm, base_state):
        """The debater's output must contain exactly 2 source citations."""
        from src.agent.graph import debater_node

        argument = (
            "Unregulated AI poses systemic risks — bias rates in hiring AI reached 35% (source: https://mit.edu/ai-bias). "
            "A peer-reviewed study confirms accidents increase 40% without oversight (source: https://stanford.edu/ai-safety). "
            "The opponent's claim about market self-regulation is contradicted by documented evidence of repeated failures."
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_debater_response(argument)
        mock_build_llm.return_value = mock_llm

        base_state["research_context"] = "URL: https://mit.edu/ai-bias\nURL: https://stanford.edu/ai-safety\n"
        result = debater_node(base_state)

        citation_count = len(re.findall(r'\(source:\s*https?://\S+\)', result["final_argument"]))
        assert citation_count == 2, f"Expected 2 citations, found {citation_count}"

    @patch("src.agent.graph._build_llm")
    def test_debater_no_filler_opening(self, mock_build_llm, base_state):
        """The response must NOT start with common filler phrases."""
        from src.agent.graph import debater_node

        # Simulate a well-formed response
        good_argument = (
            "Regulatory frameworks directly reduce AI-related harms, as evidenced by a 30% drop in "
            "algorithmic bias after GDPR enforcement (source: https://example.com/gdpr-ai). "
            "Furthermore, without mandatory audits, market incentives favour profit over safety (source: https://nber.org/ai-incentives)."
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_debater_response(good_argument)
        mock_build_llm.return_value = mock_llm

        base_state["research_context"] = "URL: https://example.com/gdpr-ai\nURL: https://nber.org/ai-incentives\n"
        result = debater_node(base_state)

        fillers = ["certainly", "here is", "as an ai", "great question", "of course", "sure,"]
        opening = result["final_argument"][:60].lower()
        for filler in fillers:
            assert filler not in opening, f"Filler phrase detected: {filler!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline smoke test
# ──────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    @patch("src.agent.graph._run_tavily_search")
    @patch("src.agent.graph._build_llm")
    def test_end_to_end_pipeline(self, mock_build_llm, mock_tavily):
        """Run the full graph with mocked LLM and Tavily; check output."""
        from src.agent.graph import run_debate_turn
        import re

        # Mock search results
        mock_tavily.return_value = [
            {"url": "https://who.int/ai-safety", "content": "WHO recommends mandatory AI audits.", "query": "AI safety"},
            {"url": "https://ec.europa.eu/ai-act", "content": "EU AI Act passed with bipartisan support.", "query": "AI regulation EU"},
        ]

        # Mock LLMs
        researcher_resp = MagicMock()
        researcher_resp.tool_calls = [
            {"name": "tavily_search", "args": {"queries": ["AI safety WHO", "EU AI Act impact"]}, "id": "c1"}
        ]
        researcher_resp.content = ""

        debater_resp = MagicMock()
        debater_resp.content = (
            "Mandatory AI regulation is not optional — the WHO explicitly calls for audits to prevent harm (source: https://who.int/ai-safety). "
            "The EU AI Act demonstrates that regulation and innovation coexist, with adoption rates rising 15% post-enactment "
            "(source: https://ec.europa.eu/ai-act). The opponent's assertion of market self-correction ignores documented failures."
        )

        # Alternate mock LLM instances for the two invocations
        mock_llm_researcher = MagicMock()
        mock_llm_researcher_bound = MagicMock()
        mock_llm_researcher.bind_tools.return_value = mock_llm_researcher_bound
        mock_llm_researcher_bound.invoke.return_value = researcher_resp

        mock_llm_debater = MagicMock()
        mock_llm_debater.invoke.return_value = debater_resp

        mock_build_llm.side_effect = [mock_llm_researcher, mock_llm_debater]

        result = run_debate_turn(
            topic="AI regulation is necessary for public safety",
            stance="PRO",
            our_team="team1",
            opponent_message="Markets regulate AI better than governments.",
            message_history=["[team2]: Markets regulate AI better than governments."],
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) <= 2900
        assert len(re.findall(r'\(source:\s*https?://\S+\)', result)) == 2
