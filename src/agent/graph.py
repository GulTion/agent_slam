"""
LangGraph debate pipeline.
Exactly two LLM calls per turn:
  1. Researcher (Gemini Flash)  — calls Tavily tool to gather real evidence
  2. Debater (Gemini Pro)       — crafts the final < 3000 char argument
"""

import logging
import random

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.agent.prompts import DEBATER_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT
from src.agent.state import GraphState
from src.config import get_settings

logger = logging.getLogger(__name__)

MAX_RESPONSE_CHARS = 2_900  # keep a 100-char safety buffer below the 3 000 limit


# ──────────────────────────────────────────────────────────────────────────────
# Tavily wrapper (selects a random key from the pool on every call)
# ──────────────────────────────────────────────────────────────────────────────

def _get_tavily_client():
    """Return a TavilyClient initialised with a randomly chosen API key."""
    from tavily import TavilyClient

    settings = get_settings()
    key = random.choice(settings.tavily_keys_list)
    logger.debug("Using Tavily key pool index (masked): %s***", key[:8])
    return TavilyClient(api_key=key)


def _run_tavily_search(queries: list[str]) -> list[dict]:
    """
    Execute each query against Tavily and collect results.
    Returns a list of dicts: {url, content, query}.
    A fresh random key is used for EVERY call to distribute load.
    """
    client = _get_tavily_client()
    results: list[dict] = []
    for q in queries:
        try:
            logger.info("Tavily search: %r", q)
            resp = client.search(query=q, max_results=3, search_depth="advanced")
            for r in resp.get("results", []):
                results.append(
                    {
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:800],
                        "query": q,
                    }
                )
        except Exception as exc:
            logger.warning("Tavily search failed for query %r: %s", q, exc)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build LLM instances
# ──────────────────────────────────────────────────────────────────────────────

def _build_llm(model_name: str, *, temperature: float = 0.7) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key="dummy",          # proxy handles auth; key is irrelevant
        model=model_name,
        temperature=temperature,
        max_tokens=1024,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 — Researcher  (LLM call #1)
# ──────────────────────────────────────────────────────────────────────────────

def researcher_node(state: GraphState) -> dict:
    """
    Ask the Flash model what to search for.
    The model must call the tavily_search tool with a list of queries.
    We parse the tool_calls out of the response and execute them.
    """
    logger.info("[Researcher] Generating search queries...")
    settings = get_settings()
    llm = _build_llm(settings.model_researcher, temperature=settings.temperature_researcher)

    # Build the human prompt with all context
    history_block = "\n".join(
        f"- {m}" for m in (state.get("message_history") or [])
    ) or "(no prior exchanges)"

    human_content = (
        f"**Debate Topic:** {state['topic']}\n"
        f"**Our Stance:** {state['stance']}\n\n"
        f"**Opponent's Latest Argument:**\n{state.get('opponent_message', '(opening turn — no opponent message yet)')}\n\n"
        f"**Debate History:**\n{history_block}\n\n"
        "Generate targeted search queries to retrieve real evidence using the tavily_search tool."
    )

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    # Define a minimal tool schema so the LLM knows how to call Tavily
    tavily_tool_schema = {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Search the web for factual information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 1–3 specific web search queries.",
                    }
                },
                "required": ["queries"],
            },
        },
    }

    llm_with_tools = llm.bind_tools([tavily_tool_schema], tool_choice="required")
    response: AIMessage = llm_with_tools.invoke(messages)

    # Parse tool call arguments
    queries: list[str] = []
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "tavily_search":
                raw_queries = tc["args"].get("queries", [])
                if isinstance(raw_queries, list):
                    queries.extend(raw_queries)
                elif isinstance(raw_queries, str):
                    queries.append(raw_queries)

    # Fallback: derive a basic query from the topic if parsing fails
    if not queries:
        logger.warning("[Researcher] No tool_calls found — using fallback query.")
        queries = [f"{state['topic']} evidence statistics 2024"]

    logger.info("[Researcher] Queries: %s", queries)
    return {"research_queries": queries}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 — Tool Executor  (Tavily search — NOT an LLM call)
# ──────────────────────────────────────────────────────────────────────────────

def search_node(state: GraphState) -> dict:
    """Execute the queries and format a clean context string for the Debater."""
    queries = state.get("research_queries") or []
    raw_results = _run_tavily_search(queries)

    # Format context block for the Debater
    lines: list[str] = []
    for r in raw_results:
        lines.append(f"URL: {r['url']}\nSummary: {r['content']}\n")

    context = "\n".join(lines) if lines else "No search results were retrieved."
    logger.info("[Search] Retrieved %d result(s).", len(raw_results))
    return {"search_results": raw_results, "research_context": context}


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 — Debater  (LLM call #2)
# ──────────────────────────────────────────────────────────────────────────────

def debater_node(state: GraphState) -> dict:
    """Craft the final debate argument using facts from the research context."""
    logger.info("[Debater] Composing final argument...")
    settings = get_settings()
    llm = _build_llm(settings.model_debater, temperature=settings.temperature_debater)

    history_block = "\n".join(
        f"- {m}" for m in (state.get("message_history") or [])
    ) or "(no prior exchanges)"

    human_content = (
        f"**Debate Topic:** {state['topic']}\n"
        f"**Time Remaining in Match:** {state.get('time_remaining', 0)} seconds\n"
        f"**Our Stance:** {state['stance']}\n\n"
        f"**Opponent's Latest Argument:**\n{state.get('opponent_message', '(opening turn)')}\n\n"
        f"**Debate History (oldest first):**\n{history_block}\n\n"
        f"**Research Context (use ONLY these URLs for citations):**\n{state.get('research_context', 'None available.')}\n\n"
        "Compose the final argument. Remember: under 3 000 characters, exactly two inline "
        "citations using `(Source: URL)` format, no filler intro."
    )

    messages = [
        SystemMessage(content=DEBATER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response: AIMessage = llm.invoke(messages)
    argument = response.content.strip()

    # Hard truncate as safety net before transmit
    if len(argument) > MAX_RESPONSE_CHARS:
        logger.warning(
            "[Debater] Response too long (%d chars); truncating to %d.",
            len(argument),
            MAX_RESPONSE_CHARS,
        )
        truncated = argument[:MAX_RESPONSE_CHARS]
        last_period_idx = truncated.rfind('.')
        if last_period_idx != -1:
            argument = truncated[:last_period_idx + 1]
        else:
            argument = truncated

    logger.info("[Debater] Final argument length: %d chars.", len(argument))
    return {"final_argument": argument}


# ──────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the debate state graph."""
    g = StateGraph(GraphState)

    g.add_node("researcher", researcher_node)
    g.add_node("search", search_node)
    g.add_node("debater", debater_node)

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "search")
    g.add_edge("search", "debater")
    g.add_edge("debater", END)

    return g.compile()


# Singleton — built once
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_debate_turn(
    topic: str,
    stance: str,
    our_team: str,
    opponent_message: str,
    message_history: list[str],
    time_remaining: int,
) -> str:
    """
    Public interface called by the WebSocket client for every turn.
    Returns the final argument string ready for broadcast.
    """
    graph = get_graph()
    initial_state: GraphState = {
        "topic": topic,
        "stance": stance,
        "our_team": our_team,
        "time_remaining": time_remaining,
        "opponent_message": opponent_message,
        "message_history": message_history,
        "research_queries": [],
        "search_results": [],
        "research_context": "",
        "final_argument": "",
    }
    result = graph.invoke(initial_state)
    return result.get("final_argument", "")
