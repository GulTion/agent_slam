

# 🏆 Agent SLAM 2026: Autonomous Debater

> **An advanced, fully autonomous debate agent engineered for the Agent SLAM 2026 Competition.**

This repository contains the source code for our autonomous debating agent, designed to engage in live, automated debates on complex topics (Finance, Marketing, Ethics) via the official competition WebSocket API. The agent is built to strictly adhere to the Agent SLAM Rulebook, ensuring zero human intervention, zero hallucinations, and high-speed, evidence-backed argumentation.

---

## 📑 Table of Contents
1. [Architecture of the Agent](#1-architecture-of-the-agent)
2. [Debate Strategy Used](#2-debate-strategy-used)
3. [Frameworks, APIs, and Models Used](#3-frameworks-apis-and-models-used)
4. [Instructions for Running / Deploying](#4-instructions-for-running--deploying)
5. [Competition Rule Compliance](#5-competition-rule-compliance)

---

## 1. Architecture of the Agent

Our agent utilizes a decoupled, asynchronous architecture split into two primary domains: a **Resilient WebSocket Client** and a **Multi-Agent Reasoning Pipeline** orchestrated via state graphs.

### A. The Networking Layer (Asynchronous WSS Client)
Built using Python's `websockets` and `asyncio`, the networking layer (`src/wss_client.py`) is responsible for robust communication with The Oracle (Judging Bot). 
* **State Management:** Maintains an in-memory `DebateState` that tracks the match status, turn assignment, exact time remaining, and complete message history.
* **Resilience & Reconnection:** Implements an exponential backoff reconnection strategy (capped at 90 seconds) to handle transient network instability, adhering strictly to the competition's Disconnection & Reconnection Policy.
* **Non-Blocking Execution:** Incoming WebSocket messages are handled asynchronously. When it is our turn, the reasoning pipeline is invoked via `asyncio.to_thread` with a strict `RESPONSE_DEADLINE_SECONDS` (110s) timeout to ensure we never miss the 2-minute server SLA.

### B. The Reasoning Pipeline (LangGraph)
The core intelligence (`src/agent/graph.py`) is modeled as a deterministic Directed Acyclic Graph (DAG) using **LangGraph**. Every debate turn triggers exactly one flow through the following nodes:

1. **The Researcher Node:** Analyzes the debate topic, our assigned stance (PRO/CON), and the opponent's latest argument. It generates 1–3 highly specific web search queries.
2. **The Search Executor Node:** Executes the queries asynchronously against real-time web data using a rotating pool of API keys to prevent rate-limiting. It formats the raw HTML/text into a clean, synthesized context block.
3. **The Debater Node:** Ingests the historical context and the fresh web evidence to craft a persuasive, logically sound argument. It strictly limits outputs to under 2,900 characters to provide a safety buffer against the 3,000-character platform limit.

---

## 2. Debate Strategy Used

To maximize our score on the Judging Bot's matrix (Persuasiveness 40%, Logic 30%, API Robustness 20%, Agility 10%), we implemented a dynamic, evidence-first strategy.

* **Anti-Hallucination & Fact-Grounding:** The rules strictly penalize fabricated statistics. Our pipeline forces the Debater Node to *only* draw conclusions from the verified URLs provided by the Search Node. Every claim is followed by an exact, raw-text inline citation `(Source: <URL>)` as mandated by the User Manual.
* **Agile Rebuttals:** The Researcher Node specifically formulates queries designed to fact-check the opponent's previous message. This ensures high marks in the *Agility* category by directly dismantling the opponent's unique points rather than reciting a pre-written script.
* **Time-Aware State Transitions:** The system is deeply aware of the remaining match time. If `< 70 seconds` remain, the `Debater` dynamically switches its system prompt from a *Rebuttal Strategy* to a *Closing Argument Strategy*. Instead of introducing new evidence, it synthesizes our strongest points to leave a persuasive final impression on The Oracle.

---

## 3. Frameworks, APIs, and Models Used

Our agent relies on a modern, asynchronous Python stack:

### Frameworks & Libraries
* **Language/Runtime:** Python `3.12+` managed via `uv`.
* **LangGraph (`langgraph>=1.1.3`):** Orchestrates the multi-step reasoning pipeline (DAG).
* **LangChain (`langchain-openai>=1.1.12`):** Provides the LLM abstraction and tool-calling interfaces.
* **WebSockets (`websockets>=16.0`):** Manages the low-level, async WebSocket protocol required to interface with the Agent SLAM match server.
* **Pydantic Settings:** Type-safe environment variable management.

### APIs & LLM Models
* **Model 1: Gemini Flash (Researcher):** Used for low-latency query generation and tool-calling (`tavily_search`).
* **Model 2: Gemini Pro (Debater):** Used for high-reasoning, final argument synthesis, ensuring maximum logic and persuasiveness.
* **Tavily Search API:** A search engine optimized for AI agents, used to retrieve real-time, factual data to substantiate debate claims and eliminate hallucinations.

---

## 4. Instructions for Running / Deploying

### Prerequisites
1. **Python 3.12+** installed on your system.
2. **`uv` package manager** installed (recommended for fast dependency resolution).

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd agent_slam
   ```
2. **Configure Environment Variables:**
   Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure the following critical parameters:
   * `WSS_URL`: The WebSocket connection link provided by the Admin (must include the embedded token).
   * `OUR_TEAM_NAME`: Your exact team name as registered (e.g., `team1`). **Crucial for turn detection.**
   * `TAVILY_KEYS`: Comma-separated list of Tavily API keys (used for round-robin load balancing).
   * `BASE_URL` & `API_KEY` .

### Running the Agent
Once configured, use `uv` to install dependencies and execute the agent in one command:
```bash
uv run python -m src.main
```
**Expected Output:**
The terminal will log the connection status. Once connected, the agent will idle until it receives the `welcome` and `match-state` payloads, at which point it operates entirely autonomously.

### Testing (Sandbox Mode)
To verify your API formatting and network stability prior to the event, point the `WSS_URL` in your `.env` to the Sandbox WebSocket link provided by the organizers and start the agent. 

---

## 5. Competition Rule Compliance

This repository has been engineered to strictly comply with the **Agent SLAM 2026 Rulebook**:
* **Payload Limits (Rule 5):** Hard-coded character truncation at 2,900 characters ensures we never hit the 3,000 character limit. Payloads are strictly serialized to valid JSON.
* **Zero Autonomy Violation (Rule 5):** The script requires zero human intervention once initiated. All responses are triggered by server-side `match-state` broadcasts.
* **Response Timing (Rule 4.2):** The pipeline has a hard timeout of 110 seconds, guaranteeing a payload is fired before the 2-minute penalty threshold.
* **Code Freeze Policy (Rule 11):** We acknowledge that no further commits, updates, or modifications will be made to this repository, the agent, or its deployment environment after the official start time.

---
*Sponsored by Incresol. Competent. Collaborative. Consistent.*