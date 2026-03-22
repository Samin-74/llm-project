# Fact-Check Debate System

## The Problem Solved
**Misinformation & Information Verification.** 
Individuals frequently rely on large language models (LLMs) for factual queries, but models can produce unsubstantiated claims. Traditional search engines return unaggregated links without synthesizing arguments. This system addresses the challenge of automated fact-verification through a structured multi-agent debate framework (Proposer vs. Skeptic) evaluated by a designated Evaluation node.
The system securely processes a statement, fetches targeted evidence from the web, and computes a validated consensus score without user intervention.

## Grading Outline (Total 100 Points)
* **Agent-based system solving a real-life problem:** Automates information verification via an adversarial multi-agent workflow.
* **Source Code:** Provided within the `src/` and `data/` directories.
* **Demo Webpage:** A Streamlit-based interface (`src/app.py`) visualizing argument generation and dynamically citing retrieved sources in real-time.

## Tech Stack
* **Framework:** LangGraph (Stateful Multi-Agent Workflows)
* **Models:** OpenRouter (configured in `src/agents/llm_utils.py`, currently using `stepfun/step-3.5-flash:free`).
* **Tools:** Tavily Search API with LRU Caching for robust search quota management.
* **UI:** Streamlit (Real-time system state bridging and interactive UI).

## Setup
1. `pip install -r requirements.txt`
2. Configure `.env` with required keys (at minimum `OPENROUTER_API_KEY` and `TAVILY_API_KEY`).
3. Execute `python -m streamlit run src/app.py`

## Streamlit Cloud Deployment
1. Deploy the repo with main file `src/app.py`.
2. In Streamlit Cloud app settings, open **Secrets** and add:

```toml
OPENROUTER_API_KEY = "your_openrouter_key"
TAVILY_API_KEY = "your_tavily_key"
```

3. Save secrets and redeploy/reboot the app.

Notes:
- `runtime.txt` pins Python to 3.12 for better compatibility with current LangChain dependencies.
- The app automatically reads keys from Streamlit secrets when environment variables are missing.
