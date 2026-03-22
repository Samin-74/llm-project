import os
from functools import lru_cache

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

try:
    from langchain_tavily import TavilySearch
except Exception:
    # Backward-compatible fallback for environments without langchain-tavily.
    from langchain_community.tools.tavily_search import TavilySearchResults

# LRU Cache to prevent burning through Tavily free credits globally
# This ensures if we search the exact same query in 1 session, it's virtually free.
@lru_cache(maxsize=128)
def _cached_tavily_search(query: str):
    """Execute a Tavily query through an LRU-cached wrapper."""
    cleaned_query = " ".join((query or "").split()).strip()[:180]
    if not cleaned_query:
        cleaned_query = "fact check claim"

    # Tavily wrapper
    if "TavilySearch" in globals():
        tool = TavilySearch(max_results=3, search_depth="advanced")
    else:
        tool = TavilySearchResults(max_results=3, search_depth="advanced")

    # Wrapper compatibility: some versions expect string input, others expect dict.
    invoke_errors = []
    for payload in (cleaned_query, {"query": cleaned_query}):
        try:
            return tool.invoke(payload)
        except Exception as e:
            invoke_errors.append(str(e))

    # Fallback to official Tavily client if wrapper signatures are incompatible.
    if TavilyClient is not None and os.environ.get("TAVILY_API_KEY"):
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        return client.search(query=cleaned_query, max_results=3, search_depth="advanced")

    # Bubble the most useful wrapper error if all strategies fail.
    raise RuntimeError(" | ".join(invoke_errors) if invoke_errors else "Tavily search failed")


def _normalize_search_results(raw_results) -> list:
    """Normalize heterogeneous search outputs into a list of url/content dictionaries."""
    if isinstance(raw_results, dict):
        # Some wrappers return {'results': [...]}.
        if isinstance(raw_results.get("results"), list):
            raw_results = raw_results["results"]
        else:
            raw_results = [raw_results]

    if isinstance(raw_results, str):
        return [{"url": "raw_result", "content": raw_results}]

    if not isinstance(raw_results, list):
        return [{"url": "raw_result", "content": str(raw_results)}]

    normalized = []
    max_content_chars = 900
    for item in raw_results:
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("snippet") or item.get("title") or "")
            if len(content) > max_content_chars:
                content = content[:max_content_chars].rstrip() + "..."
            normalized.append(
                {
                    "url": str(item.get("url") or item.get("source") or "Unknown"),
                    "content": content,
                }
            )
        else:
            normalized.append({"url": "raw_result", "content": str(item)})

    return normalized

def perform_search(query: str) -> list:
    """
    Execute a cached web search and return normalized source records.

    Returns:
        list[dict]: Each item includes ``url`` and ``content`` keys.
    """
    if "TAVILY_API_KEY" not in os.environ:
        return [{"url": "error", "content": "TAVILY_API_KEY not set in environment."}]
    
    try:
        results = _cached_tavily_search(query)
        return _normalize_search_results(results)
    except Exception as e:
        return [{"url": "error", "content": f"Search failed: {str(e)}"}]
