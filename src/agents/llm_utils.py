import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def get_llm(model_role: str = "flash", temperature: float = 0.5):
    """
    Returns an OpenRouter chat model instance.
    Uses stepfun/step-3.5-flash:free as the unified model backend.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in the .env file.")

    return ChatOpenAI(
        model="stepfun/step-3.5-flash:free",
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=2,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Fact-Check Debate System",
        },
    )
