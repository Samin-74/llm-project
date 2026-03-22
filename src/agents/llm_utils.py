import os
import logging
from langchain_openai import ChatOpenAI

try:
    import streamlit as st
except Exception:
    st = None

logger = logging.getLogger(__name__)


def _get_secret_value(key: str) -> str | None:
    """Resolve a secret from env first, then Streamlit secrets if available."""
    env_value = os.environ.get(key)
    if env_value:
        return env_value

    if st is None:
        return None

    # 1) Flat key support in secrets.toml
    try:
        flat_value = st.secrets.get(key)
        if flat_value:
            return str(flat_value)
    except Exception:
        pass

    # 2) Grouped key support, e.g. [api_keys] section
    try:
        section = st.secrets.get("api_keys")
        if hasattr(section, "get"):
            grouped_value = section.get(key)
            if grouped_value:
                return str(grouped_value)
    except Exception:
        pass

    return None

def get_llm(model_role: str = "flash", temperature: float = 0.5):
    """
    Returns an OpenRouter chat model instance.
    Uses stepfun/step-3.5-flash:free as the unified model backend.
    """
    api_key = _get_secret_value("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Set it in Streamlit Cloud Secrets or local .env."
        )

    model_name = os.environ.get("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=2,
        timeout=75,
        max_completion_tokens=1500,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Fact-Check Debate System",
        },
    )
