import os
import json
import re
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is importable when Streamlit executes src/app.py directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(override=True)

def _load_cloud_secrets_to_env() -> None:
    """Mirror Streamlit secrets into environment variables when missing."""
    required_keys = ["OPENROUTER_API_KEY", "TAVILY_API_KEY"]

    for key in required_keys:
        if os.environ.get(key):
            continue

        value = st.secrets.get(key)
        if value:
            os.environ[key] = str(value)

    # Optional grouped secrets section support, e.g. [api_keys] in secrets.toml.
    api_keys_section = st.secrets.get("api_keys")
    if hasattr(api_keys_section, "get"):
        for key in required_keys:
            if not os.environ.get(key):
                value = api_keys_section.get(key)
                if value:
                    os.environ[key] = str(value)


_load_cloud_secrets_to_env()

from src.agents.graph import build_graph
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Fact-Check Debate System", page_icon="⚖️", layout="wide")

# UI Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .claim-box {
        background-color: #F3F4F6;
        border-left: 5px solid #2563EB;
        padding: 1.5rem;
        border-radius: 5px;
        margin-bottom: 2rem;
        font-size: 1.3rem;
        font-style: italic;
    }
    .score-badge {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin-top: 10px;
        display: inline-block;
        border: 1px solid;
    }
    .score-high { background-color: #DEF7EC; color: #046C4E; border-color: #31C48D; }
    .score-med { background-color: #FEF08A; color: #9A3412; border-color: #FACA15; }
    .score-low { background-color: #FDE8E8; color: #9B1C1C; border-color: #F8B4B4; }
    .conclusion-banner {
        text-align: center;
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #EFF6FF;
        border: 2px solid #3B82F6;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .round-chip {
        display: inline-block;
        margin-bottom: 0.5rem;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #1E3A8A;
        background: #DBEAFE;
        border: 1px solid #93C5FD;
    }
    .sidebar-stat {
        padding: 0.65rem 0.8rem;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        background: rgba(15, 23, 42, 0.25);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>⚖️ Autonomous Fact-Check Debate System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A Multi-Agent Framework for Rigorous Truth-Seeking</div>", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_claims():
    """Load selectable claims from disk with a safe fallback payload."""
    try:
        with open("data/snippets.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return [{"claim": "The Great Wall of China is visible from the Moon.", "category": "Fallback"}]
claims_dataset = load_claims()

# Mapping Roles to Avatars
AVATARS = {
    "proposer": "🛡️",
    "skeptic": "🔍",
    "judge": "🧑‍⚖️"
}


def get_round_number(messages) -> int:
    """A round equals one Proposer turn + one Skeptic turn."""
    participant_msgs = sum(
        1
        for m in messages
        if (getattr(m, "name", "") or "").lower() in ("proposer", "skeptic")
    )
    return max(1, (participant_msgs + 1) // 2)


def get_latest_context_for_role(search_contexts, role):
    """Return the most recent search context for a given debate role."""
    for ctx in reversed(search_contexts or []):
        if isinstance(ctx, dict) and ctx.get("role") == role:
            return ctx
    for ctx in reversed(search_contexts or []):
        if isinstance(ctx, dict) and "query" in ctx:
            return ctx
    return {"query": "N/A", "results": []}

# Sidebar
with st.sidebar:
    st.header("⚙️ Debate Settings")

    input_mode = st.radio("Statement Input Mode", ["Select from dataset", "Custom Statement"])
    if input_mode == "Select from dataset":
        selected_item = st.selectbox(
            "Select a statement:",
            options=claims_dataset,
            format_func=lambda x: f"[{x.get('category', 'Misc')}] {x.get('claim')}"
        )
        current_claim = selected_item['claim']
    else:
        current_claim = st.text_area("Enter a custom statement to fact-check:")

    st.divider()

    chaos_mode = st.slider(
        "Skeptic Agent Volatility",
        min_value=0.0,
        max_value=1.5,
        value=0.5,
        step=0.1,
        help="Higher values increase the variance in the Skeptic agent's search and argument generation."
    )

    start_debate = st.button("🚀 Start Debate Cycle", type="primary", use_container_width=True)

    if st.session_state.get("messages"):
        participant_msgs = [
            m
            for m in st.session_state["messages"]
            if (getattr(m, "name", "") or "").lower() in ("proposer", "skeptic")
        ]
        judges = [
            m
            for m in st.session_state["messages"]
            if (getattr(m, "name", "") or "").lower() == "judge"
        ]
        round_count = len(participant_msgs) // 2
        st.divider()
        st.markdown("### Debate Timeline")
        st.markdown(f"<div class='sidebar-stat'><b>Rounds Completed:</b> {round_count}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-stat'><b>Total Rebuttals:</b> {len(participant_msgs)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-stat'><b>Judge Evaluations:</b> {len(judges)}</div>", unsafe_allow_html=True)

if start_debate and not current_claim.strip():
    st.warning("Please enter a valid statement to start the debate.")
elif start_debate:
    st.session_state["messages"] = []
    st.session_state["search_contexts"] = []
    st.session_state["turns"] = 0
    st.session_state["debate_active"] = True
    st.session_state["graph"] = build_graph()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["search_contexts"] = []
    st.session_state["debate_active"] = False

# Render claim block if there's an active debate or previous messages
if st.session_state.get("messages") or st.session_state.get("debate_active"):
    st.markdown(f"<div class='claim-box'>\"{current_claim}\"</div>", unsafe_allow_html=True)

# Container to hold all messages to prevent them from disappearing
chat_container = st.container()

with chat_container:
    # Render static chat history
    for idx, msg in enumerate(st.session_state["messages"]):
        role = msg.name.lower() if hasattr(msg, "name") and msg.name else "user"
        avatar = AVATARS.get(role, "🤖")
        round_num = get_round_number(st.session_state["messages"][: idx + 1])

        with st.chat_message(role, avatar=avatar):
            st.markdown(f"**{role.capitalize()}**")
            if role in ("proposer", "skeptic"):
                st.markdown(f"<span class='round-chip'>Round {round_num}</span>", unsafe_allow_html=True)
            
            content = msg.content
            # Special Formatting for Judge output to make score pop visually
            if role == "judge" and "Calculated Finality Score:" in content:
                parts = content.split("*Calculated Finality Score:")
                if len(parts) == 2:
                    main_text = parts[0].strip()
                    st.markdown(main_text)
                    
                    score_text = parts[1].replace("*", "").strip()
                    match = re.search(r"(\d+)", score_text)
                    score_val = int(match.group(1)) if match else 0
                    
                    css_class = "score-low"
                    if score_val >= 8:
                        css_class = "score-high"
                    elif score_val >= 5:
                        css_class = "score-med"
                        
                    st.markdown(f"<div class='score-badge {css_class}'>Finality Score: {score_text}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(content)
            else:
                st.markdown(content)
                
            # Display search context if proposer/skeptic
            if role in ("proposer", "skeptic") and hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("search_context"):
                expander_title = "📚 Supporting Sources" if role == "proposer" else "📚 Counter-Evidence Sources"
                with st.expander(expander_title):
                    ctx = msg.additional_kwargs["search_context"]
                    if not isinstance(ctx, dict):
                        ctx = {"query": "N/A", "results": []}
                    st.write(f"**Query Executed:** {ctx.get('query', 'N/A')}")
                    for res in ctx.get("results", []):
                        st.write(f"- [{res.get('url', 'Link')}]({res.get('url', '')}): *{res.get('content', '')}*")

    # If the debate is fully completed, show a strong concluding banner below the final messages
    if st.session_state.get("messages") and not st.session_state.get("debate_active"):
        final_verdict = ""
        for msg in reversed(st.session_state["messages"]):
            name = getattr(msg, "name", "").lower()
            if name == "judge":
                # First check if the structured winner string was sent
                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("winner"):
                    final_verdict = msg.additional_kwargs["winner"]
                break

        # Keep verdict to one sentence for the conclusion banner.
        if final_verdict:
            sentence_match = re.search(r"^.*?[.!?](?:\s|$)", final_verdict.strip())
            if sentence_match:
                final_verdict = sentence_match.group(0).strip()
            else:
                final_verdict = final_verdict.strip() + "."
                
        final_verdict_html = f"<div style='margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.7); border-radius: 5px; color: #1E3A8A; font-weight: 600; font-size: 1.2rem; border-left: 5px solid #10B981; text-align: left;'>🏆 <b>Winner Declared:</b> {final_verdict}</div>" if final_verdict else ""

        st.markdown(f"""
        <div class="conclusion-banner">
            <h2 style="color: #1E3A8A; margin-bottom: 10px;">🏁 Debate Concluded</h2>
            <p style="color: #3B82F6; font-size: 1.1rem; margin-top: 0px; margin-bottom: 0px;">The evaluation threshold was met and the Judge's decision has been recorded.</p>
            {final_verdict_html}
        </div>
        """, unsafe_allow_html=True)

# Run Graph Stream
if st.session_state["debate_active"] and current_claim.strip():
    graph = st.session_state["graph"]

    initial_state = {
        "messages": [],
        "current_claim": current_claim,
        "skeptic_temp": chaos_mode,
        "finality_score": 0,
        "turn_count": 0,
        "judge_decision": "[REBUTTAL_REQUIRED]",
        "search_contexts": []
    }

    with st.status("🧠 Multi-Agent Debate in Progress...", expanded=True) as status:
        for event in graph.stream(initial_state, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue

            latest_msg = messages[-1]

            if (
                not st.session_state["messages"]
                or st.session_state["messages"][-1].content != latest_msg.content
                or getattr(st.session_state["messages"][-1], "name", "") != getattr(latest_msg, "name", "")
            ):
                role = latest_msg.name.lower() if hasattr(latest_msg, "name") and latest_msg.name else "system"
                avatar = AVATARS.get(role, "🤖")
                
                if role in ("proposer", "skeptic") and event.get("search_contexts"):
                    latest_context = get_latest_context_for_role(event["search_contexts"], role)
                    latest_msg.additional_kwargs["search_context"] = latest_context
                
                st.session_state["messages"].append(latest_msg)

                # Write directly to the main container outside the status box
                with chat_container:
                    with st.chat_message(role, avatar=avatar):
                        st.markdown(f"**{role.capitalize()}**")
                        if role in ("proposer", "skeptic"):
                            st.markdown(f"<span class='round-chip'>Round {get_round_number(st.session_state['messages'])}</span>", unsafe_allow_html=True)
                        
                        content = latest_msg.content
                        if role == "judge" and "Calculated Finality Score:" in content:
                            parts = content.split("*Calculated Finality Score:")
                            if len(parts) == 2:
                                main_text = parts[0].strip()
                                st.markdown(main_text)
                                
                                score_text = parts[1].replace("*", "").strip()
                                match = re.search(r"(\d+)", score_text)
                                score_val = int(match.group(1)) if match else 0
                                
                                css_class = "score-low"
                                if score_val >= 8:
                                    css_class = "score-high"
                                elif score_val >= 5:
                                    css_class = "score-med"
                                    
                                st.markdown(f"<div class='score-badge {css_class}'>Finality Score: {score_text}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(content)
                        else:
                            st.markdown(content)
                            
                        if role in ("proposer", "skeptic") and event.get("search_contexts"):
                            expander_title = "📚 Supporting Sources" if role == "proposer" else "📚 Counter-Evidence Sources"
                            with st.expander(expander_title):
                                ctx = get_latest_context_for_role(event["search_contexts"], role)
                                if not isinstance(ctx, dict):
                                    ctx = {"query": "N/A", "results": []}
                                st.write(f"**Query Executed:** {ctx.get('query', 'N/A')}")
                                for res in ctx.get("results", []):
                                    st.write(f"- [{res.get('url', 'Link')}]({res.get('url', '')}): *{res.get('content', '')}*")

                round_num = get_round_number(st.session_state["messages"])
                if role == "judge":
                    status.update(label=f"Round {round_num}: Judge evaluating completion conditions...", state="running")
                else:
                    status.update(label=f"Round {round_num}: {role.capitalize()} rebuttal in progress...", state="running")
                
        status.update(label="Debate Concluded! Final verdict reached. ✅", state="complete", expanded=False)

    st.session_state["debate_active"] = False
    st.rerun() # Force a rerun so that state refreshes clearly
