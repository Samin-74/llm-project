import operator
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Operator.add ensures messages are appended, not overwritten
    messages: Annotated[list[BaseMessage], operator.add]
    
    # Global state variables
    current_claim: str
    skeptic_temp: float
    
    # Internal routing state
    turn_count: int
    finality_score: int
    judge_decision: str
    
    # For UI transparency: store raw sources to show in the Streamlit expander
    search_contexts: Annotated[list[dict], operator.add]
