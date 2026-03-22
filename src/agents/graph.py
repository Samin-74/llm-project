from langgraph.graph import StateGraph, END, START
from src.agents.state import AgentState
from src.agents.nodes import proposer_node, skeptic_node, judge_node

def route_debate(state: AgentState):
    """
    Conditional edge logic.
    Route to END only when the Judge explicitly finalizes the debate.
    """
    decision = state.get("judge_decision", "[REBUTTAL_REQUIRED]")
    
    if decision == "[FINALIZE]":
        return END
    
    return "Proposer"

def build_graph():
    """Compiles the LangGraph workflow."""
    builder = StateGraph(AgentState)
    
    # Add Nodes
    builder.add_node("Proposer", proposer_node)
    builder.add_node("Skeptic", skeptic_node)
    builder.add_node("Judge", judge_node)
    
    # Add Edges
    builder.add_edge(START, "Proposer")
    builder.add_edge("Proposer", "Skeptic")
    builder.add_edge("Skeptic", "Judge")
    
    # Conditional Routing
    builder.add_conditional_edges("Judge", route_debate)
    
    # Compile
    graph = builder.compile()
    return graph
