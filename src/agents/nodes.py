import json
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage      
from src.agents.state import AgentState
from src.agents.llm_utils import get_llm
from src.agents.tools import perform_search

def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return " ".join([item.get("text", "") for item in content if isinstance(item, dict) and "text" in item])
    return str(content)


def normalize_winner_sentence(raw_winner: str, reasoning: str) -> str:
    """Normalize free-form winner text into one concise, non-draw outcome sentence."""
    combined = f"{raw_winner or ''} {reasoning or ''}".lower()

    has_proposer = "proposer" in combined
    has_skeptic = "skeptic" in combined
    is_draw_like = "draw" in combined or "tie" in combined or "both win" in combined

    if not is_draw_like and has_proposer and not has_skeptic:
        sentence = "The Proposer wins by maintaining the stronger overall case under the final criteria."
    elif not is_draw_like and has_skeptic and not has_proposer:
        sentence = "The Skeptic wins by presenting stronger evidence and clearer factual grounding."
    else:
        # Tie-break helper sentence when free-form winner text is ambiguous.
        sentence = "The Proposer wins by presenting the stronger literal defense under the final criteria."

    # Enforce one sentence in case prompt-drift appears in model output.
    return sentence.split(".")[0].strip() + "."

def format_history(messages):
    transcript = []
    for m in messages:
        name = getattr(m, "name", "") or m.__class__.__name__.replace("Message", "")                                                                                    
        text = extract_text(m.content)
        transcript.append(f"{name}: {text}")
    return "\n\n".join(transcript)


def format_history_compact(messages, max_messages: int = 8, max_chars: int = 2600) -> str:
    """Return a compact transcript window to avoid model context overflows/timeouts."""
    if not messages:
        return ""

    window = list(messages)[-max_messages:]
    transcript = format_history(window)
    if len(transcript) <= max_chars:
        return transcript

    # Keep the most recent context because turn-by-turn debate state matters most.
    return transcript[-max_chars:]


def get_latest_role_text(messages, role_name: str) -> str:
    """Return the most recent message text for a specific debate role."""
    for m in reversed(messages or []):
        name = (getattr(m, "name", "") or "").strip().lower()
        if name == role_name.lower():
            return extract_text(getattr(m, "content", ""))
    return ""


def has_semantic_reframing(text: str) -> bool:
    """Detect semantic-evasion phrasing that avoids literal claim interpretation."""
    lowered = (text or "").lower()
    reframing_markers = [
        "symbolic prominence",
        "conceptual prominence",
        "metaphorical",
        "ontological",
        "in the global imagination",
        "not optical",
        "not physically discernible",
    ]
    return any(marker in lowered for marker in reframing_markers)


def clamp_score(value, low: int = 0, high: int = 10) -> int:
    """Coerce a score to a bounded integer value."""
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return low


def strip_process_preface(text: str) -> str:
    """Remove process-oriented lead-ins so responses begin with substantive argumentation."""
    if not text:
        return text

    original = (text or "").strip()
    cleaned = original
    # Remove common lead-ins that mention retrieval/process rather than argument substance.
    lead_patterns = [
        "the retrieved context",
        "based on the retrieved context",
        "from the retrieved context",
        "according to the retrieved context",
    ]

    lowered = cleaned.lower()
    for pattern in lead_patterns:
        if lowered.startswith(pattern):
            # Remove first sentence if it is process-preface.
            first_period = cleaned.find(".")
            if first_period != -1 and first_period + 1 < len(cleaned):
                cleaned = cleaned[first_period + 1 :].strip()
            else:
                cleaned = re.sub(r"^.*?(,|:)\s*", "", cleaned, count=1).strip()
            break

    # If cleanup removed everything, return the original text instead of a blank message.
    return cleaned if cleaned else original


def safe_invoke_text(llm, messages, fallback_text: str = "", retries: int = 1) -> str:
    """Invoke an LLM call with retry and optional fallback text."""
    last_error = ""
    attempts = max(1, retries + 1)

    for _ in range(attempts):
        try:
            response = llm.invoke(messages)
            text = extract_text(getattr(response, "content", ""))
            if text and text.strip():
                return text
        except Exception as e:
            last_error = str(e)

    if fallback_text:
        if last_error:
            return f"{fallback_text} (Temporary model/provider issue: {last_error[:160]})"
        return fallback_text

    return ""


def build_proposer_backup(claim: str, support_results: list) -> str:
    """Generate a deterministic Proposer response when model output is unavailable."""
    urls = [str(r.get("url", "Unknown")) for r in (support_results or []) if isinstance(r, dict)]
    citation = ", ".join(urls[:2]) if urls else "available sources"

    return (
        f"The Skeptic-side framing overstates certainty by treating one observational standard as the only valid lens for assessing '{claim}'. "
        f"Evidence discussions across {citation} show the claim's persistence hinges on scale, detectability, and interpretation under viewing conditions rather than a single binary test. "
        "A defensible Proposer position is that prominence can be argued through geometric extent, contrast-dependent perception, and technology-informed observation from lunar distance. "
        "Therefore, the claim remains contestable in favor of the Proposer when the debate explicitly evaluates detectability and prominence criteria together."
    )


def extract_first_json_object(text: str) -> str:
    """Extract the first top-level JSON object from mixed model output."""
    if not text:
        return ""

    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return ""

def proposer_node(state: AgentState):
    """Generate a Proposer argument or rebuttal grounded in supporting sources."""
    llm = get_llm(temperature=0.7)
    current_claim = state["current_claim"]

    # Retrieve supporting sources so Proposer can ground rebuttals too.
    support_query_prompt = (
        f"Write a concise 3-5 word search query to find evidence supporting this statement: '{current_claim}'"
    )
    query_llm = get_llm(temperature=0.1)
    support_query_content = safe_invoke_text(
        query_llm,
        [HumanMessage(content=support_query_prompt)],
        "supporting evidence query",
    )
    support_query = extract_text(support_query_content).strip('"\'')
    support_results = perform_search(support_query)
    support_context_str = "\n".join(
        [f"- {r.get('url', 'Unknown')}: {r.get('content', '')}" for r in support_results]
    )

    system_prompt = (
        "Role: Zealous Proposer and Statement Defender.\n"
        "Objective: You MUST formulate a logical defense strictly IN FAVOR of the following statement:\n" 
        f"STATEMENT: '{current_claim}'\n\n"
        "CRITICAL INSTRUCTION: You must defend this statement as if it is an absolute undeniable truth, even if you know it is a myth, factually incorrect, or impossible. "
        "NEVER agree with the Skeptic. NEVER concede that the statement is false. Find creative, academic, or theoretical angles to support the core premise.\n"
        "You MUST keep the claim literal. Do NOT redefine key claim words into symbolic, cultural, metaphorical, or conceptual meanings. "
        "Argue the exact physical claim as written.\n"
        "Start directly with the argument itself. Do not begin with process phrases such as 'the retrieved context'.\n"
        "When rebutting the Skeptic, use this concrete structure:\n"
        "1) First sentence identifies a concrete error in the Skeptic's reasoning.\n"
        "2) Second sentence gives 1-2 evidence-grounded rebuttal points.\n"
        "3) Third sentence adds 1-2 new supporting points of your own (not just rebuttal).\n"
        "4) Final sentence gives a firm conclusion IN FAVOR of the claim.\n"
        "Reference the retrieved supporting context when useful and cite sources clearly.\n\n"
        f"RETRIEVED SUPPORTING CONTEXT:\n{support_context_str}\n\n"
        "Provide an initial argument actively supporting the statement, or fiercely rebut the Skeptic's counter-arguments. "
        "Maintain a highly professional, academic tone. Keep the response under 220 words."
    )
    
    transcript = format_history_compact(state["messages"], max_messages=8, max_chars=2200)
    user_prompt = (
        f"Transcript so far:\n{transcript}\n\n"
        "Provide a direct rebuttal using the required 4-sentence structure, including new supporting points of your own."
    ) if transcript else "Please provide your primary argument supporting the statement."
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    model_text = safe_invoke_text(llm, msgs, retries=1)
    response_text = strip_process_preface(model_text)
    if not response_text.strip():
        response_text = build_proposer_backup(current_claim, support_results)
    saved_context = {
        "role": "proposer",
        "query": support_query,
        "results": support_results,
    }
    return {
        "messages": [AIMessage(content=response_text, name="Proposer")],
        "search_contexts": [saved_context],
    }

def skeptic_node(state: AgentState):
    """Generate a Skeptic rebuttal grounded in retrieved counter-evidence."""
    temp = state.get("skeptic_temp", 0.5)
    llm = get_llm(temperature=temp)

    current_claim = state["current_claim"]

    query_prompt = f"Write a concisely phrased 3-5 word search query strictly to find contrary evidence for this statement: '{current_claim}'"                      
    query_llm = get_llm(temperature=0.1)
    
    search_query_content = safe_invoke_text(
        query_llm,
        [HumanMessage(content=query_prompt)],
        "counter-evidence query",
    )
    search_query = extract_text(search_query_content).strip('\"\'')
                                                                  
    search_results = perform_search(search_query)
    context_str = "\n".join([f"- {r.get('url', 'Unknown')}: {r.get('content', '')}" for r in search_results])                                                       
    
    system_prompt = (
        "Role: Critical Analyst.\n"
        "Objective: Challenge the primary statement and provide sourced counter-arguments.\n"                                                                           
        f"STATEMENT UNDER REVIEW: '{current_claim}'\n\n"
        "Enforce literal semantics: explicitly flag and reject any semantic reframing "
        "(e.g., symbolic, conceptual, metaphorical reinterpretation) that avoids the claim as written.\n"
        "Start directly with the counter-argument itself. Do not begin with process phrases such as 'the retrieved context' or 'based on retrieved context'.\n"
        "Write in a direct rebuttal style similar to formal academic refutations.\n"
        "Required structure:\n"
        "1) First sentence identifies a concrete error in the Proposer's reasoning.\n"
        "2) Second sentence gives 1-2 evidence-grounded rebuttal points.\n"
        "3) Third sentence adds 1-2 new critical points of your own (not just rebuttal).\n"
        "4) Final sentence states a firm conclusion against the claim.\n"
        "Reference the following retrieved context to structure a counter-argument. "                                                                                   
        "Identify nuances, discrepancies, or missing context. Cite provided sources clearly.\n\n"                                                                       
        f"RETRIEVED CONTEXT:\n{context_str}\n\n"
        "Maintain a highly professional, academic tone. Keep the response under 220 words."
    )

    transcript = format_history_compact(state["messages"], max_messages=8, max_chars=2200)
    user_prompt = (
        f"Transcript so far:\n{transcript}\n\n"
        "Provide a direct rebuttal in four parts: identify the Proposer's key error, rebut with evidence, add new critical points of your own, and conclude firmly. "
        "Do not use retrieval-process phrasing."
    )
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response_text = strip_process_preface(
        safe_invoke_text(
            llm,
            msgs,
            "The Proposer argument remains unsupported under direct evidence review, but the model response failed."
        )
    )
    if not response_text.strip():
        response_text = "The Proposer argument remains unsupported under direct evidence review, but the model returned no usable text."

    saved_context = {
        "role": "skeptic",
        "query": search_query,
        "results": search_results
    }

    return {
        "messages": [AIMessage(content=response_text, name="Skeptic")],      
        "search_contexts": [saved_context]
    }

def judge_node(state: AgentState):
    """Evaluate debate quality, compute scores, and route the next workflow step."""
    llm = get_llm(temperature=0.1)

    system_prompt = (
        "Role: Evaluation Metric Generator.\n"
        f"OBJECTIVE: Assess the discourse regarding the statement: '{state['current_claim']}'\n\n"                                                                      
        "Analyze the provided arguments, logical cohesion, and sources. Calculate a 'finality_score' (1-10).\n"
        "Primary rule: evaluate the literal statement exactly as written. If any side reframes key claim terms "
        "into symbolic/conceptual meanings, treat it as semantic evasion and penalize heavily.\n"                                                         
        "Evaluation fairness rule: do not privilege skepticism by default. Either side can win if their evidence and logic are stronger.\n"
        "- Score 1-7: Inconclusive; further argument generation is required.\n" 
        "- Score 8-10: Factual limit reached, or cyclical repetition detected (evaluation complete).\n\n"                                                               
        "You MUST declare exactly one winner: Proposer OR Skeptic. Draws are forbidden.\n"
        "Output EXACTLY in the following strict JSON format (no markdown code fences):\n"
        "{\n"
        '  "reasoning": "brief explanation of the evaluation",\n'
        '  "proposer_score": 0,\n'
        '  "skeptic_score": 0,\n'
        '  "winner_role": "Proposer" or "Skeptic",\n'
        '  "winner": "one concise sentence explicitly stating either The Proposer wins... or The Skeptic wins...",\n'
        '  "finality_score": 8,\n'
        '  "decision": "[FINALIZE]" (if score >= 8) or "[REBUTTAL_REQUIRED]" (if score < 8)\n'                                                                          
        "}"
    )

    transcript = format_history_compact(state["messages"], max_messages=10, max_chars=2800)
    user_prompt = f"Transcript to evaluate:\n{transcript}\n\nPlease provide your evaluation in the strictly requested JSON format."                             
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response_text = safe_invoke_text(
        llm,
        msgs,
        '{"reasoning":"Judge fallback due to temporary model/provider issue.","proposer_score":4,"skeptic_score":6,"winner_role":"Skeptic","winner":"The Skeptic wins by presenting stronger evidence and clearer factual grounding.","finality_score":6,"decision":"[REBUTTAL_REQUIRED]"}'
    )

    try:
        content = response_text.replace("```json", "").replace("```", "").strip()
        if not content.startswith("{"):
            extracted = extract_first_json_object(content)
            if extracted:
                content = extracted
        parsed = json.loads(content)
        f_score = clamp_score(parsed.get("finality_score", 5), 1, 10)
        parsed_decision = parsed.get("decision", "[REBUTTAL_REQUIRED]")
        reasoning = parsed.get("reasoning", "No reasoning provided.")
        proposer_score = clamp_score(parsed.get("proposer_score", 0), 0, 10)
        skeptic_score = clamp_score(parsed.get("skeptic_score", 0), 0, 10)
        winner_role = str(parsed.get("winner_role", "")).strip().lower()
        raw_winner = parsed.get("winner", "")

        # Deterministic winner logic with no neutral fallback:
        # winner is selected only when one side has a strict advantage
        # or the judge explicitly names a winner_role.
        winner_selected = True
        if proposer_score > skeptic_score:
            final_winner_role = "proposer"
        elif skeptic_score > proposer_score:
            final_winner_role = "skeptic"
        elif winner_role in ("proposer", "skeptic"):
            final_winner_role = winner_role
        else:
            winner_selected = False
            final_winner_role = ""

        if final_winner_role == "proposer":
            winner = "The Proposer wins by maintaining the stronger overall case under the final criteria."
        elif final_winner_role == "skeptic":
            winner = "The Skeptic wins by presenting stronger evidence and clearer factual grounding."
        else:
            winner = ""

        # Deterministic guardrail: semantic reframing should prevent immediate finalization.
        latest_proposer = get_latest_role_text(state.get("messages", []), "proposer")
        if has_semantic_reframing(latest_proposer):
            f_score = min(f_score, 7)
            proposer_score = min(proposer_score, 4)
            parsed_decision = "[REBUTTAL_REQUIRED]"
            final_winner_role = "skeptic"
            winner_selected = True
            winner = "The Skeptic wins by presenting stronger evidence and clearer factual grounding."
            reasoning = (
                f"{reasoning} Semantic reframing detected in Proposer argument; "
                "literal-claim fidelity penalty applied."
            )

        if not winner_selected:
            parsed_decision = "[REBUTTAL_REQUIRED]"
            f_score = min(f_score, 7)
            reasoning = (
                f"{reasoning} No concrete winner selected (scores tied and winner_role missing); "
                "continuing rebuttal cycle."
            )

        # Continue debate unless finalized; require at least a minimum number of rounds
        # and a concrete winner selection before conclusion.
        min_rounds_before_finalize = 2
        current_round = state.get("turn_count", 0) + 1
        can_finalize_now = current_round >= min_rounds_before_finalize
        should_finalize = (
            winner_selected
            and f_score >= 8
            and parsed_decision == "[FINALIZE]"
            and can_finalize_now
        )
        decision = "[FINALIZE]" if should_finalize else "[REBUTTAL_REQUIRED]"
    except Exception as e:
        f_score = 5
        decision = "[REBUTTAL_REQUIRED]"
        reasoning = f"Parsing error occurred. Continuing debate with conservative fallback. Log: {str(e)}"
        proposer_score = 0
        skeptic_score = 0
        final_winner_role = ""
        winner = ""
        
    judge_message = (
        f"**Evaluation Reasoning:** {reasoning} "
        f"\n\n*Calculated Finality Score: {f_score}/10*"
        f"\n\n*Proposer Score: {proposer_score}/10 | Skeptic Score: {skeptic_score}/10*"
    )
    return {
        "messages": [
            AIMessage(
                content=judge_message,
                name="Judge",
                additional_kwargs={
                    "winner": winner,
                    "winner_role": final_winner_role,
                    "score_breakdown": {
                        "proposer": proposer_score,
                        "skeptic": skeptic_score,
                    },
                },
            )
        ],
        "finality_score": f_score,
        "judge_decision": decision,
        "turn_count": state.get("turn_count", 0) + 1
    }
