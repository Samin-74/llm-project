import os
import sys
import time
import json
import traceback
import re
import argparse
from pathlib import Path

# Add the project root to sys.path so we can import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.graph import build_graph
from evaluation.dataset import EVALUATION_DATASET

RESULTS_FILE = Path(__file__).resolve().parent / "evaluation_results.json"


def _normalize_winner(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw == "proposer":
        return "Proposer"
    if raw == "skeptic":
        return "Skeptic"
    return "Unknown"


def _extract_usage_metadata(msg) -> tuple[int, int, int]:
    """Extract token usage from multiple possible LangChain/OpenRouter metadata shapes."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # 1) usage_metadata (newer LangChain format)
    usage_meta = getattr(msg, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        prompt_tokens += int(usage_meta.get("input_tokens", 0) or 0)
        completion_tokens += int(usage_meta.get("output_tokens", 0) or 0)
        total_tokens += int(usage_meta.get("total_tokens", 0) or 0)

    # 2) response_metadata token usage (provider dependent)
    response_meta = getattr(msg, "response_metadata", None)
    if isinstance(response_meta, dict):
        candidate_dicts = []
        for key in ("token_usage", "usage", "usage_metadata"):
            candidate = response_meta.get(key)
            if isinstance(candidate, dict):
                candidate_dicts.append(candidate)
        candidate_dicts.append(response_meta)

        for candidate in candidate_dicts:
            prompt_tokens += int(
                candidate.get("prompt_tokens", 0)
                or candidate.get("input_tokens", 0)
                or 0
            )
            completion_tokens += int(
                candidate.get("completion_tokens", 0)
                or candidate.get("output_tokens", 0)
                or 0
            )
            total_tokens += int(candidate.get("total_tokens", 0) or 0)

    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def _estimate_tokens_from_text(text: str) -> int:
    """Fast tokenizer-free estimate. Roughly 0.75 tokens per word for this task domain."""
    if not text:
        return 0
    words = re.findall(r"\S+", text)
    return int(round(len(words) * 0.75))


def _estimate_total_tokens(claim: str, messages: list, search_contexts: list) -> int:
    transcript = [claim or ""]
    for msg in messages or []:
        transcript.append(str(getattr(msg, "content", "") or ""))
    for ctx in search_contexts or []:
        transcript.append(str(ctx.get("query", "") or ""))
        results = ctx.get("results", []) or []
        for item in results:
            transcript.append(str(item.get("content", "") or ""))
    return _estimate_tokens_from_text("\n".join(transcript))

def save_checkpoint(results, results_file: Path):
    """Saves progress incrementally so we don't lose data if an API call fails or times out."""
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def _load_existing_results(results_file: Path) -> list:
    if not results_file.exists():
        return []
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def run_evaluations(
    max_claims: int | None = None,
    start_id: int | None = None,
    overwrite: bool = False,
    skip_done: bool = True,
    output_file: str | None = None,
    claim_ids: list[int] | None = None,
    rerun_fails: bool = False,
):
    print("Building graph...")
    graph = build_graph()
    results_file = Path(output_file) if output_file else RESULTS_FILE
    
    existing_results = [] if overwrite else _load_existing_results(results_file)
    existing_by_id = {entry.get("id"): entry for entry in existing_results if isinstance(entry, dict)}
    completed_ids = set(existing_by_id.keys())
    results = list(existing_by_id.values())

    dataset_to_run = list(EVALUATION_DATASET)
    if claim_ids:
        selected_ids = set(claim_ids)
        dataset_to_run = [item for item in dataset_to_run if int(item.get("id", 0)) in selected_ids]

    if rerun_fails:
        failed_ids = {
            int(entry.get("id", 0))
            for entry in existing_results
            if entry.get("expected_alignment") == "Fail" or entry.get("accuracy") == "Fail"
        }
        dataset_to_run = [item for item in dataset_to_run if int(item.get("id", 0)) in failed_ids]

    if start_id is not None:
        dataset_to_run = [item for item in dataset_to_run if int(item.get("id", 0)) >= start_id]
    if skip_done and not rerun_fails and not claim_ids:
        dataset_to_run = [item for item in dataset_to_run if item.get("id") not in completed_ids]
    if max_claims is not None and max_claims > 0:
        dataset_to_run = dataset_to_run[:max_claims]
    
    total_claims = len(dataset_to_run)
    print(f"Starting evaluation suite for {total_claims} claims...\n")
    print(f"Results will be progressively saved to: {results_file.name}\n")
    
    if total_claims == 0:
        print("No claims to run after filters. Generating report from existing results...")
        from evaluation.generate_visuals import generate_report
        generate_report(results_file=results_file)
        return

    for idx, item in enumerate(dataset_to_run, start=1):
        claim_id = item["id"]
        claim = item["claim"]
        ground_truth = item["ground_truth"]
        expected_winner = item.get("expected_winner", "Either")
        topic = item.get("topic", "General")
        skeptic_temp = float(item.get("skeptic_temp", 0.7))
        turn_bias = int(item.get("turn_bias", 0))
        
        print(f"[{idx}/{total_claims}] Evaluating claim_id={claim_id}: \"{claim}\"")
        print(
            f"         (Truth: {ground_truth}, skeptic_temp={skeptic_temp}, turn_bias={turn_bias})"
        )
        
        initial_state = {
            "messages": [],
            "current_claim": claim,
            "skeptic_temp": skeptic_temp,
            "finality_score": 0,
            "turn_count": turn_bias,
            "judge_decision": "",
            "search_contexts": []
        }
        
        start_time = time.time()
        
        try:
            # We use invoke to run the graph synchronously until completion
            final_state = graph.invoke(initial_state, {"recursion_limit": 50})
            
            # Extract basic metrics
            turn_count = final_state.get("turn_count", 0)
            finality_score = final_state.get("finality_score", 0)
            search_contexts = final_state.get("search_contexts", [])
            messages = final_state.get("messages", [])
            rounds_completed = sum(
                1 for m in messages if getattr(m, "name", "") in ("Proposer", "Skeptic")
            ) // 2
            
            # Calculate tokens (provider metadata first, estimator fallback)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            for msg in messages:
                p, c, t = _extract_usage_metadata(msg)
                prompt_tokens += p
                completion_tokens += c
                total_tokens += t

            estimated_tokens = _estimate_total_tokens(claim, messages, search_contexts)
            token_source = "provider_metadata" if total_tokens > 0 else "estimated_from_text"
            resolved_tokens = total_tokens if total_tokens > 0 else estimated_tokens
            
            # Find the final judge message
            winner = "Unknown"
            winner_role = "Unknown"
            proposer_score = 0
            skeptic_score = 0
            parse_success = False
            
            for msg in reversed(messages):
                if msg.name == "Judge":
                    kwargs = msg.additional_kwargs
                    if "winner_role" in kwargs:
                        winner_role = kwargs["winner_role"]
                        winner = kwargs.get("winner", "")
                        
                        score_breakdown = kwargs.get("score_breakdown", {})
                        if isinstance(score_breakdown, dict):
                            proposer_score = score_breakdown.get("proposer", 0)
                            skeptic_score = score_breakdown.get("skeptic", 0)

                    # judge_node always includes winner_role in additional_kwargs,
                    # so parse success must be inferred from the fallback marker.
                    msg_text = str(getattr(msg, "content", "") or "")
                    used_heuristic = "heuristic fallback" in msg_text.lower()
                    parse_success = not used_heuristic

                    if parse_success is False:
                        if "Proposer wins" in msg_text:
                            winner_role = "proposer"
                        elif "Skeptic wins" in msg_text:
                            winner_role = "skeptic"
                    break

            winner_role_normalized = _normalize_winner(winner_role)
            
            # Determine alignment
            alignment = "Fail"
            if ground_truth == "True" and winner_role_normalized == "Proposer":
                alignment = "Pass"
            elif ground_truth == "False" and winner_role_normalized == "Skeptic":
                alignment = "Pass"
            elif ground_truth == "Ambiguous":
                alignment = "N/A (Ambiguous)" 

            expected_alignment = "N/A"
            if expected_winner in ("Proposer", "Skeptic"):
                expected_alignment = "Pass" if winner_role_normalized == expected_winner else "Fail"

            score_margin = proposer_score - skeptic_score
            # Count searches
            num_searches = len(search_contexts)
            failed_searches = sum(1 for ctx in search_contexts if not ctx.get("results", []))
            
            duration = round(time.time() - start_time, 2)
            
            result_entry = {
                "id": claim_id,
                "claim": claim,
                "ground_truth": ground_truth,
                "expected_winner": expected_winner,
                "topic": topic,
                "skeptic_temp": skeptic_temp,
                "turn_bias": turn_bias,
                "winner_role": winner_role_normalized,
                "accuracy": alignment,
                "expected_alignment": expected_alignment,
                "turn_count": turn_count,
                "rounds_completed": rounds_completed,
                "finality_score": finality_score,
                "proposer_score": proposer_score,
                "skeptic_score": skeptic_score,
                "score_margin": score_margin,
                "num_searches": num_searches,
                "failed_searches": failed_searches,
                "parse_success": parse_success,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "actual_total_tokens": total_tokens,
                "estimated_total_tokens": estimated_tokens,
                "total_tokens": resolved_tokens,
                "token_source": token_source,
                "duration": duration
            }
            
            existing_by_id[claim_id] = result_entry
            results = sorted(existing_by_id.values(), key=lambda x: x.get("id", 0))
            save_checkpoint(results, results_file)
            
            print(
                "  -> Winner: "
                f"{winner_role_normalized} | Accuracy: {alignment} | Turns: {turn_count} "
                f"(rounds={rounds_completed}) "
                f"| Tokens: {resolved_tokens} ({token_source}) | Time: {duration}s\n"
            )
            
        except Exception as e:
            traceback.print_exc()
            print(f"  -> Error executing graph for claim '{claim}': {e}\n")
            existing_by_id[claim_id] = {
                "id": claim_id,
                "claim": claim,
                "ground_truth": ground_truth,
                "expected_winner": expected_winner,
                "topic": topic,
                "skeptic_temp": skeptic_temp,
                "turn_bias": turn_bias,
                "winner_role": "ERROR",
                "accuracy": "Error",
                "expected_alignment": "Error",
                "turn_count": -1,
                "rounds_completed": 0,
                "finality_score": 0,
                "proposer_score": 0,
                "skeptic_score": 0,
                "score_margin": 0,
                "num_searches": 0,
                "failed_searches": 0,
                "parse_success": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "actual_total_tokens": 0,
                "estimated_total_tokens": 0,
                "total_tokens": 0,
                "token_source": "error",
                "duration": round(time.time() - start_time, 2),
                "error": str(e)
            }
            results = sorted(existing_by_id.values(), key=lambda x: x.get("id", 0))
            save_checkpoint(results, results_file)
            
    print("\nEvaluation complete. Proceeding to report generation...")
    # Once finished, generate the visualization and markdown report.
    from evaluation.generate_visuals import generate_report
    generate_report(results_file=results_file)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation claims for the LangGraph debate system.")
    parser.add_argument("--max-claims", type=int, default=None, help="Run only the first N filtered claims.")
    parser.add_argument("--start-id", type=int, default=None, help="Run claims with id >= start-id.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing results and start fresh.")
    parser.add_argument("--no-skip-done", action="store_true", help="Do not skip claim IDs already present in output file.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: evaluation/evaluation_results.json).")
    parser.add_argument(
        "--claim-ids",
        type=str,
        default=None,
        help="Comma-separated claim IDs to run, e.g. 10,12",
    )
    parser.add_argument(
        "--rerun-fails",
        action="store_true",
        help="Re-run only failed True/False claims from existing results file.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    parsed_claim_ids = None
    if args.claim_ids:
        parsed_claim_ids = [int(x.strip()) for x in args.claim_ids.split(",") if x.strip()]

    run_evaluations(
        max_claims=args.max_claims,
        start_id=args.start_id,
        overwrite=args.overwrite,
        skip_done=not args.no_skip_done,
        output_file=args.output,
        claim_ids=parsed_claim_ids,
        rerun_fails=args.rerun_fails,
    )