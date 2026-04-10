import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")
np.random.seed(42)

TRUTH_PALETTE = {
    "False": "#3A86FF",
    "True": "#2A9D8F",
    "Ambiguous": "#F4A261",
}

WINNER_PALETTE = {
    "Proposer": "#2A9D8F",
    "Skeptic": "#E76F51",
    "Unknown": "#6C757D",
}

TOKEN_SOURCE_PALETTE = {
    "provider_metadata": "#118AB2",
    "estimated_from_text": "#EF476F",
    "unknown_legacy": "#8D99AE",
    "error": "#D00000",
}


def _place_legend_outside(ax, title: str | None = None):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        title=title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
    )


def _save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def _accuracy_by_truth_chart(df_valid: pd.DataFrame, output_dir: Path):
    df_truth = df_valid[df_valid["ground_truth"].isin(["True", "False"])].copy()
    if df_truth.empty:
        return

    acc = (
        df_truth.groupby("ground_truth")["expected_alignment"]
        .apply(lambda s: (s == "Pass").mean() * 100)
        .reset_index(name="accuracy_pct")
    )

    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(
        data=acc,
        x="ground_truth",
        y="accuracy_pct",
        hue="ground_truth",
        palette=TRUTH_PALETTE,
        legend=False,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=10)
    plt.ylim(0, 100)
    plt.title("Accuracy by Ground-Truth Type")
    plt.xlabel("Ground Truth")
    plt.ylabel("Accuracy (%)")
    _save_fig(output_dir / "chart_accuracy_by_truth.png")


def _winner_distribution_chart(df_valid: pd.DataFrame, output_dir: Path):
    counts = df_valid["winner_role"].value_counts().reset_index()
    counts.columns = ["winner_role", "count"]

    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(
        data=counts,
        x="winner_role",
        y="count",
        hue="winner_role",
        palette=WINNER_PALETTE,
        legend=False,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=10)
    plt.title("Winner Distribution Across Claims")
    plt.xlabel("Winner Role")
    plt.ylabel("Count")
    _save_fig(output_dir / "chart_winner_distribution.png")


def _duration_by_claim_chart(df_valid: pd.DataFrame, output_dir: Path):
    order = df_valid.sort_values("id", ascending=True)["id"]

    plt.figure(figsize=(11.5, 5.6))
    ax = sns.barplot(
        data=df_valid,
        x="id",
        y="duration",
        hue="ground_truth",
        order=order,
        palette=TRUTH_PALETTE,
    )
    _place_legend_outside(ax, title="Ground Truth")
    plt.title("Runtime per Claim (Seconds, by Claim ID)")
    plt.xlabel("Claim ID")
    plt.ylabel("Duration (s)")
    _save_fig(output_dir / "chart_duration_by_claim.png")


def _turns_by_truth_chart(df_valid: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(10, 5.2))
    ax = sns.violinplot(
        data=df_valid,
        x="ground_truth",
        y="rounds_completed",
        hue="ground_truth",
        palette=TRUTH_PALETTE,
        inner=None,
        cut=0,
        legend=False,
    )
    sns.stripplot(
        data=df_valid,
        x="ground_truth",
        y="rounds_completed",
        color="#1f2937",
        alpha=0.75,
        jitter=0.12,
        size=6,
    )
    sns.pointplot(
        data=df_valid,
        x="ground_truth",
        y="rounds_completed",
        estimator="mean",
        errorbar=None,
        color="#111827",
        markers="D",
        linestyles="",
    )
    plt.title("Turn Count Distribution by Claim Type")
    plt.xlabel("Ground Truth")
    plt.ylabel("Completed Rounds")
    plt.yticks(sorted(df_valid["rounds_completed"].dropna().unique()))
    _save_fig(output_dir / "chart_turns_by_truth.png")


def _search_quality_chart(df_valid: pd.DataFrame, output_dir: Path):
    melted = df_valid[["id", "num_searches", "failed_searches"]].melt(
        id_vars=["id"],
        value_vars=["num_searches", "failed_searches"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(11.5, 5.6))
    ax = sns.barplot(
        data=melted,
        x="id",
        y="value",
        hue="metric",
        palette={"num_searches": "#1D3557", "failed_searches": "#E63946"},
    )
    _place_legend_outside(ax, title="Metric")
    plt.title("Search Usage and Failure Counts per Claim")
    plt.xlabel("Claim ID")
    plt.ylabel("Count")
    _save_fig(output_dir / "chart_search_quality.png")


def _score_margin_chart(df_valid: pd.DataFrame, output_dir: Path):
    chart_df = df_valid.copy()
    chart_df["margin_side"] = chart_df["score_margin"].apply(
        lambda v: "Proposer Leading" if v > 0 else ("Skeptic Leading" if v < 0 else "Tie")
    )

    plt.figure(figsize=(11.5, 5.6))
    ax = sns.barplot(
        data=chart_df,
        x="id",
        y="score_margin",
        hue="margin_side",
        palette={
            "Proposer Leading": "#2A9D8F",
            "Skeptic Leading": "#E63946",
            "Tie": "#6C757D",
        },
        dodge=False,
    )
    _place_legend_outside(ax, title="Margin Side")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Judge Score Margin (Positive = Proposer, Negative = Skeptic)")
    plt.xlabel("Claim ID")
    plt.ylabel("Score Margin")
    _save_fig(output_dir / "chart_score_margin.png")


def _token_usage_chart(df_valid: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(11.5, 5.6))
    ax = sns.barplot(
        data=df_valid,
        x="id",
        y="total_tokens",
        hue="token_source",
        palette=TOKEN_SOURCE_PALETTE,
    )
    _place_legend_outside(ax, title="Token Source")
    plt.title("Token Usage per Claim")
    plt.xlabel("Claim ID")
    plt.ylabel("Tokens")
    _save_fig(output_dir / "chart_token_usage.png")


def _finality_distribution_chart(df_valid: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(8, 4.8))
    sns.histplot(df_valid["finality_score"], bins=10, kde=True, color="#2a9d8f")
    plt.title("Finality Score Distribution")
    plt.xlabel("Finality Score")
    plt.ylabel("Frequency")
    _save_fig(output_dir / "chart_finality_distribution.png")


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def _rounds_by_claim_chart(df_valid: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(11.5, 5.2))
    ax = sns.barplot(
        data=df_valid,
        x="id",
        y="rounds_completed",
        hue="ground_truth",
        palette=TRUTH_PALETTE,
        dodge=False,
    )
    _place_legend_outside(ax, title="Ground Truth")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=9)
    plt.title("Completed Debate Rounds per Claim")
    plt.xlabel("Claim ID")
    plt.ylabel("Completed Rounds")
    _save_fig(output_dir / "chart_rounds_by_claim.png")


def generate_report(results_file: str | Path | None = None):
    print("Generating expanded visual report...")

    current_dir = Path(__file__).resolve().parent
    if results_file is None:
        resolved_results_file = current_dir / "evaluation_results.json"
    else:
        resolved_results_file = Path(results_file)
        if not resolved_results_file.is_absolute():
            resolved_results_file = (Path.cwd() / resolved_results_file).resolve()

    report_file = current_dir / "evaluation_report.md"

    if not resolved_results_file.exists():
        print("No evaluation_results.json found. Run the evaluators first.")
        return

    with open(resolved_results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("Evaluation results are empty.")
        return

    df = pd.DataFrame(results)

    # Backward-compat columns for old result files
    truth_to_expected = {"True": "Proposer", "False": "Skeptic"}

    if "expected_winner" not in df.columns:
        df["expected_winner"] = df["ground_truth"].map(truth_to_expected).fillna("Either")
    else:
        df["expected_winner"] = df["expected_winner"].fillna(df["ground_truth"].map(truth_to_expected)).fillna("Either")

    if "topic" not in df.columns:
        df["topic"] = "General"
    else:
        df["topic"] = df["topic"].fillna("General")

    if "score_margin" not in df.columns:
        df["score_margin"] = df.get("proposer_score", 0) - df.get("skeptic_score", 0)
    else:
        df["score_margin"] = df["score_margin"].fillna(df.get("proposer_score", 0) - df.get("skeptic_score", 0))

    if "token_source" not in df.columns:
        df["token_source"] = "unknown_legacy"
    else:
        df["token_source"] = df["token_source"].fillna("unknown_legacy")

    if "actual_total_tokens" not in df.columns:
        df["actual_total_tokens"] = df.get("total_tokens", 0)
    else:
        df["actual_total_tokens"] = df["actual_total_tokens"].fillna(df.get("total_tokens", 0))

    if "estimated_total_tokens" not in df.columns:
        df["estimated_total_tokens"] = 0
    else:
        df["estimated_total_tokens"] = df["estimated_total_tokens"].fillna(0)

    if "rounds_completed" not in df.columns:
        df["rounds_completed"] = df.get("turn_count", 0)
    else:
        df["rounds_completed"] = df["rounds_completed"].fillna(df.get("turn_count", 0))

    def _legacy_align(row):
        ew = row.get("expected_winner", "Either")
        wr = row.get("winner_role", "Unknown")
        if ew in ("Proposer", "Skeptic"):
            return "Pass" if ew == wr else "Fail"
        return "N/A"

    if "expected_alignment" not in df.columns:
        df["expected_alignment"] = df.apply(_legacy_align, axis=1)
    else:
        df["expected_alignment"] = df["expected_alignment"].fillna(df.apply(_legacy_align, axis=1))

    # Legacy runs often have token source unset and all totals as zero.
    # Reclassify these rows to avoid misleading "provider metadata" reporting.
    zero_token_legacy = (df["actual_total_tokens"].fillna(0) == 0) & (df["total_tokens"].fillna(0) == 0)
    df.loc[zero_token_legacy & (df["token_source"] == "provider_metadata"), "token_source"] = "unknown_legacy"

    df_valid = df[df["winner_role"] != "ERROR"].copy()

    if df_valid.empty:
        report_file.write_text("# Evaluation Report\n\nAll evaluations failed.", encoding="utf-8")
        print("All evaluations failed.")
        return

    # Generate charts
    _accuracy_by_truth_chart(df_valid, current_dir)
    _winner_distribution_chart(df_valid, current_dir)
    _duration_by_claim_chart(df_valid, current_dir)
    _turns_by_truth_chart(df_valid, current_dir)
    _search_quality_chart(df_valid, current_dir)
    _score_margin_chart(df_valid, current_dir)
    _token_usage_chart(df_valid, current_dir)
    _finality_distribution_chart(df_valid, current_dir)
    _rounds_by_claim_chart(df_valid, current_dir)

    # KPI calculations
    evaluatable = df_valid[df_valid["expected_winner"].isin(["Proposer", "Skeptic"])].copy()
    evaluatable_count = len(evaluatable)
    pass_count = int((evaluatable["expected_alignment"] == "Pass").sum()) if evaluatable_count else 0
    fail_count = int((evaluatable["expected_alignment"] == "Fail").sum()) if evaluatable_count else 0

    overall_accuracy = (pass_count / evaluatable_count * 100) if evaluatable_count else 0.0
    parse_success_rate = float(df_valid["parse_success"].mean() * 100)
    avg_turns = float(df_valid["rounds_completed"].mean())
    avg_duration = float(df_valid["duration"].mean())
    avg_finality = float(df_valid["finality_score"].mean())
    avg_searches = float(df_valid["num_searches"].mean())
    failed_searches = int(df_valid["failed_searches"].sum())
    avg_tokens = float(df_valid["total_tokens"].mean())
    metadata_token_coverage = float((df_valid["actual_total_tokens"] > 0).mean() * 100)

    # Claim-level issues
    wrong_claims = evaluatable[evaluatable["expected_alignment"] == "Fail"][
        ["id", "claim", "ground_truth", "expected_winner", "winner_role", "score_margin"]
    ]

    token_source_counts = df_valid["token_source"].value_counts().to_dict()

    report = []
    report.append("# Multi-Agent Fact-Check Debate: Evaluation Report")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Results Source: {resolved_results_file.name}")
    report.append("")
    report.append(f"Total Claims Run: {len(df_valid)}")
    report.append(f"Evaluatable Claims (True/False only): {evaluatable_count}")
    report.append("")

    report.append("## Metric 1: Verdict Quality")
    report.append("Measures whether the winner matches expected winner by claim truth label.")
    report.append("")
    report.append("| KPI | Value |")
    report.append("|---|---:|")
    report.append(f"| Overall accuracy | {_format_pct(overall_accuracy)} ({pass_count}/{evaluatable_count}) |")
    report.append(f"| Correct verdicts | {pass_count} |")
    report.append(f"| Incorrect verdicts | {fail_count} |")
    report.append("")
    report.append("![Accuracy by Truth](chart_accuracy_by_truth.png)")
    report.append("")
    report.append("![Winner Distribution](chart_winner_distribution.png)")
    report.append("")

    if wrong_claims.empty:
        report.append("No incorrect verdicts in evaluatable claims.")
    else:
        report.append("### Incorrect Verdicts to Review")
        report.append("| ID | Claim | Ground Truth | Expected | Actual | Score Margin (P-S) |")
        report.append("|---:|---|---|---|---|---:|")
        for _, row in wrong_claims.iterrows():
            report.append(
                f"| {int(row['id'])} | {row['claim']} | {row['ground_truth']} | {row['expected_winner']} | {row['winner_role']} | {row['score_margin']} |"
            )
    report.append("")

    report.append("## Metric 2: Debate Efficiency")
    report.append("Measures how quickly and decisively the debate converges.")
    report.append("")
    report.append("| KPI | Value |")
    report.append("|---|---:|")
    report.append(f"| Average turns | {avg_turns:.2f} |")
    report.append(f"| Average runtime (seconds) | {avg_duration:.2f} |")
    report.append(f"| Average finality score | {avg_finality:.2f} |")
    report.append("")
    report.append("![Runtime by Claim](chart_duration_by_claim.png)")
    report.append("")
    report.append("![Turns by Truth](chart_turns_by_truth.png)")
    report.append("")
    report.append("![Rounds by Claim](chart_rounds_by_claim.png)")
    report.append("")
    report.append("![Finality Distribution](chart_finality_distribution.png)")
    report.append("")

    report.append("## Metric 3: Retrieval Effectiveness")
    report.append("Tracks how often search is used and how often it fails to return context.")
    report.append("")
    report.append("| KPI | Value |")
    report.append("|---|---:|")
    report.append(f"| Average searches per claim | {avg_searches:.2f} |")
    report.append(f"| Total failed searches | {failed_searches} |")
    report.append("")
    report.append("![Search Quality](chart_search_quality.png)")
    report.append("")

    report.append("## Metric 4: Judge Reliability")
    report.append("Evaluates parsing stability and score separation behavior.")
    report.append("")
    report.append("| KPI | Value |")
    report.append("|---|---:|")
    report.append(f"| Parse success rate | {_format_pct(parse_success_rate)} |")
    report.append("")
    report.append("![Score Margin](chart_score_margin.png)")
    report.append("")

    report.append("## Metric 5: Token and Cost Profile")
    report.append("Reports token usage and source reliability.")
    report.append("")
    report.append("| KPI | Value |")
    report.append("|---|---:|")
    report.append(f"| Average total tokens per claim | {avg_tokens:.2f} |")
    report.append(f"| Claims with provider token metadata | {_format_pct(metadata_token_coverage)} |")
    report.append(f"| Token source counts | {token_source_counts} |")
    report.append("")
    report.append("![Token Usage](chart_token_usage.png)")
    report.append("")

    report.append("## Full Claim Log")
    report.append(
        "| ID | Topic | Ground Truth | Expected | Winner | Verdict | Rounds | Finality | Searches | Failed Searches | Tokens | Token Source | Parse | Duration (s) |"
    )
    report.append(
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|"
    )

    for _, row in df.sort_values("id").iterrows():
        report.append(
            f"| {int(row['id'])} | {row.get('topic', 'General')} | {row['ground_truth']} | {row.get('expected_winner', 'Either')} | "
            f"{row['winner_role']} | {row.get('expected_alignment', row.get('accuracy', 'N/A'))} | {row['rounds_completed']} | "
            f"{row['finality_score']} | {row['num_searches']} | {row['failed_searches']} | {row['total_tokens']} | "
            f"{row.get('token_source', 'provider_metadata')} | {'Yes' if bool(row['parse_success']) else 'No'} | {row['duration']:.2f} |"
        )

    report_file.write_text("\n".join(report), encoding="utf-8")
    print(f"Report generated: {report_file}")


if __name__ == "__main__":
    generate_report()
