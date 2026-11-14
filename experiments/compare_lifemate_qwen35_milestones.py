"""Compare fixed LifeMate development predictions and audit the strongest checkpoint."""

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.evaluate_lifemate_asr import paired
from experiments.evaluate_lifemate_qwen35 import VllmAgent, serving_agent
from lifemate_baseline import load_dataset
from lifemate_eval import calls_match, read_rows
from lifemate_experiment import selection
from lifemate_phase2 import run_variant


def first_difference(expected, actual, fuzzy):
    for gold, predicted in zip(expected, actual):
        if gold["name"] != predicted["name"]:
            return "wrong_tool"
        if not calls_match(predicted, gold, fuzzy):
            return "wrong_arguments"
    if len(actual) < len(expected):
        return "no_calls" if not actual else "missing_suffix"
    if len(actual) > len(expected):
        return "extra_calls" if expected else "false_action"
    return "matching_calls_state_or_timeout"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "models", "asr", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18085)
    args = parser.parse_args()

    ids = set(selection("development"))
    dataset = load_dataset(args.data, "train", ids=ids)
    rows, _, tools, _ = dataset
    fuzzy = {name: {key for key, field in tool["parameters"].get("properties", {}).items()
                    if field.get("x-match") == "fuzzy"} for name, tool in tools.items()}
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    reports = {}
    for step in (150, 362, 548):
        folder = args.output / f"step{step}"
        folder.mkdir(mode=0o700)
        agent = args.models / f"merged-step{step}"
        print("DEVELOPMENT_START", step, flush=True)
        with serving_agent(agent, folder / "server.log", port=args.port) as url:
            models = VllmAgent(url, agent, args.asr)
            with (folder / "trace.private.jsonl").open("x", encoding="utf-8") as trace:
                reports[step] = run_variant("development", args.data, folder, dataset, models, ids,
                                            trace, inference_options=models.prediction_options)
        print("DEVELOPMENT_READY", step, flush=True)

    candidates = [step for step, report in reports.items() if not report["at_or_over_30s"]]
    winner = max(candidates or list(reports), key=lambda step: reports[step]["score"])
    folder = args.output / f"step{winner}"
    predictions = read_rows(folder / "development.jsonl")
    scores = {item["id"]: item["score"] for item in reports[winner]["details"]}
    failures = [row for row in rows if scores[row["id"]] < 1]
    ordered = Counter(first_difference(row["label"]["tool_calls"],
                                       predictions[row["id"]]["tool_calls"], fuzzy)
                      for row in failures)
    # Focus the transcript experiment on reproducible failures with expected calls.
    subset = sorted((row for row in failures if scores[row["id"]] == 0 and row["label"]["tool_calls"]),
                    key=lambda row: row["id"])[:48]
    diagnostic = {"samples": len(subset)}
    if len(subset) >= 2:
        agent = args.models / f"merged-step{winner}"
        with serving_agent(agent, folder / "reference-server.log", port=args.port) as url:
            models = VllmAgent(url, agent, args.asr)
            report = run_variant("reference_request", args.data, folder, (subset, *dataset[1:]),
                                 models, {row["id"] for row in subset}, None,
                                 inference_options=models.prediction_options)
        ref = read_rows(folder / "reference_request.jsonl")
        passed = {item["id"] for item in report["details"] if item["score"] > 0}
        diagnostic.update(reference_score_on_audio_zero_cases=report["score"],
                          audio_zero_to_positive=len(passed),
                          residual_first_difference=dict(Counter(
                              first_difference(row["label"]["tool_calls"], ref[row["id"]]["tool_calls"], fuzzy)
                              for row in subset if row["id"] not in passed)))

    summary = {"selected": winner, "selected_meets_latency_limit": bool(candidates),
               "comparison_to_step150": {step: paired(reports[step], reports[150]) for step in (362, 548)},
               "development": {}, "first_ordered_difference": dict(ordered),
               "reference_request_diagnostic": diagnostic}
    for step, report in reports.items():
        details = report["details"]
        multi = [item["score"] for item in details if item["gold_calls"] > 1]
        summary["development"][step] = dict({key: value for key, value in report.items() if key != "details"},
                                            multi_call_score=100 * sum(multi) / len(multi))
    temporary = args.output / "summary.partial"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "summary.json")
    print("DEVELOPMENT_COMPARISON_READY", winner, flush=True)


if __name__ == "__main__":
    main()
