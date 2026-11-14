"""Test whole-batch funding guidance with the frozen LifeMate step-548 agent."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean

from experiments.compare_lifemate_qwen35_asr import save
from experiments.evaluate_lifemate_asr import paired, read_report
from experiments.evaluate_lifemate_qwen35 import VllmAgent, serving_agent
from lifemate_baseline import load_dataset
from lifemate_eval import calls_match, read_rows
from lifemate_experiment import selection
from lifemate_phase2 import run_variant


FUNDING_GUIDANCE = (
    "\nFor a request that uses a fallback account if the primary account cannot "
    "cover all requested transfers, resolve each recipient and amount first. "
    "Use banking.get_balance observations to match the named accounts to their "
    "IDs and check the primary balance against the combined amount before any "
    "transfer. Use the primary for the full batch if it covers the total; "
    "otherwise use the requested fallback for the full batch. Follow any "
    "explicit per-transfer funding instructions instead."
)


class FundingAgent(VllmAgent):
    def next_action(self, messages, seconds_left, *, native_tools=None, thinking=False, observe=None):
        if any(tool["function"]["name"] == "banking.transfer" for tool in (native_tools or ())):
            system, *conversation = messages
            messages = [dict(system, content=system["content"] + FUNDING_GUIDANCE), *conversation]
        return super().next_action(messages, seconds_left, native_tools=native_tools,
                                   thinking=thinking, observe=observe)


def diagnostic_ids(dataset, baseline_folder, baseline):
    rows, _, tools, _ = dataset
    predictions = read_rows(baseline_folder / "reference_request.jsonl")
    fuzzy = {
        name: {key for key, field in tool["parameters"].get("properties", {}).items()
               if field.get("x-match") == "fuzzy"}
        for name, tool in tools.items()
    }
    failures = {item["id"] for item in baseline["details"] if item["score"] == 0}
    selected = set()
    for row in rows:
        key = row["id"]
        if key not in failures:
            continue
        for expected, actual in zip(row["label"]["tool_calls"], predictions[key]["tool_calls"]):
            if calls_match(actual, expected, fuzzy):
                continue
            if (expected["name"] == actual["name"] == "banking.transfer"
                    and expected["arguments"].get("from_account_id")
                    != actual["arguments"].get("from_account_id")):
                selected.add(key)
            break
    return selected


def evaluate(name, data, output, dataset, models, ids):
    return run_variant(name, data, output, dataset, models, ids, None,
                       inference_options=models.prediction_options)


def compare(stage, args, models, baseline):
    ids = set(selection(stage))
    dataset = load_dataset(args.data, "train", ids=ids)
    result = evaluate(stage, args.data, args.output, dataset, models, ids)
    changes = paired(result, baseline)
    for report, label in ((baseline, "baseline"), (result, "candidate")):
        multi = [item["score"] for item in report["details"] if item["gold_calls"] > 1]
        changes[label + "_multi_call_score"] = 100 * mean(multi) if multi else None
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "agent", "asr", "output", "development-control", "pilot-control"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18086)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir()
    summary = {"checkpoint": 548}
    dev = read_report(args.development_control / "development_score.json")
    pilot = read_report(args.pilot_control / "greedy_score.json")
    reference = read_report(args.development_control / "reference_request_score.json")
    dataset = load_dataset(args.data, "train", ids=set(selection("development")))
    ids = diagnostic_ids(dataset, args.development_control, reference)
    with serving_agent(args.agent, args.output / "server.log", port=args.port) as url:
        models = FundingAgent(url, args.agent, args.asr)
        if ids:
            subset = ([row for row in dataset[0] if row["id"] in ids], *dataset[1:])
            report = evaluate("reference_request", args.data, args.output, subset, models, ids)
            summary["diagnostic"] = {
                "samples": len(ids), "baseline": 0.0, "candidate": report["score"],
                "recovered": sum(item["score"] > 0 for item in report["details"]),
            }
            save(args.output, summary)
        summary["pilot"] = compare("pilot", args, models, pilot)
        save(args.output, summary)
        screen = summary["pilot"]
        recovered = summary.get("diagnostic", {}).get("recovered", 0)
        if (not screen["at_or_over_30s"]
                and (screen["delta_points"] > 0 or (screen["losses"] == 0 and recovered))):
            summary["development"] = compare("development", args, models, dev)
            result = summary["development"]
            baseline_false = result["baseline_failures"].get("calls_when_empty_expected", 0)
            candidate_false = result["candidate_failures"].get("calls_when_empty_expected", 0)
            summary["outcome"] = ("development_improved" if result["delta_points"] > 0
                                  and not result["at_or_over_30s"]
                                  and candidate_false <= baseline_false else "development_gate_failed")
        else:
            summary["outcome"] = "pilot_gate_failed"
        save(args.output, summary)
    print("BANKING_FUNDING_READY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
