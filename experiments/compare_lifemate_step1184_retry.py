"""Run one conservative first-turn retry against the step-1184 public export."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import monotonic

from experiments.evaluate_lifemate_qwen35 import VllmAgent, serving_agent
from lifemate_baseline import infer, json_text, load_dataset, parse_action
from lifemate_eval import validate_public
from lifemate_experiment import file_hash, write_json

REVIEW = (
    "\nThe first pass ended without a tool call. Recheck only the latest Vietnamese user request. "
    "If it explicitly asks for an action supported by the supplied tools and the first step "
    "can be determined from the request or context, make that one tool call. A lookup is "
    "appropriate if it is needed for that explicit action. Otherwise finish, including greetings, "
    "general questions, unsupported actions, hypothetical requests and unclear intent. "
    "Do not invent values, infer permission for an action, repeat historical actions, or ask for clarification on this retry."
)


class RetryAgent(VllmAgent):
    def next_action(self, messages, seconds_left, *, native_tools=None, thinking=False, observe=None):
        first = self.first
        self.first = False
        start = monotonic()
        raw = super().next_action(messages, seconds_left, native_tools=native_tools,
                                  thinking=thinking, observe=observe)
        if (not first or raw is None or not native_tools
                or parse_action(raw, True, thinking) != {"finish": True}):
            return raw
        remaining = seconds_left - (monotonic() - start)
        if remaining < 5:
            self.counts["budget_skip"] += 1
            return raw
        self.counts["retry_attempts"] += 1
        system, *rest = messages
        review = [dict(system, content=system["content"] + REVIEW), *rest]
        second = super().next_action(review, min(remaining - 1, 5),
                                     native_tools=native_tools, thinking=thinking, observe=observe)
        action = parse_action(second, True, thinking) if second is not None else None
        allowed = {tool["function"]["name"] for tool in native_tools} - {"sys.ask_missing_info"}
        if action and action.get("name") in allowed and isinstance(action.get("arguments"), dict):
            self.counts["retry_proposed_tool"] += 1
            return second
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "agent", "asr", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    rows, split_dir, tools, factory = load_dataset(args.data, "public_test")
    with args.baseline.open(encoding="utf-8") as source:
        control = {row["id"]: row for row in map(json.loads, source)}
    if len(control) != 634 or {row["id"] for row in rows} != control.keys():
        raise ValueError("Baseline IDs must match all 634 public requests")
    counts = Counter()
    changed = []
    temporary = args.output / "submission.partial"
    maximum = 0
    with serving_agent(args.agent, args.output / "server.log", port=18089) as url:
        model = RetryAgent(url, args.agent, args.asr)
        model.counts = counts
        with temporary.open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows, 1):
                model.first = True
                attempts = counts["retry_attempts"]
                result, status, milliseconds = infer(
                    row, split_dir, tools, factory, model, **model.prediction_options)
                maximum = max(maximum, milliseconds)
                baseline = control[row["id"]]
                was_retried = attempts != counts["retry_attempts"]
                if result != {key: baseline[key] for key in ("id", "tool_calls", "state_hash")}:
                    changed.append(row["id"])
                    counts["changed_with_retry" if was_retried else "changed_without_retry"] += 1
                    if not baseline["tool_calls"] and result["tool_calls"]:
                        counts["empty_to_calls"] += 1
                    elif baseline["tool_calls"] and not result["tool_calls"]:
                        counts["calls_to_empty"] += 1
                    elif baseline["tool_calls"]:
                        counts["nonempty_control_changed"] += 1
                stream.write(json_text(result) + "\n")
                if index % 20 == 0 or index == len(rows):
                    print("RETRY_PROGRESS", index, len(rows), status, maximum, dict(counts), flush=True)
    summary = dict(validate_public(args.data, temporary), counts=dict(counts), max_ms=maximum,
                   baseline_sha256=file_hash(args.baseline), submission_sha256=file_hash(temporary))
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "changed_ids.private.json", changed)
    if maximum >= 30000 or counts["changed_without_retry"] or counts["nonempty_control_changed"]:
        raise RuntimeError("Retry candidate failed the timing or baseline parity check; see summary.json")
    submission = args.output / "lifemate_submission_step1184_retry.jsonl"
    temporary.replace(submission)
    print("RETRY_PUBLIC_READY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
