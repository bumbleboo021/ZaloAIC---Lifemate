"""Measure the latest-request transcript effect across fixed TRAIN development."""

import argparse
import json
import os
from pathlib import Path

from collections import Counter
from experiments.evaluate_lifemate_asr import paired
from lifemate_eval import calls_match, read_rows
from lifemate_baseline import load_dataset
from lifemate_experiment import selection
from lifemate_phase2 import run_variant
from predict import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-root", "adapter", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "development"), default="development")
    parser.add_argument("--audio-predictions", type=Path,
                        help="Matching real-audio predictions for the multi-call failure audit")
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir()
    ids = set(selection(args.stage))
    dataset = load_dataset(args.data_root, "train", ids=ids)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    before = {row["id"]: row["score"] for row in baseline["details"]}
    if not ids <= before.keys() or len(before) != len(baseline["details"]):
        raise ValueError("Reference audio baseline does not match fixed development IDs.")
    if before.keys() != ids:
        details = [row for row in baseline["details"] if row["id"] in ids]
        baseline = dict(baseline, details=details, score=100 * sum(before[k] for k in ids) / len(ids))
        baseline["failures"] = None
    os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers.utils import import_utils
    import_utils._torchvision_available = False
    models = load_models(model_root=args.model_root, adapter_path=args.adapter)
    candidate = run_variant("reference_request", args.data_root, args.output,
                            dataset, models, ids, None,
                            inference_options=models.prediction_options)
    after = {row["id"]: row["score"] for row in candidate["details"]}

    def summarize(rows):
        count = len(rows)
        if not count:
            return {"samples": 0}
        return {
            "samples": count,
            "audio": 100 * sum(before[row["id"]] for row in rows) / count,
            "reference": 100 * sum(after[row["id"]] for row in rows) / count,
            "wins": sum(after[row["id"]] > before[row["id"]] for row in rows),
            "losses": sum(after[row["id"]] < before[row["id"]] for row in rows),
        }

    rows = dataset[0]
    groups = {
        "empty_call": [row for row in rows if not row["label"]["tool_calls"]],
        "single_call": [row for row in rows if len(row["label"]["tool_calls"]) == 1],
        "multi_call": [row for row in rows if len(row["label"]["tool_calls"]) > 1],
        "with_history": [row for row in rows if row.get("history")],
        "without_history": [row for row in rows if not row.get("history")],
    }
    summary = {
        "diagnostic_only": True, "samples": len(ids),
        "comparison": paired(candidate, baseline),
        "slices": {name: summarize(items) for name, items in groups.items()},
    }
    if args.audio_predictions is not None:
        summary["multi_call_audit"] = audit_multi_call(
            groups["multi_call"], dataset[2], read_rows(args.audio_predictions),
            read_rows(args.output / "reference_request.jsonl"), before, after,
        )
    temporary = args.output / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "summary.json")
    print("DEVELOPMENT_TRANSCRIPT_DIAGNOSTIC_READY", flush=True)


def audit_multi_call(rows, tools, audio, reference, before, after):
    fields = {
        name: {key for key, spec in tool["parameters"].get("properties", {}).items()
               if spec.get("x-match") == "fuzzy"}
        for name, tool in tools.items()
    }

    def divergence(expected, predicted):
        for gold, actual in zip(expected, predicted):
            if gold["name"] != actual["name"]:
                return "tool_name_differs"
            if not calls_match(actual, gold, fields):
                return "arguments_differ"
        if len(predicted) < len(expected):
            return "missing_suffix"
        if len(predicted) > len(expected):
            return "extra_calls"
        return "matching_sequence_state_or_timeout"

    repaired = Counter()
    residual = Counter()
    for row in rows:
        key = row["id"]
        expected = row["label"]["tool_calls"]
        if before[key] == 0 and after[key] > 0:
            repaired[divergence(expected, audio[key]["tool_calls"])] += 1
        if after[key] == 0:
            residual[divergence(expected, reference[key]["tool_calls"])] += 1
    return {"audio_zero_to_positive": dict(repaired), "reference_zero": dict(residual),
            "method": "first ordered divergence using scorer argument matching; alternate call orders may be valid"}


if __name__ == "__main__":
    main()
