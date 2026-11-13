"""Measure the latest-request transcript effect across fixed TRAIN development."""

import argparse
import json
import os
from pathlib import Path

from experiments.evaluate_lifemate_asr import paired
from lifemate_baseline import load_dataset
from lifemate_experiment import selection
from lifemate_phase2 import run_variant
from predict import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-root", "adapter", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir()
    ids = set(selection("development"))
    dataset = load_dataset(args.data_root, "train", ids=ids)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    before = {row["id"]: row["score"] for row in baseline["details"]}
    if before.keys() != ids or len(before) != len(baseline["details"]):
        raise ValueError("Reference audio baseline does not match fixed development IDs.")
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
    temporary = args.output / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "summary.json")
    print("DEVELOPMENT_TRANSCRIPT_DIAGNOSTIC_READY", flush=True)


if __name__ == "__main__":
    main()

