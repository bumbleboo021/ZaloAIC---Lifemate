"""Compare one bill-payment prompt change with the saved step-362 LoRA run."""

import argparse
import json
from pathlib import Path
from statistics import mean

from lifemate_baseline import load_dataset
from lifemate_experiment import selection
from lifemate_phase2 import run_variant
from predict import load_models, predict


def compare(stage, data, output, reference, models):
    ids = set(selection(stage))
    dataset = load_dataset(data, "train", ids=ids)
    prior = json.loads((reference / "chronological_score.json").read_text())
    old = {row["id"]: row for row in prior["details"]}
    if old.keys() != ids:
        raise ValueError("Reference report IDs differ from the fixed split.")
    if stage == "pilot":
        replay = output / "shared_path_replay.jsonl"
        predict(models, dataset, replay)

        def read_predictions(path):
            rows = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row.pop("time", None)
                rows[row["id"]] = row
            return rows

        if read_predictions(replay) != read_predictions(reference / "chronological.jsonl"):
            raise ValueError("Shared prediction path differs from recorded LoRA pilot.")
        print("SHARED_PATH_PARITY: 48/48", flush=True)

    report = run_variant(
        "bill_payment", data, output, dataset, models, ids, None,
        inference_options=dict(models.prediction_options, bill_payment_guidance=True),
    )
    deltas = [row["score"] - old[row["id"]]["score"] for row in report["details"]]
    empty_false_actions = sum(
        row["gold_calls"] == 0 and row["predicted_calls"] > 0 for row in report["details"]
    )
    baseline_empty_false_actions = sum(
        row["gold_calls"] == 0 and row["predicted_calls"] > 0 for row in old.values()
    )
    result = {
        "baseline": prior["score"], "candidate": report["score"],
        "delta_points": 100 * mean(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "median_ms": report["median_ms"], "p95_ms": report["p95_ms"],
        "max_time_ms": report["max_time_ms"], "at_or_over_30s": report["at_or_over_30s"],
        "empty_false_actions": empty_false_actions,
        "baseline_empty_false_actions": baseline_empty_false_actions,
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(stage, result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data-root", "model-root", "adapter", "reference", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    models = load_models(model_root=args.model_root, adapter_path=args.adapter)
    pilot = compare("pilot", args.data_root, args.output, args.reference, models)
    if (
        pilot["delta_points"] > 0 and not pilot["at_or_over_30s"]
        and pilot["empty_false_actions"] <= pilot["baseline_empty_false_actions"]
    ):
        development = args.output / "development"
        development.mkdir()
        compare("development", args.data_root, development, args.reference / "development", models)
    print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
