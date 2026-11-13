"""Compare completed ASR LoRAs and optionally export the development winner."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

ARMS = {"phowhisper_medium": "training", "whisper_turbo": "turbo-training"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(candidate, baseline, references):
    left = {x["id"]: x for x in candidate["details"]}
    right = {x["id"]: x for x in baseline["details"]}
    if left.keys() != right.keys() or left.keys() != references.keys():
        raise ValueError("Development coverage differs.")
    from lifemate_eval import normalize

    grouped = defaultdict(list)
    for key in sorted(left):
        transcript = " ".join(normalize(references[key]["audio_text"]).split())
        grouped[transcript].append(left[key]["score"] - right[key]["score"])
    groups = [(sum(deltas), len(deltas)) for deltas in grouped.values()]
    rng = random.Random(42)
    draws = []
    for _ in range(5000):
        draw = rng.choices(groups, k=len(groups))
        draws.append(100 * sum(x[0] for x in draw) / sum(x[1] for x in draw))
    draws.sort()
    multi = [x["score"] for x in left.values() if x["gold_calls"] > 1]
    return {
        "samples": len(left), "score": candidate["score"], "baseline": baseline["score"],
        "delta_points": candidate["score"] - baseline["score"],
        "grouped_bootstrap_95_ci": [draws[125], draws[4875]],
        "state_match_pct": candidate["state_match_pct"], "call_f1_pct": candidate["call_f1_pct"],
        "multi_call_score": 100 * sum(multi) / len(multi),
        "false_actions": candidate["failures"].get("calls_when_empty_expected", 0),
        "median_ms": candidate["median_ms"], "p95_ms": candidate["p95_ms"],
        "max_ms": candidate["max_time_ms"], "at_or_over_30s": candidate["at_or_over_30s"],
    }


def run_model(args, arm, *, public=False):
    from transformers.utils import import_utils
    import_utils._torchvision_available = False

    from lifemate_baseline import json_text, load_dataset, write_predictions
    from lifemate_eval import read_rows, validate_public
    from lifemate_experiment import file_hash, selection, write_json
    from lifemate_phase2 import run_variant
    from predict import load_models

    directory = args.output / arm
    directory.mkdir(exist_ok=True)
    root = directory / "models"
    root.mkdir(exist_ok=True)
    for name, source in {
        "asr": args.experiment / ARMS[arm] / "merged", "agent": args.agent,
    }.items():
        link = root / name
        if not link.is_symlink():
            link.symlink_to(source.resolve(strict=True), target_is_directory=True)
    models = load_models(model_root=root, adapter_path=args.reference / "adapter")
    if public:
        ids = load_dataset(args.data_root, "public_test")
        path = args.output / f"lifemate_submission_{arm}_lora.jsonl"
        timed = args.output / "public_timed.jsonl"
        write_predictions(ids, timed, models, include_time=True, **models.prediction_options)
        rows = read_rows(timed)
        times = [row.pop("time") for row in rows.values()]
        if any(value >= 30_000 for value in times):
            raise ValueError("A public sample reached the 30-second limit.")
        pending = path.with_suffix(".partial")
        with pending.open("x", encoding="utf-8") as stream:
            for row in rows.values():
                stream.write(json_text(row) + "\n")
        validation = validate_public(args.data_root, pending)
        if validation["samples"] != 634:
            raise ValueError("Expected all 634 public rows.")
        pending.replace(path)
        write_json(args.output / "public.json", {
            **validation, "file": path.name, "sha256": file_hash(path), "max_ms": max(times),
        })
        print("ASR_PUBLIC_READY", path, flush=True)
    else:
        ids = set(selection("development"))
        dataset = load_dataset(args.data_root, "train", ids=ids)
        run_variant("tuned", args.data_root, directory, dataset, models, ids, None,
                    inference_options=models.prediction_options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "experiment", "agent", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--export-public", action="store_true")
    parser.add_argument("--evaluate-arm", choices=ARMS)
    parser.add_argument("--export-arm", choices=ARMS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.evaluate_arm or args.export_arm:
        run_model(args, args.evaluate_arm or args.export_arm, public=bool(args.export_arm))
        return

    from lifemate_eval import read_rows
    from lifemate_experiment import selection, write_json

    args.output.mkdir(parents=True)
    reference_report = load_json(args.reference / "evaluation/development/chronological_score.json")
    all_rows = read_rows(args.data_root / "train/train.jsonl")
    references = {key: all_rows[key] for key in selection("development")}
    results = {"models": {}, "errors": {}}
    reports = {}

    def child(flag, arm, timeout):
        command = [sys.executable, "-u", "-m", "experiments.compare_lifemate_asr_loras"]
        for name in ("data-root", "experiment", "agent", "reference", "output"):
            command.extend((f"--{name}", str(getattr(args, name.replace("-", "_")))))
        with (args.output / f"{arm}-{flag}.log").open("x", encoding="utf-8") as log:
            subprocess.run([*command, f"--{flag}", arm], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout,
                           cwd=Path(__file__).resolve().parents[1])

    for arm, folder in ARMS.items():
        path = args.experiment / folder
        try:
            summary = load_json(path / "summary.json")
            if summary["training"]["completed_epochs"] != 1:
                raise ValueError("Expected a completed one-epoch run.")
            if not (path / "merged/config.json").is_file():
                raise ValueError("Merged model is missing.")
            report_path = path / "development/tuned_score.json"
            if not report_path.is_file():
                print("EVALUATING", arm, flush=True)
                child("evaluate-arm", arm, 1800)
                report_path = args.output / arm / "tuned_score.json"
            report = load_json(report_path)
            results["models"][arm] = compare(report, reference_report, references)
            reports[arm] = report
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            results["errors"][arm] = str(error)
        write_json(args.output / "summary.json", results)

    if len(reports) == 2:
        results["head_to_head"] = compare(
            reports["phowhisper_medium"], reports["whisper_turbo"], references
        )
    stock_path = args.experiment / "stock-development/phowhisper_stock_score.json"
    if stock_path.is_file():
        stock_report = load_json(stock_path)
        results["stock_phowhisper"] = compare(stock_report, reference_report, references)
    baseline_multi = [x["score"] for x in reference_report["details"] if x["gold_calls"] > 1]
    baseline_multi_score = 100 * sum(baseline_multi) / len(baseline_multi)
    stock_score = results.get("stock_phowhisper", {}).get("score", float("inf"))
    eligible = {
        arm: row for arm, row in results["models"].items()
        if row["grouped_bootstrap_95_ci"][0] > 0 and row["false_actions"] <= 10
        and row["multi_call_score"] >= baseline_multi_score and row["at_or_over_30s"] == 0
        and row["score"] > stock_score
    }
    results["selected"] = max(eligible, key=lambda arm: eligible[arm]["score"]) if (
        eligible and len(reports) == 2
    ) else None
    write_json(args.output / "summary.json", results)
    if args.export_public and results["selected"]:
        try:
            child("export-arm", results["selected"], 5400)
            results["public"] = load_json(args.output / "public.json")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            results["public_error"] = str(error)
        write_json(args.output / "summary.json", results)
    print("ASR_FINAL_COMPARISON_READY", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

