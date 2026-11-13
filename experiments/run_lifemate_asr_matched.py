"""Run a matched LoRA comparison using screened current-request ASR variants."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from experiments.evaluate_lifemate_asr import paired
from lifemate_baseline import json_text, load_dataset
from lifemate_experiment import selection
from lifemate_phase2 import run_variant
from predict import load_models
from training_code.screen_lifemate_asr import read_rows

REQUEST_PREFIX = "Latest Vietnamese request. Handle this turn only:\n"


def make_fitting_pairs(fitting, accepted, output, excluded):
    examples = [json.loads(line) for line in fitting.read_text(encoding="utf-8").splitlines()]
    fitting_ids = {row["id"] for row in examples}
    if len(fitting_ids) != 1158 or fitting_ids & excluded or not accepted.keys() <= fitting_ids:
        raise ValueError("Unexpected fitting partition.")
    background = random.Random(42).sample(sorted(fitting_ids - accepted.keys()), 128)
    selected = set(background) | accepted.keys()
    total = altered = 0
    with (output / "control.private.jsonl").open("x", encoding="utf-8") as control, (
        output / "whisper.private.jsonl"
    ).open("x", encoding="utf-8") as whisper:
        for row in examples:
            sample_id = row["id"]
            if sample_id not in selected:
                continue
            control.write(json_text(row) + "\n")
            total += 1
            if sample_id in accepted:
                row = deepcopy(row)
                turn = next(
                    message for message in reversed(row["prompt"])
                    if message["role"] == "user"
                )
                if not turn["content"].startswith(REQUEST_PREFIX):
                    raise ValueError("Unexpected request marker in fitting prompts.")
                turn["content"] = REQUEST_PREFIX + json_text(accepted[sample_id]["text"])
                altered += 1
            whisper.write(json_text(row) + "\n")
    return dict(reviewed_variants=len(accepted), original_background=len(background),
                targets_per_arm=total, altered_targets=altered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "screen", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.environ["USE_TF"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.umask(0o077)
    root = args.root
    out = args.output
    out.mkdir()
    for _ in range(90):
        if (args.screen / "summary.json").is_file():
            break
        time.sleep(2)
    else:
        raise RuntimeError("Screening did not finish.")
    accepted = read_rows(args.screen / "accepted.private.jsonl")
    if not accepted:
        raise RuntimeError("No screened variants available.")
    excluded = set()
    for split in ("development", "holdout"):
        excluded.update((root / "validation" / f"{split}.txt").read_text().split())
    summary = {"fitting": make_fitting_pairs(root / "fitting.jsonl", accepted, out, excluded)}
    print(summary, flush=True)
    train = "training_code.train_lifemate_lora"
    for arm in ("control", "whisper"):
        with (out / f"{arm}_training.log").open("x") as log:
            subprocess.run([
                sys.executable, "-u", "-m", train, "--fitting", str(out / f"{arm}.private.jsonl"),
                "--agent", str(root / "saved_models/agent"), "--output", str(out / arm),
                "--init-adapter", str(args.reference / "adapter"), "--max-minutes", "35",
            ], stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((out / arm / "training_metrics.json").read_text())
        if metrics["epoch"] < 0.999:
            raise RuntimeError(f"{arm} did not finish the matched epoch.")
        final_log = (out / f"{arm}_training.log").read_text()
        if f"Encoded {summary['fitting']['targets_per_arm']} targets, skipped 0 " not in final_log:
            raise RuntimeError(f"{arm} dropped fitting targets.")
        print(arm, metrics, flush=True)

    import gc
    import torch
    from transformers.utils import import_utils
    import_utils._torchvision_available = False

    data = root / "train+public_test"
    ids = set(selection("pilot"))
    dataset = load_dataset(data, "train", ids=ids)
    baseline = json.loads((args.reference / "evaluation/chronological_score.json").read_text())
    reports = {}
    for arm in ("control", "whisper"):
        models = load_models(model_root=root / "saved_models", adapter_path=out / arm / "adapter")
        reports[arm] = run_variant(arm, data, out, dataset, models, ids, None,
                                   inference_options=models.prediction_options)
        del models
        gc.collect()
        torch.cuda.empty_cache()

    summary.update(
        control_vs_step362=paired(reports["control"], baseline),
        whisper_vs_step362=paired(reports["whisper"], baseline),
        whisper_vs_control=paired(reports["whisper"], reports["control"]),
    )
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    if (summary["whisper_vs_step362"]["delta_points"] > 0
            and summary["whisper_vs_control"]["delta_points"] > 0
            and not reports["whisper"]["at_or_over_30s"]):
        ids = set(selection("development"))
        dataset = load_dataset(data, "train", ids=ids)
        models = load_models(model_root=root / "saved_models", adapter_path=out / "whisper/adapter")
        dev = out / "development"
        dev.mkdir()
        report = run_variant("whisper", data, dev, dataset, models, ids, None,
                             inference_options=models.prediction_options)
        prior = json.loads((args.reference / "evaluation/development/chronological_score.json").read_text())
        summary["development"] = paired(report, prior)
        path.write_text(json.dumps(summary, indent=2))
    print("MATCHED_ASR_READY", flush=True)


if __name__ == "__main__":
    main()

