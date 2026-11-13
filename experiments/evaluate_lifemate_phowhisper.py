"""Compare stock PhoWhisper-medium against the recorded step-362 turbo pilot."""

import argparse
import json
import os
from pathlib import Path

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-root", "pho", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    from transformers.utils import import_utils
    import_utils._torchvision_available = False
    from experiments.evaluate_lifemate_asr import paired, read_report
    from lifemate_baseline import load_dataset
    from lifemate_experiment import selection
    from lifemate_phase2 import run_variant
    from predict import load_models

    args.output.mkdir(parents=True, exist_ok=True)
    model_root = args.output / "stock-models"
    model_root.mkdir()
    (model_root / "agent").symlink_to((args.model_root / "agent").resolve(), target_is_directory=True)
    (model_root / "asr").symlink_to(args.pho.resolve(), target_is_directory=True)
    models = load_models(model_root=model_root, adapter_path=args.reference / "adapter")
    ids = set(selection("pilot"))
    dataset = load_dataset(args.data_root, "train", ids=ids)
    with (args.output / "stock.private.jsonl").open("x", encoding="utf-8") as trace:
        report = run_variant("phowhisper_stock", args.data_root, args.output,
                             dataset, models, ids, trace,
                             inference_options=models.prediction_options)
    baseline = read_report(args.reference / "evaluation/chronological_score.json")
    result = paired(report, baseline)
    (args.output / "stock_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("STOCK_PILOT_READY", result, flush=True)


if __name__ == "__main__":
    main()

