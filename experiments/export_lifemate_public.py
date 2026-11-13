"""Generate and validate the LoRA public submission on a remote GPU."""

import argparse
import json
from pathlib import Path

from lifemate_baseline import load_dataset, write_predictions
from lifemate_eval import validate_public
from predict import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-root", "adapter", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path.")
    dataset = load_dataset(args.data_root, "public_test")
    models = load_models(model_root=args.model_root, adapter_path=args.adapter)
    write_predictions(
        dataset, args.output, models, include_time=False, **models.prediction_options
    )
    print(json.dumps(validate_public(args.data_root, args.output)), flush=True)


if __name__ == "__main__":
    main()
