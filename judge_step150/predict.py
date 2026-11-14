"""Offline Qwen3.5 step-150 judge entry point."""

import argparse
import os
import random
from pathlib import Path
from time import perf_counter

from experiments.evaluate_lifemate_qwen35 import VllmAgent, serving_agent
from lifemate_baseline import load_dataset, write_predictions

CODE = Path(__file__).resolve().parent
RESULT = Path("/result")


def seed_everything():
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_input(data_root=Path("/data"), *, limit=None):
    for split in ("private_test", "public_test"):
        if (data_root / split / f"{split}.jsonl").is_file():
            return load_dataset(data_root, split, limit=limit)
    raise FileNotFoundError("Mount the official test dataset under /data.")


def load_models(stack, result_dir=RESULT, *, model_root=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    root = model_root or Path(os.environ.get("LIFEMATE_MODEL_ROOT", CODE / "saved_models"))
    gpu_fraction = float(os.environ.get("LIFEMATE_VLLM_GPU_FRACTION", "0.60"))
    port = int(os.environ.get("LIFEMATE_VLLM_PORT", "18083"))
    url = stack.enter_context(serving_agent(root / "agent", result_dir / "vllm-server.log",
                                            port=port, gpu_memory_utilization=gpu_fraction))
    models = VllmAgent(url, root / "agent", root / "asr")
    print(f"Model loading and warm-up took {perf_counter() - started:.2f}s", flush=True)
    return models


def predict(models, dataset, output):
    started = perf_counter()
    write_predictions(dataset, output, models, include_time=True, **models.prediction_options)
    print(f"Prediction took {perf_counter() - started:.2f}s. Saved {output}", flush=True)


def main():
    from contextlib import ExitStack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    seed_everything()
    with ExitStack() as stack:
        models = load_models(stack)
        predict(models, load_input(limit=args.limit), RESULT / "submission.jsonl")


if __name__ == "__main__":
    main()
