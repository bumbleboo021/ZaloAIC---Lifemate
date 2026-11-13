"""Frozen LifeMate competition entry point."""

import argparse
import os
from pathlib import Path
from time import perf_counter

from lifemate_baseline import Models, load_dataset, write_predictions

ROOT = Path("/data")
RESULT = Path("/result")
MODEL_ROOT = Path(__file__).resolve().parent / "saved_models"
OPTIONS = {"reuse_reads": True, "native_tool_format": True, "request_last": True}


def load_models(*, model_root=MODEL_ROOT, adapter_path=None):
    if adapter_path is None:
        adapter_path = os.environ.get("LIFEMATE_ADAPTER") or None
    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing LoRA weights in {adapter_path}")
        os.environ["USE_TF"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = perf_counter()
    models = Models(str(model_root / "asr"), str(model_root / "agent"), offline=True)
    models.prediction_options = OPTIONS.copy()
    if adapter_path is not None:
        from peft import PeftModel

        models.agent = PeftModel.from_pretrained(
            models.agent, adapter_path, local_files_only=True,
        ).merge_and_unload(safe_merge=True).eval()
        models.agent.config.use_cache = True
        models.prediction_options["chronological_history"] = True
    print(f"Model loading took {perf_counter() - started:.1f}s", flush=True)
    return models


def load_input(split=None, *, limit=None):
    if split is None:
        available = [name for name in ("private_test", "public_test")
                     if (ROOT / name / f"{name}.jsonl").is_file()]
        if not available:
            raise FileNotFoundError("Mount an official test split under /data.")
        split = available[0]
    return load_dataset(ROOT, split, limit=limit)


def predict(models, dataset, output):
    started = perf_counter()
    options = getattr(models, 'prediction_options', OPTIONS)
    write_predictions(dataset, output, models, include_time=True, **options)
    print(f"Prediction took {perf_counter() - started:.1f}s. Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("private_test", "public_test"))
    parser.add_argument("--limit", type=int, help="Optional smoke-test sample count")
    parser.add_argument("--adapter", type=Path, help="Use a local LoRA adapter with chronological history")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    dataset = load_input(args.split, limit=args.limit)
    predict(load_models(adapter_path=args.adapter), dataset, RESULT / "submission.jsonl")
