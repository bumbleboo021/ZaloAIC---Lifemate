"""Merge a fresh Qwen3.5 adapter into the pinned multimodal checkpoint."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("agent", "adapter", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    import torch
    from transformers.utils import import_utils
    import_utils._torchvision_available = False
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.agent, local_files_only=True, torch_dtype=torch.bfloat16
    )
    tuned = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    tuned.merge_and_unload(safe_merge=True).save_pretrained(args.output)
    print("QWEN35_MERGED_READY", flush=True)


if __name__ == "__main__":
    main()

