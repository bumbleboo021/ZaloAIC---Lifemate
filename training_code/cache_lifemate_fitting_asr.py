"""Cache current-request Whisper transcripts from the LifeMate fitting split."""

import argparse
import json
import os
from pathlib import Path

from lifemate_baseline import Models, audio_file, json_text


class FittingTranscriber:
    transcribe = Models.transcribe

    def __init__(self, asr_path):
        os.environ["USE_TF"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import pipeline

        self.asr = pipeline(
            "automatic-speech-recognition", model=str(asr_path), device=0,
            torch_dtype=torch.float16, model_kwargs={"local_files_only": True},
        )


def fitting_rows(data_root, validation):
    dev = set((validation / "development.txt").read_text(encoding="utf-8").split())
    holdout = set((validation / "holdout.txt").read_text(encoding="utf-8").split())
    if len(dev) != 232 or len(holdout) != 155 or dev & holdout:
        raise ValueError("Expected the fixed development and holdout splits.")
    with (data_root / "train/train.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = {row["id"] for row in rows}
    excluded = dev | holdout
    if len(rows) != 1545 or len(ids) != len(rows) or not excluded <= ids:
        raise ValueError("Expected the official 1,545 unique TRAIN IDs.")
    return [row for row in rows if row["id"] not in excluded]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "asr", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=Path(__file__).resolve().parents[1] / "validation")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.data_root.resolve()):
        parser.error("Save the restricted cache outside the kit.")
    os.umask(0o077)
    rows = fitting_rows(args.data_root, args.validation)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists():
        parser.error("Choose a fresh output path.")
    model = FittingTranscriber(args.asr)
    with partial.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            text = model.transcribe(audio_file(args.data_root / "train", row["audio"]), 25)
            stream.write(json_text({"id": row["id"], "text": text}) + "\n")
            if index == 1 or index % 50 == 0:
                stream.flush()
                print(f"{index}/{len(rows)} fitting requests transcribed", flush=True)
    partial.replace(args.output)
    print(f"FITTING_ASR_READY {len(rows)}", flush=True)


if __name__ == "__main__":
    main()

