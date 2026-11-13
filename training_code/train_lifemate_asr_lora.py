"""Fit and evaluate one Whisper LoRA using official LifeMate fitting transcripts."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import re
from time import monotonic

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def fitting_manifest(root, output):
    from training_code.cache_lifemate_fitting_asr import fitting_rows
    from lifemate_baseline import audio_file
    import soundfile as sf

    rows = fitting_rows(root, Path(__file__).resolve().parents[1] / "validation")
    clips = []
    rejected = {"empty_reference": 0, "over_30s": 0}
    for row in rows:
        path = audio_file(root / "train", row["audio"])
        text = row["audio_text"].strip()
        duration = sf.info(path).duration
        if not text:
            rejected["empty_reference"] += 1
        elif duration > 30:
            rejected["over_30s"] += 1
        elif duration <= 0:
            raise ValueError(f"Empty audio for fitting ID {row['id']}")
        else:
            clips.append(dict(id=row["id"], audio=str(path), text=text, seconds=duration))
    if not clips:
        raise ValueError("No aligned fitting examples.")
    with (output / "fitting.private.jsonl").open("x", encoding="utf-8") as stream:
        for clip in clips:
            stream.write(json.dumps(clip, ensure_ascii=False) + "\n")
    stats = dict(fitting=len(rows), retained=len(clips), excluded=rejected,
                 hours=sum(x["seconds"] for x in clips) / 3600)
    print("MANIFEST_READY", stats, flush=True)
    return clips, stats


def train(base, clips, output, *, epochs, minutes):
    from math import gcd

    from peft import LoraConfig, PeftModel, get_peft_model
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    random.seed(42)
    torch.manual_seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base, torch_dtype=dtype, local_files_only=True
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config).train()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=1e-4
    )
    order = list(clips)
    accumulation = 8
    expected_rate = processor.feature_extractor.sampling_rate
    target_limit = model.config.max_target_positions
    deadline = monotonic() + 60 * minutes
    updates = 0
    completed_epochs = 0

    def save(label):
        path = output / label
        model.save_pretrained(path, safe_serialization=True)
        processor.save_pretrained(path)
        return path

    checkpoint = None
    for epoch in range(1, epochs + 1):
        random.shuffle(order)
        loss_total = 0.0
        seen = 0
        for start in range(0, len(order), accumulation):
            if monotonic() >= deadline:
                break
            batch = order[start : start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for clip in batch:
                samples, rate = sf.read(clip["audio"], dtype="float32", always_2d=True)
                samples = samples.mean(axis=1)
                if rate != expected_rate:
                    divisor = gcd(rate, expected_rate)
                    samples = resample_poly(samples, expected_rate // divisor, rate // divisor)
                labels = processor.tokenizer(clip["text"]).input_ids
                if labels[0] == model.config.decoder_start_token_id:
                    labels = labels[1:]
                if not labels or len(labels) > target_limit:
                    raise ValueError(f"Invalid label length for fitting ID {clip['id']}")
                inputs = processor.feature_extractor(
                    samples, sampling_rate=expected_rate, return_tensors="pt"
                ).input_features.to(device="cuda", dtype=dtype)
                labels = torch.tensor([labels], device="cuda")
                loss = model(input_features=inputs, labels=labels).loss
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite ASR loss")
                (loss / len(batch)).backward()
                loss_total += value
                seen += 1
            optimizer.step()
            updates += 1
            if updates == 1:
                save("smoke")
            if updates == 1 or updates % 10 == 0:
                print(f"ASR_TRAIN epoch={epoch} clips={seen}/{len(order)} "
                      f"updates={updates} mean_loss={loss_total / seen:.4f}", flush=True)
        if seen == len(order):
            completed_epochs += 1
        if seen:
            checkpoint = save(f"epoch-{epoch}" if seen == len(order) else f"partial-{epoch}")
        if seen != len(order):
            break
    if checkpoint is None:
        raise RuntimeError("Training timed out before the first checkpoint")
    print("TRAIN_READY", checkpoint.name, updates, completed_epochs, flush=True)
    del model, optimizer, loss, inputs, labels
    gc.collect()
    torch.cuda.empty_cache()
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base, torch_dtype=torch.float32, local_files_only=True
    )
    merged = PeftModel.from_pretrained(base_model, checkpoint, local_files_only=True)
    merged = merged.merge_and_unload(safe_merge=True).to(torch.float16)
    merged.config.use_cache = True
    merged.save_pretrained(output / "merged")
    processor.save_pretrained(output / "merged")
    return dict(checkpoint=checkpoint.name, updates=updates, completed_epochs=completed_epochs)


def edit_distance(reference, candidate):
    previous = list(range(len(candidate) + 1))
    for index, token in enumerate(reference, 1):
        row = [index]
        for offset, prediction in enumerate(candidate, 1):
            row.append(min(row[-1] + 1, previous[offset] + 1,
                           previous[offset - 1] + (token != prediction)))
        previous = row
    return previous[-1]


def asr_metrics(rows, traces):
    from experiments.evaluate_lifemate_asr import words

    predicted = {}
    with traces.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            for event in record["events"]:
                if event["event"] == "transcripts":
                    predicted[record["id"]] = event["request"]
    errors = chars = tokens = reference_chars = numeric = 0
    for row in rows:
        reference = row["audio_text"]
        candidate = predicted.get(row["id"], "")
        ref_words, hyp_words = words(reference), words(candidate)
        errors += edit_distance(ref_words, hyp_words)
        tokens += len(ref_words)
        ref_chars, hyp_chars = " ".join(ref_words), " ".join(hyp_words)
        chars += edit_distance(ref_chars, hyp_chars)
        reference_chars += len(ref_chars)
        numeric += re.findall(r"\d+", reference) != re.findall(r"\d+", candidate)
    return dict(normalized_token_error_pct=100 * errors / max(tokens, 1),
                normalized_char_error_pct=100 * chars / max(reference_chars, 1),
                digit_string_mismatches=numeric, missing_transcripts=len(rows) - len(predicted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-root", "pho", "reference", "stock", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-minutes", type=int, default=35)
    parser.add_argument("--base", choices=("auto", "pho", "turbo"), default="auto")
    args = parser.parse_args()
    os.umask(0o077)
    from transformers.utils import import_utils
    import_utils._torchvision_available = False
    from experiments.evaluate_lifemate_asr import paired, read_report
    from lifemate_baseline import load_dataset
    from lifemate_experiment import selection, write_json
    from lifemate_phase2 import run_variant
    from predict import load_models

    stock = read_report(args.stock / "stock_summary.json")
    use_pho = args.base == "pho" or (
        args.base == "auto" and stock["delta_points"] > 0 and not stock["at_or_over_30s"]
    )
    base = args.pho if use_pho else args.model_root / "asr"
    args.output.mkdir()
    clips, manifest = fitting_manifest(args.data_root, args.output)
    print("STARTING_MODEL", base, flush=True)
    summary = {"base": str(base), "manifest": manifest, "stock_pilot": stock}
    write_json(args.output / "summary.json", summary)
    summary["training"] = train(base, clips, args.output, epochs=args.epochs,
                                minutes=args.max_minutes)
    write_json(args.output / "summary.json", summary)
    model_root = args.output / "models"
    model_root.mkdir()
    (model_root / "agent").symlink_to((args.model_root / "agent").resolve(), target_is_directory=True)
    (model_root / "asr").symlink_to((args.output / "merged").resolve(), target_is_directory=True)
    models = load_models(model_root=model_root, adapter_path=args.reference / "adapter")

    def evaluate(split, output):
        ids = set(selection(split))
        dataset = load_dataset(args.data_root, "train", ids=ids)
        traces = output / "tuned.private.jsonl"
        with traces.open("x", encoding="utf-8") as stream:
            report = run_variant("tuned", args.data_root, output, dataset, models, ids,
                                 stream, inference_options=models.prediction_options)
        metrics = asr_metrics(dataset[0], traces)
        return report, metrics

    tuned, summary["tuned_asr_metrics"] = evaluate("pilot", args.output)
    pilot_ref = read_report(args.reference / "evaluation/chronological_score.json")
    summary["tuned_vs_turbo"] = paired(tuned, pilot_ref)
    summary["tuned_vs_pho"] = paired(
        tuned, read_report(args.stock / "phowhisper_stock_score.json")
    )
    ids = set(selection("pilot"))
    rows = load_dataset(args.data_root, "train", ids=ids)[0]
    summary["pho_asr_metrics"] = asr_metrics(rows, args.stock / "stock.private.jsonl")
    write_json(args.output / "summary.json", summary)
    if (tuned["score"] > max(stock["candidate"], stock["baseline"])
            and not tuned["at_or_over_30s"]):
        dev = args.output / "development"
        dev.mkdir()
        result, summary["development_asr_metrics"] = evaluate("development", dev)
        dev_ref = read_report(args.reference / "evaluation/development/chronological_score.json")
        summary["development_vs_turbo"] = paired(result, dev_ref)
        write_json(args.output / "summary.json", summary)
    print("ASR_LORA_EXPERIMENT_READY", flush=True)


if __name__ == "__main__":
    main()

