"""Train a bounded Qwen LoRA run on prepared LifeMate fitting targets."""

import argparse
import json
import os
import shutil
from pathlib import Path
import time

os.environ["USE_TF"] = "0"
if not os.environ.get("LIFEMATE_HF_BACKUP_REPO"):
    os.environ["HF_HUB_OFFLINE"] = "1"


def encode(tokenizer, example, limit):
    messages = example["prompt"]
    options = {"tools": example["tools"], "enable_thinking": False, "tokenize": False}
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(messages + example["completion"], **options)
    # The rendered assistant turn can omit the empty thinking block added by
    # the inference generation prompt. Keep the actual inference prefix.
    _, marker, target = full.rpartition("<|im_start|>assistant\n")
    if not marker or not target.rstrip().endswith("<|im_end|>"):
        raise ValueError("Unexpected Qwen assistant target template")
    target = target.removeprefix("<think>\n\n</think>\n\n")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if len(prefix_ids) + len(target_ids) > limit:
        return None
    return {"input_ids": prefix_ids + target_ids, "labels": [-100] * len(prefix_ids) + target_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitting", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-minutes", type=int, default=90)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap optimizer steps, or use -1 for the epoch schedule")
    parser.add_argument("--milestone-steps", type=int, nargs="+", default=[],
                        help="Save these optimizer steps in a separate milestone directory")
    parser.add_argument("--step-offset", type=int, default=0,
                        help="Earlier updates to add to milestone names and metrics")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-adapter", type=Path, help="Start with adapter weights and a fresh optimizer")
    parser.add_argument("--max-sequence-tokens", type=int, default=12288)
    parser.add_argument("--model-family", choices=("qwen3", "qwen35"), default="qwen3")
    args = parser.parse_args()
    if args.resume and args.init_adapter:
        parser.error("Choose either --resume or --init-adapter.")
    if args.max_steps != -1 and args.max_steps < 1:
        parser.error("--max-steps must be positive or -1.")
    if args.step_offset < 0 or any(step <= args.step_offset for step in args.milestone_steps):
        parser.error("Milestones must be greater than the non-negative step offset.")
    if args.max_minutes < 1:
        parser.error("--max-minutes must be positive")
    if not 1 <= args.max_sequence_tokens <= 16384:
        parser.error("--max-sequence-tokens must be between 1 and 16384.")
    if args.output.exists() and args.resume is None:
        parser.error("Use a new output directory or provide --resume.")

    import torch
    from transformers.utils import import_utils

    # Text-only training does not need molab's optional vision backend.
    import_utils._torchvision_available = False
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("A molab CUDA GPU is required")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.agent, local_files_only=True)
    encoded = []
    skipped = 0
    with args.fitting.open(encoding="utf-8") as stream:
        for line in stream:
            item = encode(tokenizer, json.loads(line), args.max_sequence_tokens)
            if item is None:
                skipped += 1
            else:
                encoded.append(item)
    if not encoded:
        raise ValueError("All fitting targets exceeded the sequence limit")
    print(f"Encoded {len(encoded)} targets, skipped {skipped} over {args.max_sequence_tokens} tokens", flush=True)
    model_class = AutoModelForCausalLM
    target_modules = ["q_proj", "v_proj"]
    if args.model_family == "qwen35":
        from transformers import Qwen3_5ForConditionalGeneration
        model_class = Qwen3_5ForConditionalGeneration
        target_modules = r".*language_model.*(?:q_proj|v_proj|in_proj_qkv)$"
    model = model_class.from_pretrained(args.agent, local_files_only=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    if args.init_adapter:
        model = PeftModel.from_pretrained(
            model, args.init_adapter, is_trainable=True, local_files_only=True,
        )
    else:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                                               target_modules=target_modules, lora_dropout=0.05))
    model.print_trainable_parameters()
    deadline = time.monotonic() + args.max_minutes * 60

    milestones = set(args.milestone_steps)
    backup = None
    if backup_repo := os.environ.get("LIFEMATE_HF_BACKUP_REPO"):
        from training_code.backup_lifemate_checkpoint import CheckpointBackup
        backup = CheckpointBackup(backup_repo, os.environ["LIFEMATE_HF_BACKUP_RUN"])

    class SaveMilestonesAndStopAfterBudget(TrainerCallback):
        def on_save(self, settings, state, control, **kwargs):
            cumulative_step = state.global_step + args.step_offset
            checkpoint = args.output / f"checkpoint-{state.global_step}"
            if backup is not None:
                backup.save_checkpoint(checkpoint, cumulative_step)
            if milestones and (cumulative_step in milestones or state.global_step >= state.max_steps):
                directory = args.output / "milestones"
                directory.mkdir(exist_ok=True)
                destination = directory / f"checkpoint-{cumulative_step}"
                if not destination.exists():
                    pending = directory / f".checkpoint-{cumulative_step}.partial"
                    shutil.copytree(checkpoint, pending)
                    pending.rename(destination)
                print(f"MILESTONE_READY {cumulative_step}", flush=True)
            return control

        def on_step_end(self, settings, state, control, **kwargs):
            if milestones and (
                state.global_step + args.step_offset in milestones
                or state.global_step >= state.max_steps
            ):
                control.should_save = True
            if time.monotonic() >= deadline:
                print(f"Time budget reached at optimizer step {state.global_step}", flush=True)
                control.should_save = True
                control.should_training_stop = True
            return control

    optimizer_steps = (len(encoded) + 7) // 8
    if args.max_steps > 0:
        optimizer_steps = args.max_steps
    warmup_steps = max(1, (3 * optimizer_steps + 99) // 100)
    settings = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=1, max_steps=args.max_steps,
        learning_rate=1e-4, warmup_steps=warmup_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=10, save_total_limit=3, logging_steps=1, report_to=[], seed=42,
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=settings, train_dataset=encoded, processing_class=tokenizer,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
                      callbacks=[SaveMilestonesAndStopAfterBudget()])
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output / "adapter"))
    metrics = dict(result.metrics, optimizer_steps=trainer.state.global_step,
                   cumulative_updates=args.step_offset + trainer.state.global_step)
    (args.output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if backup is not None:
        backup.save_metrics(args.output / "training_metrics.json", metrics["cumulative_updates"])
    print("ADAPTER_READY", flush=True)


if __name__ == "__main__":
    main()
