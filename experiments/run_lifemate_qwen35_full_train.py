"""Continue step 548 on all labeled TRAIN rows and export a public submission on Molab."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from fsspec.implementations.http import HTTPFileSystem


PREFIX = "train+public_test/"


def read_jsonl(stream):
    return [json.loads(line) for line in stream if line.strip()]


def prepare_training(archive, run, simulator):
    from training_code.prepare_lifemate_sft import examples
    import sys

    sys.path.insert(0, str(simulator.parent))
    try:
        from zaic_tools import catalog_tools
    finally:
        sys.path.pop(0)

    with archive.open(PREFIX + "train/train.jsonl") as stream:
        rows = read_jsonl(stream)
    if len(rows) != 1545 or len({row["id"] for row in rows}) != 1545:
        raise ValueError("Expected all 1,545 unique labeled TRAIN requests")
    catalog = catalog_tools(json.loads(archive.read(PREFIX + "train/train_tools_catalog.json")))
    target = run / "full_train.private.jsonl"
    if not target.exists():
        pending = target.with_suffix(".partial")
        with pending.open("w", encoding="utf-8") as stream:
            for row in rows:
                for example in examples(row, catalog):
                    stream.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
        pending.replace(target)
    with target.open(encoding="utf-8") as stream:
        targets = read_jsonl(stream)
    expected = sum(len(row["label"]["tool_calls"]) + 1 for row in rows)
    if len(targets) != expected or {row["id"] for row in targets} != {row["id"] for row in rows}:
        raise ValueError("Full TRAIN coverage or target count failed")
    print("FULL_TRAIN_TARGETS_READY", len(rows), len(targets), flush=True)
    return target, len(targets)


def stage_public(archive, root):
    prefix = PREFIX + "public_test/"
    needed = {prefix + "public_test.jsonl", prefix + "public_test_tools_catalog.json"}
    with archive.open(prefix + "public_test.jsonl") as stream:
        rows = read_jsonl(stream)
    ids = {row["id"] for row in rows}
    if len(rows) != 634 or len(ids) != 634:
        raise ValueError("Expected all 634 unique PUBLIC_TEST requests")
    for row in rows:
        needed.add(prefix + row["audio"])
        needed.update(prefix + turn["audio"] for turn in row["history"] if turn["role"] == "user")
    env_ids = set()
    for member in archive.infolist():
        if member.is_dir() or not member.filename.startswith(prefix):
            continue
        path = Path(member.filename.removeprefix(PREFIX))
        if ".." in path.parts or path.is_absolute() or "\\" in member.filename:
            raise ValueError("Unsafe archive path")
        if path.parts[1] == "public_test_environment" and len(path.parts) >= 4:
            env_ids.add(path.parts[2])
        elif path.parts[1] != "zaic_tools" and member.filename not in needed:
            continue
        out = root / path
        if not out.is_file() or out.stat().st_size != member.file_size:
            out.parent.mkdir(parents=True, exist_ok=True)
            pending = out.with_suffix(out.suffix + ".partial")
            with archive.open(member) as source, pending.open("wb") as target:
                shutil.copyfileobj(source, target)
            pending.replace(out)
        needed.discard(member.filename)
    if needed or env_ids != ids or not (root / "public_test/zaic_tools/__init__.py").is_file():
        raise ValueError("Incomplete public dataset")
    print("PUBLIC_DATA_READY", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Molab recovery folder")
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--repo", default="bumbleboonq/zaic-lifemate")
    parser.add_argument("--hf-prefix", required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--resume", type=Path, help="Checkpoint created by this full-TRAIN run")
    args = parser.parse_args()
    os.umask(0o077)
    root = args.root.resolve()
    source = Path(__file__).resolve().parents[1]
    run = root / "qwen35-step548-full-train"
    run.mkdir(exist_ok=True)
    train = root / ".merge-venv/bin/python"
    infer = root / ".public-inference-venv/bin/python"
    base = root / "qwen-base"
    adapter = root / "restored-milestones/step548/checkpoint-476"
    asr = root / "pho-merged"
    extensions = root / "qwen35-step548-asr-fast/extensions"
    simulator = root / "development-kit/train/zaic_tools"
    for path in (train, infer, base / "config.json", adapter / "adapter_model.safetensors",
                 asr / "config.json", simulator / "__init__.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Restore the Molab recovery artifact: {path}")
    if args.fast and not extensions.is_dir():
        raise FileNotFoundError("Restore the tested fast-kernel environment or omit --fast")

    env = dict(os.environ, PYTHONPATH=str(source), USE_TF="0", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HOME=str(root / ".public-hf-cache"))
    for key in ("LIFEMATE_HF_BACKUP_REPO", "LIFEMATE_HF_BACKUP_RUN", "HF_TOKEN"):
        env.pop(key, None)

    def execute(python, module, *arguments, fast=False):
        local = env.copy()
        if fast:
            local["PYTHONPATH"] = os.pathsep.join(map(str, (extensions, source)))
        subprocess.run([str(python), "-u", "-m", module, *map(str, arguments)],
                       cwd=source, env=local, check=True)

    fs = HTTPFileSystem()
    with fs.open(args.archive_url, "rb", block_size=2 * 1024 * 1024) as stream, zipfile.ZipFile(stream) as archive:
        fitting, target_count = prepare_training(archive, run, simulator)

    training = run / "training"
    if not (training / "training_metrics.json").is_file():
        if args.resume and args.resume.resolve().parent != training:
            raise ValueError("Resume from a checkpoint inside this full-TRAIN run")
        initialization = ["--resume", args.resume] if args.resume else ["--init-adapter", adapter]
        module = "training_code.train_lifemate_qwen35_fast" if args.fast else "training_code.train_lifemate_lora"
        execute(train, module, "--fitting", fitting, "--agent", base, *initialization,
                "--output", training, "--model-family", "qwen35", "--step-offset", "548",
                "--max-sequence-tokens", "16384", "--max-minutes", "480", fast=args.fast)
    metrics = json.loads((training / "training_metrics.json").read_text())
    if metrics["epoch"] < 0.999 or metrics["optimizer_steps"] != (target_count + 7) // 8:
        raise RuntimeError(f"Training did not complete one pass over all {target_count} targets")
    step = metrics["cumulative_updates"]
    print("FULL_TRAIN_READY", step, flush=True)

    from dotenv import dotenv_values
    from huggingface_hub import get_token
    from training_code.backup_lifemate_checkpoint import CheckpointBackup

    os.environ.pop("HF_HUB_OFFLINE", None)
    token = dotenv_values("/marimo/.env").get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Configure the private Hugging Face backup token on Molab")
    os.environ["HF_TOKEN"] = token
    backup = CheckpointBackup(args.repo, args.hf_prefix)
    for name in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "training_metrics.json"):
        backup.upload_verified(training / name, f"{args.hf_prefix}/training/{name}")
    merged = run / "merged"
    if not (run / "merge.ready").is_file():
        execute(train, "training_code.merge_lifemate_qwen35", base, training / "adapter", merged)
        for file in base.iterdir():
            target = merged / file.name
            if (file.is_file() and file.suffix in {".json", ".jinja", ".txt"}
                    and file.name != "model.safetensors.index.json" and not target.exists()):
                shutil.copyfile(file, target)
        (run / "merge.ready").touch()

    public_kit = run / "public-kit"
    output = run / "public"
    if not (output / "summary.json").is_file():
        with fs.open(args.archive_url, "rb", block_size=2 * 1024 * 1024) as stream, zipfile.ZipFile(stream) as archive:
            stage_public(archive, public_kit)
        execute(infer, "experiments.evaluate_lifemate_qwen35", "--stage", "public",
                "--data-root", public_kit, "--agent", merged, "--asr", asr, "--output", output)
    submission = output / "lifemate_submission_qwen35_4b.jsonl"
    summary = json.loads((output / "summary.json").read_text())
    with submission.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if summary["samples"] != 634 or summary["max_ms"] >= 30000 or summary["sha256"] != digest:
        raise ValueError("Public JSONL validation failed")
    filename = f"lifemate_submission_qwen35_full_train_step{step}.jsonl"
    backup.upload_verified(submission, f"{args.hf_prefix}/public/{filename}")
    backup.upload_verified(output / "summary.json", f"{args.hf_prefix}/public/summary.json")
    print("FULL_TRAIN_PUBLIC_READY", step, filename, digest, flush=True)


if __name__ == "__main__":
    main()
