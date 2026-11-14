"""Restore verified LifeMate milestones, export public predictions, and back them up."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

from training_code.backup_lifemate_checkpoint import CheckpointBackup


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = Path(__file__).resolve().parents[1]
    dotenv = dotenv_values("/marimo/.env")
    token = dotenv.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the protected Molab .env.")
    os.environ["HF_TOKEN"] = token
    backup = CheckpointBackup(args.repo, args.run)
    info = backup.api.repo_info(args.repo, repo_type="dataset")

    merge_python = root / ".merge-venv/bin/python"
    infer_python = root / ".public-inference-venv/bin/python"
    env = {k: v for k, v in os.environ.items()
           if k not in {"HF_TOKEN", "PYTHONHOME", "VIRTUAL_ENV"}}
    env.update(PYTHONPATH=str(source), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               USE_TF="0", PYTHONUNBUFFERED="1", HF_HOME=str(root / ".public-hf-cache"))

    def run(*command):
        subprocess.run(command, cwd=source, env=env, check=True)

    output = root / "milestone-submissions"
    output.mkdir(exist_ok=True)
    restored = root / "restored-milestones"
    restored.mkdir(exist_ok=True)
    expected_sha = {548: "9d4c96fbbbe0df76a8c1696dec2aee632b83a143089c0567525e4bfc110210da"}
    for step in (362, 548):
        name = f"{args.run}/qwen35-step{step}.zip"
        archive = Path(hf_hub_download(
            args.repo, filename=name, repo_type="dataset", revision=info.sha, token=token,
        ))
        digest = sha256(archive)
        if step in expected_sha and digest != expected_sha[step]:
            raise ValueError(f"Step-{step} archive checksum mismatch.")
        state_name = f"checkpoint-{step - 72}"
        entry = f"{state_name}/trainer_state.json"
        destination = restored / f"step{step}"
        with zipfile.ZipFile(archive) as checkpoint:
            state = json.loads(checkpoint.read(entry))
            if state["global_step"] != step - 72 or checkpoint.testzip() is not None:
                raise ValueError(f"Invalid step-{step} resume ZIP.")
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in checkpoint.namelist()):
                raise ValueError("Unsafe checkpoint ZIP member.")
            checkpoint.extractall(destination)
        print(f"RESTORED_MILESTONE {step} {digest}", flush=True)

        directory = output / f"step{step}"
        summary = directory / "summary.json"
        if not summary.is_file():
            merged = output / f"merged-step{step}"
            if not (merged / "model.safetensors").is_file():
                run(str(merge_python), str(source / "training_code/merge_lifemate_qwen35.py"),
                    str(root / "qwen-base"), str(destination / state_name), str(merged))
            for file in (root / "qwen-base").iterdir():
                target = merged / file.name
                if (file.is_file() and file.suffix in {".json", ".jinja", ".txt"}
                        and file.name != "model.safetensors.index.json" and not target.exists()):
                    shutil.copyfile(file, target)
            print(f"EXPORT_START {step}", flush=True)
            run(str(infer_python), "-m", "experiments.evaluate_lifemate_qwen35",
                "--data-root", str(args.data), "--agent", str(merged), "--asr", str(root / "pho-merged"),
                "--output", str(directory), "--stage", "public")
        metadata = json.loads(summary.read_text())
        public = directory / "lifemate_submission_qwen35_4b.jsonl"
        if metadata["samples"] != 634 or metadata["max_ms"] >= 30_000 or sha256(public) != metadata["sha256"]:
            raise ValueError(f"Invalid step-{step} public output.")
        print(f"EXPORT_READY {step}", flush=True)
        remote = f"{args.run}/public/step{step}"
        backup.upload_verified(public, f"{remote}/lifemate_submission_qwen35_4b_step{step}.jsonl")
        backup.upload_verified(summary, f"{remote}/summary.json")
        print(f"HF_SUBMISSION_READY {step}", flush=True)


if __name__ == "__main__":
    main()
