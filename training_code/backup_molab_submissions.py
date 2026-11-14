"""Back up completed Molab public milestone exports to a private Hub dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from training_code.backup_lifemate_checkpoint import CheckpointBackup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run", required=True, help="Existing checkpoint backup folder on the Hub")
    parser.add_argument("--wait-minutes", type=int, default=0)
    args = parser.parse_args()

    backup = CheckpointBackup(args.repo, args.run)
    remote_files = set(backup.api.list_repo_files(args.repo, repo_type="dataset"))
    milestones = (232, 362, 548)
    for step in milestones:
        path = f"{args.run}/qwen35-step{step}.zip"
        if path not in remote_files:
            raise FileNotFoundError(f"Missing resume checkpoint on the Hub: {path}")

    assets = f"{args.run}/assets"
    asr = args.root / "pho-adapter.zip"
    backup.upload_verified(asr, f"{assets}/phowhisper-medium-epoch1-adapter.zip")

    pending = set(milestones)
    deadline = time.monotonic() + args.wait_minutes * 60
    while pending:
        for step in sorted(pending):
            directory = args.root / "milestone-submissions" / f"step{step}"
            summary = directory / "summary.json"
            if not summary.is_file():
                continue
            metadata = json.loads(summary.read_text(encoding="utf-8"))
            prediction = directory / "lifemate_submission_qwen35_4b.jsonl"
            data = prediction.read_bytes()
            if (metadata["coverage_and_format"] != "valid"
                    or metadata["samples"] != 634 or len(data.splitlines()) != 634
                    or metadata["max_ms"] >= 30_000
                    or hashlib.sha256(data).hexdigest() != metadata["sha256"]):
                raise ValueError(f"Step {step} did not pass public export checks.")
            remote = f"{args.run}/public/step{step}"
            backup.upload_verified(prediction, f"{remote}/lifemate_submission_qwen35_4b_step{step}.jsonl")
            backup.upload_verified(summary, f"{remote}/summary.json")
            print(f"HF_SUBMISSION_READY {step}", flush=True)
            pending.remove(step)
        if not pending:
            print("HF_MILESTONE_BACKUP_COMPLETE", flush=True)
            return
        if time.monotonic() >= deadline:
            raise SystemExit(f"Waiting for public exports: {sorted(pending)}")
        time.sleep(30)


if __name__ == "__main__":
    main()
