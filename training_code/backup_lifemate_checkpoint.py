"""Upload resumable checkpoints from Molab and verify each committed copy."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


class CheckpointBackup:
    def __init__(self, repo, run):
        self.repo = repo
        self.run = run
        self.token = os.environ["HF_TOKEN"]
        self.api = HfApi(token=self.token)
        if not self.api.repo_info(repo, repo_type="dataset").private:
            raise ValueError("The checkpoint backup repo must be private.")

    def upload_verified(self, source, remote):
        with source.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        for attempt in range(3):
            try:
                commit = self.api.upload_file(
                    path_or_fileobj=source, path_in_repo=remote,
                    repo_id=self.repo, repo_type="dataset",
                )
                with tempfile.TemporaryDirectory() as cache:
                    downloaded = hf_hub_download(
                        repo_id=self.repo, repo_type="dataset", filename=remote,
                        revision=commit.oid, token=self.token, cache_dir=cache,
                    )
                    with Path(downloaded).open("rb") as stream:
                        actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise RuntimeError("Remote backup checksum mismatch")
                print(f"HF_BACKUP_READY {remote} {expected}", flush=True)
                return
            except (HfHubHTTPError, OSError, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    def save_checkpoint(self, checkpoint, cumulative_step):
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                     "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(shutil.make_archive(
                str(Path(temporary) / f"qwen35-step{cumulative_step}"), "zip",
                root_dir=checkpoint.parent, base_dir=checkpoint.name,
            ))
            self.upload_verified(archive, f"{self.run}/{archive.name}")

    def save_metrics(self, path, cumulative_step):
        self.upload_verified(path, f"{self.run}/training_metrics_step{cumulative_step}.json")
