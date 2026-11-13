"""Fetch only the pinned model files during image construction."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for directory, name, revision in (
        ("agent", "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
        ("asr", "openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
    ):
        snapshot_download(repo_id=name, revision=revision, local_dir=args.output / directory,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "*.jinja", "LICENSE*", "README.md"])


if __name__ == "__main__":
    main()
