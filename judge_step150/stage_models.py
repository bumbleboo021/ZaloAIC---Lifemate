"""Stage the two verified step-150 inference models on the remote build host."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from huggingface_hub import snapshot_download

AGENT = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
ASR = ("vinai/PhoWhisper-medium", "55a7e3eb6c906de891f8f06a107754427dd3be79")
AGENT_ARCHIVE_SHA = "7fdbcea3fb679adc45976cf7f6cabbea0e29c65c31a4f2932609eac43c4f065b"
ASR_ARCHIVE_SHA = "adc71eee4a5ffb833c0962e52727573ced6c617ce2a1b48f12c035e119391fbb"
MERGED_ASR_SHA = "c2bc73b7300c76ecc4736b4b48d5d08344a4a22c046d0a0b4ca8ad67f052f2f2"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, destination, expected):
    if checksum(archive) != expected:
        raise ValueError(f"Unexpected adapter archive hash: {archive}")
    with ZipFile(archive) as source:
        for member in source.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Invalid archive member: {member.filename}")
        source.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-adapter", required=True, type=Path)
    parser.add_argument("--asr-adapter", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("judge_step150/staged_models"))
    parser.add_argument("--asr-python", type=Path, required=True,
                        help="Python in the Transformers 4.57.1 ASR merge environment")
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Choose a new output directory.")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        work = Path(tmp)
        agent_adapter = work / "agent-adapter"
        asr_adapter = work / "asr-adapter"
        unpack(args.agent_adapter, agent_adapter, AGENT_ARCHIVE_SHA)
        unpack(args.asr_adapter, asr_adapter, ASR_ARCHIVE_SHA)
        agent = Path(snapshot_download(AGENT[0], revision=AGENT[1],
                                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"]))
        asr = Path(snapshot_download(ASR[0], revision=ASR[1],
                                    allow_patterns=["*.safetensors", "pytorch_model.bin", "*.json", "*.txt"]))
        stage = work / "stage"
        stage.mkdir()
        subprocess.run([sys.executable, "-m", "training_code.merge_lifemate_qwen35",
                        str(agent), str(agent_adapter / "adapter"), str(stage / "agent")], check=True)
        import shutil
        for item in agent.iterdir():
            if (item.is_file() and item.suffix in {".json", ".txt", ".jinja"}
                    and item.name != "model.safetensors.index.json"
                    and not (stage / "agent" / item.name).exists()):
                shutil.copy2(item, stage / "agent" / item.name)
        subprocess.run([str(args.asr_python.resolve()), "-m", "judge_step150.stage_models",
                        "_merge_asr", str(asr), str(asr_adapter / "epoch-1"), str(stage / "asr")],
                       check=True)
        if checksum(stage / "asr/model.safetensors") != MERGED_ASR_SHA:
            raise ValueError("Merged PhoWhisper differs from the verified molab weights.")
        stage.rename(args.output)
    print(f"STEP150_MODELS_STAGED {args.output}")


def merge_asr(base, adapter, output):
    import torch
    from transformers.utils import import_utils
    import_utils._torchvision_available = False
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from peft import PeftModel

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base, torch_dtype=torch.float32, local_files_only=True
    )
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model = model.merge_and_unload(safe_merge=True).to(torch.float16)
    model.config.use_cache = True
    model.save_pretrained(output)
    AutoProcessor.from_pretrained(adapter, local_files_only=True).save_pretrained(output)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "_merge_asr":
        merge_asr(*sys.argv[2:])
    else:
        main()
