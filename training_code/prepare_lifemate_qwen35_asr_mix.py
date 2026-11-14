"""Prepare clean and screened PhoWhisper targets for LifeMate fitting only."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from zipfile import ZipFile

from experiments.evaluate_lifemate_qwen35 import VllmAgent, api, serving_agent
from lifemate_baseline import json_text
from training_code.prepare_lifemate_sft import examples
from training_code.screen_lifemate_asr import CHECK, SYSTEM


PREFIX = "train+public_test/train/"


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def approved(url, instruction, original, candidate):
    response = api(url, "/v1/chat/completions", {
        "model": "lifemate-agent",
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json_text([original, candidate])},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 8,
    }, timeout=30)
    return (response["choices"][0]["message"]["content"] or "").strip().rstrip(".").upper() == "YES"


def read_manifest(archive, validation):
    dev = set((validation / "development.txt").read_text().split())
    holdout = set((validation / "holdout.txt").read_text().split())
    with archive.open(PREFIX + "train.jsonl") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = {row["id"] for row in rows}
    excluded = dev | holdout
    if len(dev) != 232 or len(holdout) != 155 or dev & holdout:
        raise ValueError("Unexpected validation split")
    if len(rows) != 1545 or len(ids) != len(rows) or not excluded <= ids:
        raise ValueError("Unexpected official TRAIN manifest")
    with archive.open(PREFIX + "train_tools_catalog.json") as stream:
        raw = json.load(stream)
    from zaic_tools import catalog_tools
    catalog = catalog_tools(raw)
    return [row for row in rows if row["id"] not in excluded], catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("agent", "asr", "output", "validation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--port", type=int, default=18087)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    from fsspec.implementations.http import HTTPFileSystem

    cache = args.output / "requests.private.jsonl"
    with (
        HTTPFileSystem().open(args.archive_url, "rb", block_size=2 * 1024 * 1024) as source,
        ZipFile(source) as archive,
    ):
        rows, catalog = read_manifest(archive, args.validation)
        transcriber = VllmAgent(None, args.agent, args.asr)
        stats = Counter()
        with tempfile.TemporaryDirectory(dir=args.output) as scratch, cache.open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows, 1):
                audio = Path(scratch) / ("request" + Path(row["audio"]).suffix)
                audio.write_bytes(archive.read(PREFIX + row["audio"]))
                text = transcriber.transcribe(audio, 25)
                stream.write(json_text({"id": row["id"], "text": text}) + "\n")
                if index == 1 or index % 50 == 0:
                    stream.flush()
                    print("FITTING_ASR", index, len(rows), flush=True)
    print("FITTING_ASR_READY", len(rows), flush=True)
    del transcriber
    import torch
    torch.cuda.empty_cache()

    transcripts = {}
    with cache.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            transcripts[item["id"]] = item["text"]
    target = args.output / "mixed.private.jsonl"
    pending = target.with_suffix(".partial")
    with (
        serving_agent(args.agent, args.output / "review-server.log", port=args.port) as url,
        pending.open("x", encoding="utf-8") as stream,
    ):
        for index, row in enumerate(rows, 1):
            original = row["audio_text"]
            candidate = transcripts[row["id"]]
            keep = False
            if not candidate or words(original) == words(candidate):
                stats["empty_or_unchanged"] += 1
            elif re.findall(r"\d+", original) != re.findall(r"\d+", candidate):
                stats["changed_digit_strings"] += 1
            else:
                keep = approved(url, SYSTEM, original, candidate) and approved(url, CHECK, candidate, original)
                stats["accepted" if keep else "rejected"] += 1
            for item in examples(row, catalog):
                stream.write(json_text(item) + "\n")
                stats["clean_targets"] += 1
            if keep:
                for item in examples(dict(row, audio_text=candidate), catalog):
                    stream.write(json_text(item) + "\n")
                    stats["asr_targets"] += 1
            if index % 50 == 0:
                print("FITTING_REVIEW", index, dict(stats), flush=True)
    if not stats["asr_targets"]:
        raise RuntimeError("No ASR variants passed the screen")
    pending.replace(target)
    summary = {"fitting_requests": len(rows), **stats}
    (args.output / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("FITTING_MIX_READY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
