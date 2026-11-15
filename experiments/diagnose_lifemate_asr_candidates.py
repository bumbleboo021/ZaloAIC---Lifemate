"""Report reference-based headroom of the ASR candidate set on 48 labeled audio clips."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from fsspec.implementations.http import HTTPFileSystem

from experiments.compare_lifemate_step1184_variants import asr_candidates, tokens
from experiments.evaluate_lifemate_qwen35 import VllmAgent
from lifemate_experiment import write_json


def edits(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("agent", "asr", "ids", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--archive-url", required=True)
    args = parser.parse_args()
    ids = set(args.ids.read_text().split())
    prefix = "train+public_test/train/"
    model = VllmAgent(None, args.agent, args.asr)
    summary = {"samples": 0, "reference_words": 0, "greedy_errors": 0, "oracle_errors": 0,
               "rows_with_better_alternative": 0, "rows_with_alternatives": 0,
               "rows_below_confidence_gate": 0, "rows_above_gate_with_better_alternative": 0}
    with HTTPFileSystem().open(args.archive_url, "rb", block_size=2097152) as stream, ZipFile(stream) as archive:
        with archive.open(prefix + "train.jsonl") as manifest:
            rows = [row for row in map(json.loads, manifest) if row["id"] in ids]
        if {row["id"] for row in rows} != ids or len(rows) != 48:
            raise ValueError("Unexpected ASR diagnostic cohort")
        with TemporaryDirectory() as scratch:
            for row in rows:
                path = Path(scratch) / "speech.wav"
                path.write_bytes(archive.read(prefix + row["audio"]))
                greedy = model.transcribe(path, 20)
                confidence, candidates = asr_candidates(model, path, 9, greedy, force=True)
                reference = tokens(row["audio_text"])
                base = edits(reference, tokens(greedy))
                best = min([base, *(edits(reference, tokens(x)) for x in candidates)])
                summary["samples"] += 1
                summary["reference_words"] += len(reference)
                summary["greedy_errors"] += base
                summary["oracle_errors"] += best
                summary["rows_with_better_alternative"] += best < base
                summary["rows_with_alternatives"] += len(candidates) > 1
                below = confidence is not None and confidence < -0.65
                summary["rows_below_confidence_gate"] += below
                summary["rows_above_gate_with_better_alternative"] += best < base and not below
    summary.update({name + "_wer": summary[name + "_errors"] / summary["reference_words"]
                    for name in ("greedy", "oracle")})
    write_json(args.output, summary)
    print("ASR_ORACLE_READY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
