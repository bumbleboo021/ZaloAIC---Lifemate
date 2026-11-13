"""Evaluate current-request Whisper beam search with the frozen LoRA adapter."""
import argparse
import json
import os
from pathlib import Path
import re
from statistics import mean
import unicodedata

from lifemate_baseline import load_dataset
from lifemate_experiment import selection
from lifemate_phase2 import run_variant
from predict import load_models


def paired(candidate, baseline):
    prior = {row["id"]: row["score"] for row in baseline["details"]}
    assert prior.keys() == {row["id"] for row in candidate["details"]}
    deltas = [row["score"] - prior[row["id"]] for row in candidate["details"]]
    return dict(baseline=baseline["score"], candidate=candidate["score"],
                delta_points=100 * mean(deltas), wins=sum(x > 0 for x in deltas),
                losses=sum(x < 0 for x in deltas), ties=sum(x == 0 for x in deltas),
                median_ms=candidate["median_ms"], p95_ms=candidate["p95_ms"],
                max_ms=candidate["max_time_ms"], at_or_over_30s=candidate["at_or_over_30s"],
                baseline_failures=baseline["failures"], candidate_failures=candidate["failures"])


def read_report(path):
    return json.loads(path.read_text(encoding="utf-8"))


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def audit(dataset, diagnostics, baseline, reference):
    prior = {row["id"]: row["score"] for row in baseline["details"]}
    diffs = {row["id"]: row["score"] - prior[row["id"]] for row in reference["details"]}
    actual = {}
    for line in diagnostics.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for event in row["events"]:
            if event["event"] == "transcripts":
                actual[row["id"]] = event["request"]
    summary = {}
    for label, sign in (("reference_wins", 1), ("reference_losses", -1)):
        rows = [row for row in dataset[0] if sign * diffs[row["id"]] > 0]
        summary[label] = dict(
            samples=len(rows),
            lexical_changes=sum(words(actual[row["id"]]) != words(row["audio_text"])
                                for row in rows if row["id"] in actual),
            missing_transcripts=sum(row["id"] not in actual for row in rows),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-root", "adapter", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir()
    models = load_models(model_root=args.model_root, adapter_path=args.adapter)
    ids = set(selection("pilot"))
    dataset = load_dataset(args.data_root, "train", ids=ids)
    options = models.prediction_options
    saved = read_report(args.reference / "evaluation/chronological_score.json")
    reference = read_report(args.reference / "transcript_diagnostic_001/reference_request_score.json")
    traces = args.output / "baseline.private.jsonl"
    with traces.open("x", encoding="utf-8") as stream:
        baseline = run_variant("audio_greedy", args.data_root, args.output, dataset,
                               models, ids, stream, inference_options=options)
    parity = paired(baseline, saved)
    if parity["ties"] != len(ids):
        raise RuntimeError(f"Pilot score replay differs from saved step 362: {parity}")
    summary = {"audit": audit(dataset, traces, baseline, reference)}
    print("AUDIT", summary["audit"], flush=True)
    candidate_options = dict(options, request_asr_beams=3)
    with (args.output / "candidate.private.jsonl").open("x", encoding="utf-8") as stream:
        candidate = run_variant("audio_beam3", args.data_root, args.output, dataset,
                                models, ids, stream, inference_options=candidate_options)
    summary["pilot"] = paired(candidate, baseline)
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PILOT", summary["pilot"], flush=True)
    if summary["pilot"]["delta_points"] > 0 and not candidate["at_or_over_30s"]:
        ids = set(selection("development"))
        dataset = load_dataset(args.data_root, "train", ids=ids)
        dev = args.output / "development"
        dev.mkdir()
        candidate = run_variant("audio_beam3", args.data_root, dev, dataset, models, ids,
                                None, inference_options=candidate_options)
        previous = read_report(args.reference / "evaluation/development/chronological_score.json")
        summary["development"] = paired(candidate, previous)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("DEVELOPMENT", summary["development"], flush=True)
    print("ASR_EVALUATION_READY", flush=True)


if __name__ == "__main__":
    main()
