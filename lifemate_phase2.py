"""Run the next LifeMate development comparisons on the molab GPU."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median, quantiles

from lifemate_baseline import Models, audio_file, infer, json_text, load_dataset
from lifemate_eval import read_rows, score
from lifemate_experiment import selection
from predict import OPTIONS


class ReferenceAudio:
    def __init__(self, models, transcripts):
        self.models = models
        self.transcripts = transcripts

    def transcribe(self, path, seconds_left):
        if path in self.transcripts:
            return self.transcripts[path]
        return self.models.transcribe(path, seconds_left)

    def next_action(self, *args, **kwargs):
        return self.models.next_action(*args, **kwargs)


def transcript_overrides(row, split_dir, variant):
    if variant not in {"reference_request", "reference_all"}:
        return {}
    turns = [row]
    if variant == "reference_all":
        turns.extend(turn for turn in row.get("history", []) if turn["role"] == "user")
    return {audio_file(split_dir, turn["audio"]): turn["audio_text"] for turn in turns}


def run_variant(name, root, output, dataset, models, expected_ids, diagnostics, *, inference_options=None):
    options = dict(OPTIONS, chronological_history=name == "chronological")
    if inference_options is not None:
        options.update(inference_options)
    rows, split_dir, tools, factory = dataset
    outcomes = Counter()
    failures = Counter()
    predictions = {}
    path = output / f"{name}.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            events = []

            def observe(event, **details):
                events.append({"event": event, **details})

            refs = transcript_overrides(row, split_dir, name)
            sample_models = ReferenceAudio(models, refs) if refs else models
            result, status, milliseconds = infer(
                row, split_dir, tools, factory, sample_models, **options, observe=observe,
            )
            result["time"] = milliseconds
            predictions[row["id"]] = result
            stream.write(json_text(result) + "\n")
            outcomes[status] += 1
            gold = row["label"]["tool_calls"]
            actual = result["tool_calls"]
            if gold and not actual:
                failures["no_calls_when_expected"] += 1
            if gold and actual and gold[0]["name"] != actual[0]["name"]:
                failures["wrong_first_executed_tool"] += 1
            if gold and actual and gold[0]["name"] == actual[0]["name"] and gold[0]["arguments"] != actual[0]["arguments"]:
                failures["first_arguments_differ_exactly"] += 1
            if not gold and actual:
                failures["calls_when_empty_expected"] += 1
            for event in events:
                if event["event"] == "validation_error":
                    failures["invalid_proposals"] += 1
                if event["event"] == "proposal" and event["action"] is None:
                    failures["unparsed_proposals"] += 1
            if diagnostics is not None:
                diagnostics.write(json_text({"variant": name, "id": row["id"], "events": events}) + "\n")
                diagnostics.flush()
            print(f"{name} {index}/{len(rows)} {status} {milliseconds}ms", flush=True)
    report = score(root, {row["id"]: row for row in rows}, predictions, expected_ids)
    timings = [row["time"] for row in predictions.values()]
    report.update(median_ms=median(timings), p95_ms=quantiles(timings, n=20, method="inclusive")[-1],
                  at_or_over_30s=sum(ms >= 30000 for ms in timings), stops=outcomes, failures=failures,
                  diagnostic_only=name.startswith("reference_"))
    (output / f"{name}_score.json").write_text(json_text(report) + "\n", encoding="utf-8")
    print(json_text({key: value for key, value in report.items() if key != "details"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-path", type=Path, required=True)
    parser.add_argument("--agent-path", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, help="Optional restricted JSONL of transcripts and tool responses")
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root) or (args.diagnostics and args.diagnostics.resolve().is_relative_to(root)):
        parser.error("Keep experiment outputs outside the official kit.")
    models = Models(str(args.asr_path.resolve()), str(args.agent_path.resolve()), offline=True)
    output.mkdir(parents=True, exist_ok=False)
    pilot = set(selection("pilot"))
    dataset = load_dataset(root, "train", ids=pilot)
    trace = args.diagnostics.open("x", encoding="utf-8") if args.diagnostics else None
    try:
        reports = {name: run_variant(name, root, output, dataset, models, pilot, trace)
                   for name in ("baseline", "reference_request", "reference_all", "chronological")}
        candidate = reports["chronological"]
        if candidate["score"] > reports["baseline"]["score"] and not candidate["at_or_over_30s"]:
            ids = set(selection("development"))
            dataset = load_dataset(root, "train", ids=ids)
            full = output / "development"
            full.mkdir()
            run_variant("baseline", root, full, dataset, models, ids, trace)
            run_variant("chronological", root, full, dataset, models, ids, trace)
    finally:
        if trace is not None:
            trace.close()


if __name__ == "__main__":
    main()
