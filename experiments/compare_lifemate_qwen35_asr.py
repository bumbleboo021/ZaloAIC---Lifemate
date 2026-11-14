"""Compare greedy and beam-3 PhoWhisper requests with Qwen3.5 step 548."""

import argparse
import json
import os
from pathlib import Path
import re
from statistics import mean

from experiments.evaluate_lifemate_asr import paired, words
from experiments.evaluate_lifemate_qwen35 import VllmAgent, serving_agent
from lifemate_baseline import load_dataset
from lifemate_experiment import selection
from lifemate_phase2 import run_variant


def transcripts(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for event in row["events"]:
                if event["event"] == "transcripts":
                    result[row["id"]] = event["request"]
    return result


def compare(stage, data, output, models):
    ids = set(selection(stage))
    dataset = load_dataset(data, "train", ids=ids)
    output.mkdir()
    reports = {}
    samples = {}
    for name, beams in (("greedy", 1), ("beam3", 3)):
        print("ASR_VARIANT_START", stage, name, flush=True)
        trace = output / f"{name}.private.jsonl"
        with trace.open("x", encoding="utf-8") as stream:
            reports[name] = run_variant(name, data, output, dataset, models, ids, stream,
                                        inference_options=dict(models.prediction_options,
                                                               request_asr_beams=beams))
        samples[name] = transcripts(trace)

    result = paired(reports["beam3"], reports["greedy"])
    result["variants"] = {}
    for name, report in reports.items():
        multi = [item["score"] for item in report["details"] if item["gold_calls"] > 1]
        result["variants"][name] = dict(
            {key: value for key, value in report.items() if key != "details"},
            multi_call_score=100 * mean(multi) if multi else None,
        )

    changed = []
    digit_pattern = re.compile(r"\d+")
    for row in dataset[0]:
        key = row["id"]
        greedy = samples["greedy"].get(key)
        beam = samples["beam3"].get(key)
        if greedy is None or beam is None or words(greedy) == words(beam):
            continue
        changed.append({
            "id": key, "reference": row["audio_text"], "greedy": greedy, "beam3": beam,
            "digit_strings_changed": digit_pattern.findall(greedy) != digit_pattern.findall(beam),
        })
    with (output / "changed_requests.private.jsonl").open("x", encoding="utf-8") as stream:
        for item in changed:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    result["transcripts"] = {
        "changed": len(changed), "digit_strings_changed": sum(x["digit_strings_changed"] for x in changed),
        "missing_paired_transcripts": len(ids - (samples["greedy"].keys() & samples["beam3"].keys())),
    }
    return result


def save(output, summary):
    temporary = output / "summary.partial"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "agent", "asr", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir()
    with serving_agent(args.agent, args.output / "server.log", port=18085) as url:
        models = VllmAgent(url, args.agent, args.asr)
        pilot = compare("pilot", args.data, args.output / "pilot", models)
        summary = {"checkpoint": 548, "pilot": pilot}
        save(args.output, summary)
        print("ASR_PILOT_READY", pilot["delta_points"], flush=True)
        if pilot["delta_points"] > 0 and not pilot["at_or_over_30s"]:
            development = compare("development", args.data, args.output / "development", models)
            summary["development"] = development
            summary["outcome"] = ("development_improved" if development["delta_points"] > 0
                                  and not development["at_or_over_30s"] else "development_gate_failed")
        else:
            summary["outcome"] = "pilot_gate_failed"
        save(args.output, summary)
    print("ASR_COMPARISON_READY", summary["outcome"], flush=True)


if __name__ == "__main__":
    main()
