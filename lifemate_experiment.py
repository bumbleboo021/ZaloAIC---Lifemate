"""Run and summarize one LifeMate experiment stage in a molab GPU workspace."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import median, quantiles
import subprocess
import sys

from lifemate_eval import read_rows, validate_public


REPO = Path(__file__).resolve().parent
VARIANTS = {
    "base": [],
    "native_tools": ["--reuse-reads", "--native-tool-format"],
    "native_thinking": ["--reuse-reads", "--native-tool-format", "--thinking"],
    "native_lookups": ["--reuse-reads", "--native-tool-format", "--lookup-guidance"],
    "native_request_last": ["--reuse-reads", "--native-tool-format", "--request-last"],
    "reuse_reads": ["--reuse-reads"],
    "resolve_missing": ["--reuse-reads", "--resolve-missing"],
    "workflow_hints": ["--reuse-reads"],
    "workflow_clarification": ["--reuse-reads", "--resolve-missing"],
}
COMPARISONS = {
    "both": ("base", "reuse_reads"),
    "native": ("reuse_reads", "native_tools"),
    "thinking": ("native_tools", "native_thinking"),
    "lookups": ("native_tools", "native_lookups"),
    "request_last": ("native_tools", "native_request_last"),
    "clarification": ("reuse_reads", "resolve_missing"),
    "workflows": ("reuse_reads", "workflow_hints"),
    "workflows_with_clarification": ("resolve_missing", "workflow_clarification"),
}
SAMPLE_LOG = re.compile(r"\d+/\d+ \S+ (\w+) (\d+)ms")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def selection(stage):
    if stage == "public":
        return None
    split = "holdout" if stage == "holdout" else "development"
    ids = (REPO / "validation" / f"{split}.txt").read_text(encoding="utf-8").split()
    expected = 155 if split == "holdout" else 232
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"Expected {expected} distinct IDs in the {split} split.")
    if stage == "pilot":
        ids = random.Random(2601).sample(sorted(ids), 48)
    return sorted(ids)


def timing_summary(log, expected):
    records = [match.groups() for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
               if (match := SAMPLE_LOG.fullmatch(line))]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} sample timings, found {len(records)} in {log}.")
    times = [int(milliseconds) for _, milliseconds in records]
    return {
        "median_ms": median(times),
        "p95_ms": round(quantiles(times, n=20, method="inclusive")[-1], 1),
        "max_ms": max(times),
        "at_or_over_30s": sum(value >= 30_000 for value in times),
        "stop_statuses": dict(Counter(status for status, _ in records)),
    }


def run_logged(command, log):
    print(f"Running {log.stem}. Progress: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"Run exited with status {result.returncode}. See {log}.")


def run_variant(args, output, split, ids_path, variant, summary):
    prediction = output / f"{variant}.jsonl"
    log = output / f"{variant}.log"
    inference = [sys.executable, "-u", "lifemate_baseline.py", "--data-root", str(args.data_root),
                 "--split", split, "--output", str(prediction), *VARIANTS[variant]]
    if variant.startswith("workflow_"):
        inference.extend(["--workflow-index", str(args.workflow_index)])
    if args.offline:
        inference.append("--offline")
    for option, model_path in (("--asr-path", args.asr_path), ("--agent-path", args.agent_path)):
        if model_path:
            inference.extend([option, str(model_path)])
    if ids_path:
        inference.extend(["--ids", str(ids_path), "--include-time"])

    summary["variants"][variant] = {"status": "running", "command": inference}
    write_json(output / "summary.json", summary)
    run_logged(inference, log)
    if ids_path:
        score_path = output / f"{variant}_score.json"
        run_logged([sys.executable, "lifemate_eval.py", "--data-root", str(args.data_root),
                    "score", "--ids", str(ids_path), "--predictions", str(prediction),
                    "--report", str(score_path)], output / f"{variant}_score.log")
        report = json.loads(score_path.read_text(encoding="utf-8"))
        details = report.pop("details")
    else:
        report = validate_public(args.data_root, prediction)
        details = []
    report["timing"] = timing_summary(log, report["samples"])
    clarification_counts = [sum(call["name"] == "sys.ask_missing_info" for call in row["tool_calls"])
                            for row in read_rows(prediction).values()]
    report["clarification"] = {
        "calls": sum(clarification_counts),
        "samples": sum(count > 0 for count in clarification_counts),
        "repeat_calls": sum(max(0, count - 1) for count in clarification_counts),
    }
    summary["variants"][variant].update(report, status="complete")
    write_json(output / "summary.json", summary)
    return {row["id"]: row["score"] for row in details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "development", "holdout", "public"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New folder for this stage")
    parser.add_argument("--variant", choices=(*COMPARISONS, *VARIANTS))
    parser.add_argument("--asr-path", type=Path, help="Path to a downloaded ASR model directory")
    parser.add_argument("--agent-path", type=Path, help="Path to a downloaded agent model directory")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workflow-index", type=Path, help="Fitting-only workflow examples")
    args = parser.parse_args()

    if args.stage in ("holdout", "public") and args.variant not in VARIANTS:
        parser.error("Select a single variant for holdout and public.")
    variant = args.variant or "both"
    variants = COMPARISONS.get(variant, (variant,))
    if any(name.startswith("workflow_") for name in variants):
        if args.workflow_index is None or not args.workflow_index.is_file():
            parser.error("This variant requires a fitting-only --workflow-index JSON file.")
        args.workflow_index = args.workflow_index.resolve()
    if args.offline and not (args.asr_path and args.agent_path):
        parser.error("--offline requires --asr-path and --agent-path.")
    for field in ("asr_path", "agent_path"):
        path = getattr(args, field)
        if path is not None:
            if not path.is_dir():
                parser.error(f"Model directory does not exist: {path}")
            setattr(args, field, path.resolve())
    args.data_root = args.data_root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(args.data_root):
        parser.error("Choose an output directory outside the official kit.")
    split = "public_test" if args.stage == "public" else "train"
    manifest = args.data_root / split / f"{split}.jsonl"
    if not manifest.is_file():
        parser.error(f"Cannot find {manifest}. Use the extracted kit root.")

    ids = selection(args.stage)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {output}. Choose a fresh directory.")
    ids_path = None
    if ids is not None:
        ids_path = output / "ids.txt"
        ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    sources = [REPO / name for name in ("lifemate_experiment.py", "lifemate_baseline.py", "lifemate_eval.py")]
    if any(name.startswith("workflow_") for name in variants):
        sources.append(REPO / "lifemate_workflows.py")
    summary = {
        "stage": args.stage,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_hash(manifest),
        "source_sha256": {source.name: file_hash(source) for source in sources},
        "variants": {},
    }
    if ids_path:
        summary["ids_sha256"] = file_hash(ids_path)
    if args.workflow_index:
        summary["workflow_index_sha256"] = file_hash(args.workflow_index)
    scores = {variant: run_variant(args, output, split, ids_path, variant, summary) for variant in variants}
    if len(scores) == 2:
        reference, candidate = variants
        reference_scores, candidate_scores = scores[reference], scores[candidate]
        differences = [candidate_scores[sample_id] - score for sample_id, score in reference_scores.items()]
        summary[f"{candidate}_vs_{reference}"] = {
            "delta_points": 100 * sum(differences) / len(differences),
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
        }
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    print(f"Finished. Share {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
