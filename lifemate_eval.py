"""Create fixed TRAIN splits and score LifeMate predictions without loading models."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from statistics import mean
import sys
import unicodedata


def read_rows(path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError(f"Duplicate sample ID in {path}: {row['id']}")
                rows[row["id"]] = row
    return rows


def normalize(text):
    return unicodedata.normalize("NFC", text.lower())


def make_splits(rows, output):
    # Keep repeated requests together and balance by reference trace length.
    groups = defaultdict(list)
    for row in rows.values():
        key = " ".join(normalize(row["audio_text"]).split())
        groups[key].append(row)

    strata = defaultdict(list)
    for group in groups.values():
        bucket = min(len(group[0]["label"]["tool_calls"]), 3)
        strata[bucket].append(group)

    rng = random.Random(42)
    development, holdout = [], []
    for bucket in sorted(strata):
        groups = sorted(strata[bucket], key=lambda group: group[0]["id"])
        rng.shuffle(groups)
        size = sum(map(len, groups))
        dev_target = round(size * 0.15)
        holdout_target = round(size * 0.10)
        counts = [0, 0]
        for group in groups:
            if counts[0] < dev_target:
                development.extend(row["id"] for row in group)
                counts[0] += len(group)
            elif counts[1] < holdout_target:
                holdout.extend(row["id"] for row in group)
                counts[1] += len(group)
            else:
                break

    output.mkdir(parents=True, exist_ok=True)
    for name, ids in (("development", development), ("holdout", holdout)):
        path = output / f"{name}.txt"
        contents = "".join(f"{sample_id}\n" for sample_id in sorted(ids))
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise FileExistsError(f"Existing split differs: {path}")
        path.write_text(contents, encoding="utf-8")
        print(f"{name}: {len(ids)} IDs in {path}")


def token_f1(left, right):
    left = normalize(left).split()
    right = normalize(right).split()
    if not left or not right:
        return float(left == right)
    overlap = sum((Counter(left) & Counter(right)).values())
    return 2 * overlap / (len(left) + len(right))


def calls_match(left, right, fuzzy_fields):
    name = left["name"]
    if name != right["name"]:
        return False
    left_args, right_args = left["arguments"], right["arguments"]
    if left_args.keys() != right_args.keys():
        return False
    for key, value in left_args.items():
        expected = right_args[key]
        if key in fuzzy_fields.get(name, ()) and isinstance(value, str) and isinstance(expected, str):
            if token_f1(value, expected) <= 0.6:
                return False
        elif value != expected or isinstance(value, bool) != isinstance(expected, bool):
            return False
    return True


def count_matches(predicted, gold, fuzzy_fields):
    candidates = [
        [index for index, call in enumerate(gold) if calls_match(prediction, call, fuzzy_fields)]
        for prediction in predicted
    ]
    owners = {}

    def assign(prediction, visited):
        for index in candidates[prediction]:
            if index in visited:
                continue
            visited.add(index)
            if index not in owners or assign(owners[index], visited):
                owners[index] = prediction
                return True
        return False

    for prediction in range(len(predicted)):
        assign(prediction, set())
    return len(owners)


def score(data_root, rows, predictions, expected_ids):
    if not predictions or predictions.keys() != expected_ids:
        missing = expected_ids - predictions.keys()
        extra = predictions.keys() - expected_ids
        raise ValueError(f"Prediction IDs differ from evaluation set: {len(missing)} missing, {len(extra)} extra.")
    if expected_ids - rows.keys():
        raise ValueError("Evaluation IDs must be present in the TRAIN manifest.")

    sys.path.insert(0, str(data_root / "train"))
    try:
        from zaic_tools import ToolEnv, catalog_tools
    finally:
        sys.path.pop(0)
    catalog_path = data_root / "train/train_tools_catalog.json"
    tools = catalog_tools(json.loads(catalog_path.read_text(encoding="utf-8")))
    fuzzy_fields = {
        name: {key for key, field in tool["parameters"].get("properties", {}).items()
               if field.get("x-match") == "fuzzy"}
        for name, tool in tools.items()
    }

    details = []
    for sample_id, prediction in predictions.items():
        expected = rows[sample_id]["label"]["tool_calls"]
        actual = prediction["tool_calls"]
        if not isinstance(actual, list) or not re.fullmatch("[a-f0-9]{64}", prediction["state_hash"]):
            raise ValueError(f"Invalid submission row: {sample_id}")
        env = ToolEnv("train", sample_id, root=data_root)
        for call in expected:
            env.call(call["name"], call["arguments"])
        state_match = prediction["state_hash"] == env.get_state_hash()
        matched = count_matches(actual, expected, fuzzy_fields)
        call_f1 = 2 * matched / (len(actual) + len(expected)) if actual or expected else 1.0
        points = call_f1 if state_match or not expected else 0.0
        elapsed = prediction.get("time")
        if elapsed is not None and elapsed >= 30_000:
            points = 0.0
        details.append({
            "id": sample_id, "score": points, "state_match": state_match,
            "call_f1": call_f1, "gold_calls": len(expected), "predicted_calls": len(actual),
            "matches": matched, "time_ms": elapsed,
        })

    timings = [item["time_ms"] for item in details if item["time_ms"] is not None]
    return {
        "samples": len(details),
        "score": mean(item["score"] for item in details) * 100,
        "state_match_pct": mean(item["state_match"] for item in details) * 100,
        "call_f1_pct": mean(item["call_f1"] for item in details) * 100,
        "max_time_ms": max(timings, default=None),
        "details": details,
    }


def validate_public(root, predictions_path):
    expected = read_rows(root / "public_test/public_test.jsonl")
    predictions = read_rows(predictions_path)
    missing = expected.keys() - predictions.keys()
    extra = predictions.keys() - expected.keys()
    if not expected or missing or extra:
        raise ValueError(f"Public prediction coverage differs from manifest: {len(missing)} missing, {len(extra)} extra.")

    for sample_id, row in predictions.items():
        if row.keys() != {"id", "tool_calls", "state_hash"}:
            raise ValueError(f"Expected id, tool_calls and state_hash: {sample_id}")
        state_hash = row["state_hash"]
        if not isinstance(state_hash, str) or not re.fullmatch("[a-f0-9]{64}", state_hash):
            raise ValueError(f"Invalid state hash: {sample_id}")
        calls = row["tool_calls"]
        if not isinstance(calls, list):
            raise ValueError(f"Expected a tool call list: {sample_id}")
        for call in calls:
            if (not isinstance(call, dict) or call.keys() != {"name", "arguments"}
                    or not isinstance(call["name"], str) or not call["name"]
                    or not isinstance(call["arguments"], dict)):
                raise ValueError(f"Invalid tool call: {sample_id}")

    return {"samples": len(predictions), "coverage_and_format": "valid"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    splits = commands.add_parser("split")
    splits.add_argument("--output-dir", type=Path, required=True)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--ids", type=Path, required=True)
    evaluate.add_argument("--report", type=Path)
    public = commands.add_parser("validate-public", help="Check the three-field public JSONL format")
    public.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    if args.command == "validate-public":
        print(json.dumps(validate_public(root, args.predictions), indent=2))
        return
    rows = read_rows(root / "train/train.jsonl")
    if args.command == "split":
        make_splits(rows, args.output_dir)
        return
    report = score(root, rows, read_rows(args.predictions), set(args.ids.read_text(encoding="utf-8").split()))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
