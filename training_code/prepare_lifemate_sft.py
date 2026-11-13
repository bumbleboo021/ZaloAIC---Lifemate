"""Prepare fitting-only next-action examples from the official LifeMate TRAIN kit."""

import argparse
import json
from pathlib import Path

from lifemate_baseline import ASK_TOOL, NATIVE_TOOL_INSTRUCTIONS, READ_REUSE_INSTRUCTIONS
from lifemate_baseline import chronological_messages, json_text, load_dataset


def examples(row, catalog):
    installed = {name for app in row["context"]["installed_apps"].values() for name in app["tools"]}
    installed.add(ASK_TOOL)
    tools = [{"type": "function", "function": {key: catalog[name][key]
              for key in ("name", "description", "parameters")}} for name in sorted(installed & catalog.keys())]
    history = [{"role": "user", "text": turn["audio_text"]} if turn["role"] == "user" else turn
               for turn in row["history"]]
    messages = chronological_messages(NATIVE_TOOL_INSTRUCTIONS + READ_REUSE_INSTRUCTIONS,
                                      {"context": row["context"], "history": history}, row["audio_text"])
    calls, results = (row["label"][key] for key in ("tool_calls", "tool_results"))
    if len(calls) != len(results):
        raise ValueError(f"Mismatched calls and observations for {row['id']}")
    for call, result in zip(calls, results, strict=True):
        target = {"role": "assistant", "content": "", "tool_calls": [{"function": call}]}
        yield {"id": row["id"], "tools": tools, "prompt": messages.copy(), "completion": [target]}
        messages.extend([target, {"role": "tool", "content": json_text(result)}])
    yield {"id": row["id"], "tools": tools, "prompt": messages,
           "completion": [{"role": "assistant", "content": "Kết thúc."}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-ids", type=Path, help="Optional new audit partition to exclude before fitting")
    args = parser.parse_args()
    root = args.data_root.resolve()
    if args.output.resolve().is_relative_to(root):
        parser.error("Save the restricted output outside the official kit.")
    validation = Path(__file__).resolve().parents[1] / "validation"
    excluded = set()
    for name, expected in (("development", 232), ("holdout", 155)):
        ids = set((validation / f"{name}.txt").read_text(encoding="utf-8").split())
        if len(ids) != expected or ids & excluded:
            parser.error(f"Invalid {name} IDs for the v3.2.0 kit.")
        excluded.update(ids)
    if args.audit_ids:
        excluded.update(args.audit_ids.read_text(encoding="utf-8").split())
    rows, _, catalog, _ = load_dataset(root, "train")
    train_ids = {row["id"] for row in rows}
    if len(rows) != 1545 or len(train_ids) != len(rows) or not excluded <= train_ids:
        parser.error("Expected the official 1,545 unique TRAIN IDs.")
    count = 0
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            if row["id"] in excluded:
                continue
            for example in examples(row, catalog):
                stream.write(json_text(example) + "\n")
                count += 1
    print(f"Prepared {count} next-action and finish targets from {len(train_ids - excluded)} fitting requests.")


if __name__ == "__main__":
    main()
