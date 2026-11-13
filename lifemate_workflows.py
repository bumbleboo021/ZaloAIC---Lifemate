"""Build and retrieve tool-order examples from the LifeMate fitting partition."""

import argparse
from collections import Counter
import json
from math import log, sqrt
from pathlib import Path
import re
import unicodedata


def tokens(text):
    words = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text.casefold()))
    return [*words, *(f"{a} {b}" for a, b in zip(words, words[1:]))]


class WorkflowIndex:
    def __init__(self, path):
        examples = json.loads(path.read_text(encoding="utf-8"))
        features = [Counter(tokens(example["request"])) for example in examples]
        frequency = Counter(term for vector in features for term in vector)
        self.idf = {term: 1 + log((len(examples) + 1) / (count + 1))
                    for term, count in frequency.items()}
        self.examples = [(example, self.vector(feature)) for example, feature in zip(examples, features)]

    def vector(self, feature):
        weighted = {term: count * self.idf[term] for term, count in feature.items() if term in self.idf}
        norm = sqrt(sum(weight ** 2 for weight in weighted.values()))
        return {term: weight / norm for term, weight in weighted.items()} if norm else {}

    def retrieve(self, request, installed_tools, *, limit=3):
        query = self.vector(Counter(tokens(request)))
        ranked = []
        for example, vector in self.examples:
            if any(tool not in installed_tools for tool in example["tools"]):
                continue
            similarity = sum(weight * vector.get(term, 0) for term, weight in query.items())
            if similarity >= 0.24:
                ranked.append((similarity, example))
        chosen = []
        seen = set()
        for _, example in sorted(ranked, key=lambda item: item[0], reverse=True):
            signature = tuple(example["tools"])
            if signature in seen:
                continue
            chosen.append(example)
            seen.add(signature)
            if len(chosen) == limit:
                break
        return chosen


def build_index(data_root, validation, output):
    excluded = set()
    for name, expected in (("development", 232), ("holdout", 155)):
        ids = set((validation / f"{name}.txt").read_text(encoding="utf-8").split())
        if len(ids) != expected or ids & excluded:
            raise ValueError(f"Invalid {name} split for the official v3.2.0 kit.")
        excluded.update(ids)
    examples = []
    seen_ids = set()
    manifest = data_root / "train/train.jsonl"
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate TRAIN ID: {sample_id}")
            seen_ids.add(sample_id)
            if sample_id in excluded:
                continue
            examples.append({"request": row["audio_text"],
                             "tools": [call["name"] for call in row["label"]["tool_calls"]]})
    if excluded - seen_ids or not examples:
        raise ValueError("The evaluation splits do not match the TRAIN manifest.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(examples, stream, ensure_ascii=False)
    print(f"Wrote {len(examples)} fitting examples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=Path(__file__).parent / "validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_index(args.data_root, args.validation, args.output)
