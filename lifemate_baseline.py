"""LifeMate audio-to-tool baseline for a remote CUDA runtime."""

import argparse
import importlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

SYSTEM = """You handle the last spoken Vietnamese request in the LifeMate simulator.
The input contains a transcript, earlier conversation, current time and user context,
and installed tool schemas. Earlier assistant calls have already been applied.
Use observations to resolve IDs and schema-defined arguments. Resolve relative dates
from the supplied time and timezone. Do not assume missing values. Use
sys.ask_missing_info when required details are missing and cannot be obtained with
tools. Finish if the request is complete or cannot be handled by the installed apps.
After a tool result, decide what remains to be done. Avoid repeating failed calls.
Return exactly one JSON object: {"name":"app.tool","arguments":{...}}
or {"finish":true}. Do not include other text."""

ASR_ID = "openai/whisper-large-v3-turbo"
AGENT_ID = "Qwen/Qwen3-4B"
CONTEXT_LIMIT = 16_384
NEW_TOKENS = 384


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Models:
    def __init__(self, asr_path, agent_path, *, offline):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Attach a GPU to the molab notebook before running.")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        options = {"local_files_only": offline}
        self.tokenizer = AutoTokenizer.from_pretrained(agent_path, **options)
        self.agent = AutoModelForCausalLM.from_pretrained(
            agent_path, torch_dtype=torch.float16, **options
        ).to("cuda").eval()
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=asr_path,
            device=0,
            torch_dtype=torch.float16,
            model_kwargs=options,
        )
        if self.agent.num_parameters() >= 5_000_000_000:
            raise ValueError("The agent exceeds the competition parameter limit.")
        if self.agent.num_parameters() + self.asr.model.num_parameters() >= 6_000_000_000:
            raise ValueError("Combined model parameters exceed the competition limit.")
        self.torch = torch

    def transcribe(self, path, seconds_left):
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd

        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        expected = self.asr.feature_extractor.sampling_rate
        if rate != expected:
            divisor = gcd(rate, expected)
            samples = resample_poly(samples, expected // divisor, rate // divisor)
        result = self.asr(
            {"raw": samples, "sampling_rate": expected},
            return_timestamps=len(samples) > expected * 30,
            generate_kwargs={
                "language": "vi", "task": "transcribe",
                "do_sample": False, "num_beams": 1, "max_time": seconds_left,
            },
        )
        return result["text"].strip()

    def next_action(self, messages, seconds_left):
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to("cuda")
        length = tokens["input_ids"].shape[-1]
        remaining = min(NEW_TOKENS, CONTEXT_LIMIT - length)
        if remaining <= 0:
            return None
        with self.torch.inference_mode():
            output = self.agent.generate(
                **tokens, do_sample=False, max_new_tokens=remaining,
                max_time=seconds_left,
            )
        return self.tokenizer.decode(output[0, length:], skip_special_tokens=True)


def audio_file(split_dir, relative_path):
    path = (split_dir / relative_path).resolve()
    if not path.is_relative_to(split_dir / "audio"):
        raise ValueError("Expected an audio path inside the split's audio folder.")
    return path


def infer(row, split_dir, tools, env_factory, models):
    start = time.perf_counter()
    deadline = start + 29
    env = env_factory(row["id"])
    context = row["context"]
    names = {name for app in context["installed_apps"].values() for name in app["tools"]}
    names.add("sys.ask_missing_info")
    schemas = [tools[name] for name in sorted(names) if name in tools]
    history = []
    status = "finished"
    for turn in row.get("history", []):
        if turn["role"] == "user":
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            transcript = models.transcribe(audio_file(split_dir, turn["audio"]), remaining)
            history.append({"role": "user", "text": transcript})
        else:
            history.append(turn)

    remaining = deadline - time.perf_counter()
    if remaining > 0:
        request = models.transcribe(audio_file(split_dir, row["audio"]), remaining)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json_text({
                "request": request, "history": history,
                "context": context, "tools": schemas,
            })},
        ]
        status = "step_limit"
        for _ in range(10):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            raw = models.next_action(messages, remaining)
            if raw is None:
                status = "context_limit"
                break
            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                action = None
            if action == {"finish": True}:
                status = "finished"
                break
            if (not isinstance(action, dict)
                    or not isinstance(action.get("name"), str)
                    or not isinstance(action.get("arguments"), dict)):
                result = {"error": "Return a tool name and arguments object or finish."}
            else:
                name, arguments = action["name"], action["arguments"]
                if time.perf_counter() >= deadline:
                    break
                validation = env.validate(name, arguments)
                result = env.call(name, arguments) if validation["ok"] else validation
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": json_text({"tool_result": result})},
            ])

    calls, state_hash = env.get_tool_calls(), env.get_state_hash()
    elapsed = time.perf_counter() - start
    if elapsed >= 29:
        status = "time_budget"
    return {
        "id": row["id"], "tool_calls": calls, "state_hash": state_hash,
    }, status, int(elapsed * 1000)


def run(data_root, split, output, *, limit=None, ids=None, asr_path=ASR_ID,
        agent_path=AGENT_ID, offline=False, include_time=False):
    root = Path(data_root).resolve()
    split_dir = root / split
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Save the result outside the dataset folder.")
    if not (split_dir / "zaic_tools" / "__init__.py").is_file():
        raise FileNotFoundError(f"Cannot find the official simulator in {split_dir}")
    with (split_dir / f"{split}.jsonl").open(encoding="utf-8") as manifest:
        rows = [json.loads(line) for line in manifest if line.strip()]
    if ids is not None:
        rows = [row for row in rows if row["id"] in ids]
        if {row["id"] for row in rows} != ids:
            raise ValueError("The requested IDs do not all exist in this split.")
    rows = rows[:limit]
    if not rows:
        raise ValueError("No examples selected.")

    # Each split has its own official package. Use a fresh process per split.
    loaded = sys.modules.get("zaic_tools")
    if loaded is not None and Path(loaded.__file__).resolve().parent != split_dir / "zaic_tools":
        raise RuntimeError("Use a fresh Python process when switching dataset splits.")
    sys.path.insert(0, str(split_dir))
    try:
        official = importlib.import_module("zaic_tools")
    finally:
        sys.path.pop(0)
    with (split_dir / f"{split}_tools_catalog.json").open(encoding="utf-8") as catalog:
        tools = official.catalog_tools(json.load(catalog))

    def env_factory(sample_id):
        return official.ToolEnv(split, sample_id, root=root)

    load_started = time.perf_counter()
    models = Models(asr_path, agent_path, offline=offline)
    print(f"Model loading took {time.perf_counter() - load_started:.1f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp",
        encoding="utf-8", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for index, row in enumerate(rows, 1):
            result, status, milliseconds = infer(row, split_dir, tools, env_factory, models)
            if include_time:
                result["time"] = milliseconds
            stream.write(json_text(result) + "\n")
            print(f"{index}/{len(rows)} {row['id']} {status} {milliseconds}ms", flush=True)
    temporary.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "public_test", "private_test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", type=Path, help="Optional file with one sample ID per line")
    parser.add_argument("--asr-path", default=ASR_ID)
    parser.add_argument("--agent-path", default=AGENT_ID)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--include-time", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.ids is not None:
        args.ids = set(args.ids.read_text(encoding="utf-8").split())
    print(run(**vars(args)))
