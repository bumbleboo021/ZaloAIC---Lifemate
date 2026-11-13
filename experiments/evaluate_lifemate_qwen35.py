"""Run Qwen3.5 via local vLLM with the released PhoWhisper and LifeMate ToolEnv."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lifemate_baseline import CONTEXT_LIMIT, NEW_TOKENS, Models, json_text, load_dataset
from lifemate_eval import validate_public
from lifemate_experiment import file_hash, selection, write_json
from lifemate_phase2 import run_variant
from predict import OPTIONS


def api(url, path, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def chat_messages(messages, *, serialize_arguments=True):
    """Add OpenAI tool-call IDs to LifeMate's chronological messages."""
    normalized = []
    pending = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "tool":
            if pending:
                normalized.append(dict(message, tool_call_id=pending.pop(0)))
            else:
                normalized.append({"role": "user", "content": "Tool feedback:\n" + message["content"]})
            continue
        pending = []
        if message.get("tool_calls"):
            calls = []
            for offset, call in enumerate(message["tool_calls"]):
                key = f"call_{index}_{offset}"
                function = call["function"]
                calls.append({"id": key, "type": "function", "function": {
                    "name": function["name"],
                    "arguments": json_text(function["arguments"]) if serialize_arguments else function["arguments"],
                }})
                pending.append(key)
            message = dict(message, tool_calls=calls)
        normalized.append(message)
    return normalized


class VllmAgent:
    transcribe = Models.transcribe
    prediction_options = dict(OPTIONS, chronological_history=True)

    def __init__(self, url, agent, asr):
        self.url = url
        from transformers.utils import import_utils
        import_utils._torchvision_available = False
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(agent, local_files_only=True)
        processor = AutoProcessor.from_pretrained(asr, local_files_only=True)
        speech = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr, local_files_only=True, torch_dtype=torch.float16
        )
        self.asr = pipeline("automatic-speech-recognition", model=speech,
                            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                            device=0)

    def next_action(self, messages, seconds_left, *, native_tools=None, thinking=False, observe=None):
        start = monotonic()
        template_messages = chat_messages(messages, serialize_arguments=False)
        messages = chat_messages(messages)
        rendered = self.tokenizer.apply_chat_template(
            template_messages, tools=native_tools, add_generation_prompt=True,
            enable_thinking=False, tokenize=False,
        )
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False))
        if observe:
            observe("prompt", tokens=count)
        budget = min(NEW_TOKENS, CONTEXT_LIMIT - count)
        remaining = seconds_left - (monotonic() - start)
        if budget <= 0 or remaining <= 0:
            return None
        try:
            response = api(self.url, "/v1/chat/completions", {
                "model": "lifemate-agent", "messages": messages, "tools": native_tools,
                "tool_choice": "auto", "temperature": 0, "max_tokens": budget,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=remaining)
        except (TimeoutError, URLError) as error:
            if isinstance(error, HTTPError) or (
                isinstance(error, URLError) and not isinstance(error.reason, TimeoutError)
            ):
                raise
            return None
        choice = response["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls") or []
        if len(calls) == 1:
            try:
                function = calls[0]["function"]
                arguments = json.loads(function["arguments"])
                if isinstance(arguments, dict):
                    action = {"name": function["name"], "arguments": arguments}
                    return "<tool_call>" + json_text(action) + "</tool_call>"
            except (KeyError, TypeError, ValueError):
                pass
        if calls or choice["finish_reason"] == "length":
            return "<tool_call"
        content = (message.get("content") or "").strip()
        return content or "Done."


def run(args, url):
    models = VllmAgent(url, args.agent, args.asr)
    if args.stage == "public":
        from lifemate_baseline import write_predictions
        rows = load_dataset(args.data_root, "public_test")
        timed = args.output / "public_timed.jsonl"
        write_predictions(rows, timed, models, include_time=True, **models.prediction_options)
        path = args.output / "lifemate_submission_qwen35_4b.jsonl"
        pending = path.with_suffix(".partial")
        maximum = 0
        with timed.open(encoding="utf-8") as source, pending.open("x", encoding="utf-8") as stream:
            for line in source:
                row = json.loads(line)
                maximum = max(maximum, row.pop("time"))
                stream.write(json_text(row) + "\n")
        result = validate_public(args.data_root, pending)
        if result["samples"] != 634 or maximum >= 30_000:
            raise ValueError("Public coverage or timing check failed.")
        pending.replace(path)
        write_json(args.output / "summary.json",
                   dict(result, max_ms=maximum, sha256=file_hash(path), file=path.name))
        return

    if args.stage == "smoke":
        from training_code.cache_lifemate_fitting_asr import fitting_rows
        rows = fitting_rows(args.data_root, Path(__file__).resolve().parents[1] / "validation")
        multi = [row for row in rows if len(row["label"]["tool_calls"]) > 1][:6]
        empty = next(row for row in rows if not row["label"]["tool_calls"])
        ids = {row["id"] for row in [*multi, empty]}
    else:
        ids = set(selection(args.stage))
    dataset = load_dataset(args.data_root, "train", ids=ids)
    trace = args.output / "trace.private.jsonl"
    with trace.open("x", encoding="utf-8") as stream:
        report = run_variant(args.stage, args.data_root, args.output, dataset, models, ids,
                             stream, inference_options=models.prediction_options)
    if args.stage == "smoke":
        with trace.open(encoding="utf-8") as stream:
            records = [json.loads(line)["events"] for line in stream]
        round_trip = any(
            any(e["event"] == "response" for e in events[:i])
            and event["event"] == "proposal"
            and isinstance(event.get("action"), dict)
            for events in records for i, event in enumerate(events)
        )
        if not round_trip or not any(row["predicted_calls"] for row in report["details"]):
            raise RuntimeError("Smoke test did not exercise a valid tool response and next action.")
        if report["at_or_over_30s"] or report["stops"].get("time_budget"):
            raise RuntimeError("Smoke test hit the sample deadline.")
    write_json(args.output / "summary.json", {k: v for k, v in report.items() if k != "details"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "agent", "asr", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=("smoke", "pilot", "development", "public"))
    parser.add_argument("--port", type=int, default=18083)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True)
    os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_NO_USAGE_STATS="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    url = f"http://127.0.0.1:{args.port}"
    command = [
        str(Path(sys.executable).with_name("vllm")), "serve", str(args.agent),
        "--served-model-name", "lifemate-agent", "--host", "127.0.0.1", "--port", str(args.port),
        "--max-model-len", str(CONTEXT_LIMIT), "--language-model-only", "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--gpu-memory-utilization", "0.60", "--max-num-seqs", "1", "--enforce-eager",
    ]
    with (args.output / "server.log").open("x", encoding="utf-8") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = monotonic() + 600
            while monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited. See {log.name}")
                try:
                    api(url, "/v1/models")
                except (OSError, URLError):
                    sleep(2)
                else:
                    break
            else:
                raise TimeoutError(f"vLLM startup timed out. See {log.name}")
            warmup_tool = {"type": "function", "function": {
                "name": "lookup", "description": "Look up a record",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                               "required": ["query"]},
            }}
            api(url, "/v1/chat/completions", {
                "model": "lifemate-agent",
                "messages": [{"role": "user", "content": "Call lookup for the record alpha."}],
                "tools": [warmup_tool], "tool_choice": "auto", "max_tokens": 64,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=240)
            print("QWEN35_WARMUP_READY", flush=True)
            run(args, url)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    print("QWEN35_STAGE_READY", args.stage, flush=True)


if __name__ == "__main__":
    main()

