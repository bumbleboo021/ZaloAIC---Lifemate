"""LifeMate audio-to-tool baseline for a remote CUDA runtime."""

import argparse
import importlib
import json
import os
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

READ_REUSE_INSTRUCTIONS = """
Use the most recent successful observation for a read when no action has changed
its inputs. A read result is evidence, not a request to repeat the same read.
When a write has completed the requested change, return finish.
"""

CLARIFICATION_INSTRUCTIONS = """
Before asking for clarification, check the request, conversation history, user
context, and tool observations. If a user names a person, chat, bill, beneficiary,
or device and a relevant read tool can resolve it, use that read tool first.
For example, when a person is named but a contact ID is required, search for
that name and use the returned ID if the match is unique. If the user has not
given a required value such as a transfer amount and no observation supplies
it, ask for that value. Ask once, using the intended installed tool's exact
name and the unresolved parameter names in params. A field can be necessary
for this particular request even if the schema marks it optional, such as
contact_id when the request names a contact and no phone number is supplied.
"""

NATIVE_TOOL_INSTRUCTIONS = """You handle the last spoken Vietnamese request in LifeMate.
Use the request, conversation history, current time and user context. Earlier
assistant calls have already been applied. Choose a relevant tool from the
provided signatures. Call one tool at a time and use its actual response to
decide the next action. Use read tools to find required IDs and details.
Ask for missing information only when the request and available observations
cannot supply it. Finish with a short plain-text response when the request
is complete, requires no tool, or is outside the installed apps.
"""

LOOKUP_GUIDANCE = """When an action needs an app ID, find it before the action.
- For a phone call or SMS to a named person or relation, use
  contacts.search({"query": "<name or relation>"}), then use the returned
  contact_id when there is a unique match. Use a supplied phone number directly.
- For a message to a named chat, use chat_ott.list_chats to resolve chat_id.
- For a described smart home device, use smart_home.get_status to resolve device_id.
- For a described bill, use bill_payment.list_bills to resolve bill_id.
Use only tools that are provided. Skip a lookup if the current request or an
observation already supplies the necessary ID. Use names as search queries.
Do not invent IDs or act on an ambiguous match.
"""

BILL_PAYMENT_INSTRUCTIONS = """
For bill_payment.pay, match each requested bill to its bill_id from the
request or a bill_payment.list_bills observation. Match the described bill
type and provider when available. Use the requested payment source and
honor any stated condition about checking funds or choosing a fallback.
For multiple bills, keep each bill ID paired with its intended source.
"""

WORKFLOW_INSTRUCTIONS = """
Similar training requests may include a tool sequence. Use it to identify
which reads might provide missing IDs or details before a write. Example
values belong to those requests. Use the current request and observations to
choose actions and arguments. An empty sequence means no tool was needed in
that example. It does not decide what to do for the current request.
"""

ASK_TOOL = "sys.ask_missing_info"

ASR_ID = "openai/whisper-large-v3-turbo"
AGENT_ID = "Qwen/Qwen3-4B"
CONTEXT_LIMIT = 16_384
NEW_TOKENS = 384


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Models:
    def __init__(self, asr_path, agent_path, *, offline):
        # Molab can have Keras 3 installed alongside PyTorch.
        os.environ["USE_TF"] = "0"
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline

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
        processor = AutoProcessor.from_pretrained(asr_path, **options)
        speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_path, torch_dtype=torch.float16, **options
        )
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=speech_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0,
        )
        if self.agent.num_parameters() >= 5_000_000_000:
            raise ValueError("The agent exceeds the competition parameter limit.")
        if self.agent.num_parameters() + self.asr.model.num_parameters() >= 6_000_000_000:
            raise ValueError("Combined model parameters exceed the competition limit.")
        self.torch = torch

    def transcribe(self, path, seconds_left, *, num_beams=1):
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
                "do_sample": False, "num_beams": num_beams, "max_time": seconds_left,
            },
        )
        return result["text"].strip()

    def next_action(self, messages, seconds_left, *, native_tools=None, thinking=False, observe=None):
        tool_options = {"tools": native_tools} if native_tools is not None else {}
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=thinking, return_dict=True, return_tensors="pt", **tool_options,
        ).to("cuda")
        length = tokens["input_ids"].shape[-1]
        if observe is not None:
            observe("prompt", tokens=int(length))
        remaining = min(1024 if thinking else NEW_TOKENS, CONTEXT_LIMIT - length)
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


def parse_action(raw, native, thinking=False):
    if thinking:
        _, separator, raw = raw.partition("</think>")
        if not separator:
            return None
    if native:
        tag = "<tool_call>"
        start = raw.find(tag)
        if start == -1:
            return None if "<tool_call" in raw or not raw.strip() else {"finish": True}
        payload = raw[start + len(tag):].lstrip()
        try:
            action, end = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            return None
        return action if payload[end:].lstrip().startswith("</tool_call>") else None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def clarification_error(arguments, installed_names, tools):
    target = arguments.get("tool_name")
    if not isinstance(target, str) or target == ASK_TOOL or target not in installed_names or target not in tools:
        return "Choose the installed tool that would fulfill the request."
    properties = tools[target]["parameters"].get("properties", {})
    params = arguments.get("params")
    if (not isinstance(params, list) or not params
            or any(not isinstance(param, str) or param not in properties for param in params)):
        return "List the unresolved parameter names from that tool's schema."
    if len(set(params)) != len(params):
        return "List each unresolved parameter once."
    return None


def chronological_messages(system, prompt_input, request):
    context = {key: value for key, value in prompt_input.items() if key not in {"history", "request"}}
    messages = [{"role": "system", "content": system + "\nContext: " + json_text(context)}]
    for turn in prompt_input["history"]:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn.get("text", "")})
        else:
            assistant = {"role": "assistant", "content": turn.get("text") or ""}
            calls = turn.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = [{"function": call} for call in calls]
            messages.append(assistant)
            for result in turn.get("tool_results") or []:
                messages.append({"role": "tool", "content": json_text(result)})
    messages.append({"role": "user", "content": "Latest Vietnamese request. Handle this turn only:\n" + json_text(request)})
    return messages


def infer(row, split_dir, tools, env_factory, models, *, reuse_reads=False, resolve_missing=False, workflows=None, native_tool_format=False, thinking=False, lookup_guidance=False, request_last=False, chronological_history=False, bill_payment_guidance=False, request_asr_beams=1, observe=None):
    start = time.perf_counter()
    deadline = start + 29
    env = env_factory(row["id"])
    context = row["context"]
    names = {name for app in context["installed_apps"].values() for name in app["tools"]}
    names.add(ASK_TOOL)
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
        request_audio = audio_file(split_dir, row["audio"])
        if request_asr_beams == 1:
            request = models.transcribe(request_audio, remaining)
        else:
            request = models.transcribe(request_audio, remaining, num_beams=request_asr_beams)
        system = NATIVE_TOOL_INSTRUCTIONS if native_tool_format else SYSTEM
        if reuse_reads:
            system += READ_REUSE_INSTRUCTIONS
        if resolve_missing:
            system += CLARIFICATION_INSTRUCTIONS
        if lookup_guidance:
            system += "\n" + LOOKUP_GUIDANCE
        if bill_payment_guidance and "bill_payment.pay" in names:
            system += BILL_PAYMENT_INSTRUCTIONS
        prompt_input = {"request": request, "history": history, "context": context, "tools": schemas}
        native_tools = None
        if native_tool_format:
            native_tools = [{"type": "function", "function": {
                "name": schema["name"], "description": schema.get("description", ""),
                "parameters": schema["parameters"],
            }} for schema in prompt_input.pop("tools")]
        if workflows is not None:
            examples = workflows.retrieve(request, names)
            if examples:
                system += WORKFLOW_INSTRUCTIONS
                prompt_input["similar_training_workflows"] = examples
        if request_last:
            prompt_input.pop("request")
            user_message = (
                "Context and earlier turns (reference only):\n" + json_text(prompt_input)
                + "\nLatest Vietnamese request. Handle this turn only:\n" + json_text(request)
            )
        else:
            user_message = json_text(prompt_input)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        if chronological_history:
            messages = chronological_messages(system, prompt_input, request)
        if observe is not None:
            observe("transcripts", request=request, history=[turn["text"] for turn in history if turn["role"] == "user"])
        status = "step_limit"
        previous_read = None
        for _ in range(10):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if observe is not None:
                raw = models.next_action(messages, remaining, native_tools=native_tools, thinking=thinking, observe=observe)
            elif native_tool_format:
                raw = models.next_action(messages, remaining, native_tools=native_tools, thinking=thinking)
            else:
                raw = models.next_action(messages, remaining)
            if raw is None:
                status = "context_limit"
                break
            action = parse_action(raw, native_tool_format, thinking)
            if observe is not None:
                observe("proposal", raw=raw, action=action)
            if action == {"finish": True}:
                status = "finished"
                break
            is_call = (isinstance(action, dict) and isinstance(action.get("name"), str)
                       and isinstance(action.get("arguments"), dict))
            if not is_call:
                result = {"error": "Return one valid tool call or finish." if native_tool_format
                          else "Return a tool name and arguments object or finish."}
            else:
                name, arguments = action["name"], action["arguments"]
                if time.perf_counter() >= deadline:
                    break
                validation = env.validate(name, arguments)
                if observe is not None and not validation["ok"]:
                    observe("validation_error", result=validation)
                if not validation["ok"]:
                    result = validation
                elif resolve_missing and name == ASK_TOOL:
                    error = clarification_error(arguments, names, tools)
                    if error:
                        result = {"ok": False, "error": error}
                    else:
                        result = env.call(name, arguments)
                        if result.get("waiting_for_user") is True:
                            status = "waiting_for_user"
                            break
                        previous_read = None
                elif reuse_reads and tools[name]["kind"] == "read":
                    signature = (name, arguments)
                    if previous_read is not None and signature == previous_read[0]:
                        result = {
                            "observation": previous_read[1],
                            "note": "This read already ran. Use the observation to decide the next action.",
                        }
                    else:
                        result = env.call(name, arguments)
                        previous_read = (signature, result) if result.get("ok") else None
                else:
                    result = env.call(name, arguments)
                    previous_read = None
            if observe is not None:
                observe("response", result=result)
            if native_tool_format:
                assistant = {"role": "assistant", "content": "" if thinking else raw}
                if is_call:
                    assistant = {"role": "assistant", "content": "", "tool_calls": [{"function": action}]}
                messages.extend([assistant, {"role": "tool", "content": json_text(result)}])
            else:
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": json_text({"tool_result": result})},
                ])

    calls, state_hash = env.get_tool_calls(), env.get_state_hash()
    elapsed = time.perf_counter() - start
    if elapsed >= 29:
        status = "time_budget"
    if observe is not None:
        observe("stop", reason=status)
    return {
        "id": row["id"], "tool_calls": calls, "state_hash": state_hash,
    }, status, int(elapsed * 1000)


def load_dataset(data_root, split, *, limit=None, ids=None):
    root = Path(data_root).resolve()
    split_dir = root / split
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

    return rows, split_dir, tools, env_factory


def write_predictions(dataset, output, models, *, include_time=False, **options):
    rows, split_dir, tools, env_factory = dataset
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp",
        encoding="utf-8", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for index, row in enumerate(rows, 1):
            result, status, milliseconds = infer(
                row, split_dir, tools, env_factory, models, **options
            )
            if include_time:
                result["time"] = milliseconds
            stream.write(json_text(result) + "\n")
            print(f"{index}/{len(rows)} {row['id']} {status} {milliseconds}ms", flush=True)
    temporary.replace(output)
    return output


def run(data_root, split, output, *, limit=None, ids=None, asr_path=ASR_ID,
        agent_path=AGENT_ID, offline=False, include_time=False, reuse_reads=False,
        resolve_missing=False, workflow_index=None, native_tool_format=False, thinking=False, lookup_guidance=False, request_last=False):
    root = Path(data_root).resolve()
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Save the result outside the dataset folder.")
    dataset = load_dataset(root, split, limit=limit, ids=ids)
    workflows = None
    if workflow_index is not None:
        from lifemate_workflows import WorkflowIndex

        workflows = WorkflowIndex(workflow_index)
    load_started = time.perf_counter()
    models = Models(asr_path, agent_path, offline=offline)
    print(f"Model loading took {time.perf_counter() - load_started:.1f}s", flush=True)
    return write_predictions(
        dataset, output, models, include_time=include_time, reuse_reads=reuse_reads,
        resolve_missing=resolve_missing, workflows=workflows, native_tool_format=native_tool_format,
        thinking=thinking, lookup_guidance=lookup_guidance, request_last=request_last,
    )


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
    parser.add_argument("--reuse-reads", action="store_true", help="Try the repeated-read experiment")
    parser.add_argument("--resolve-missing", action="store_true", help="Try lookup before clarification and stop when waiting")
    parser.add_argument("--workflow-index", type=Path, help="Fitting-only workflow examples")
    parser.add_argument("--native-tool-format", action="store_true", help="Use the Qwen tool template")
    parser.add_argument("--thinking", action="store_true", help="Try Qwen reasoning with the native tool format")
    parser.add_argument("--lookup-guidance", action="store_true", help="Try explicit lookup steps for app IDs")
    parser.add_argument("--request-last", action="store_true", help="Place the current request after the history and context")
    args = parser.parse_args()
    if args.thinking and not args.native_tool_format:
        parser.error("--thinking requires --native-tool-format")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.ids is not None:
        args.ids = set(args.ids.read_text(encoding="utf-8").split())
    print(run(**vars(args)))
