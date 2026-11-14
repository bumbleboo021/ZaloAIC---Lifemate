# LifeMate Qwen3.5 step 150

Inference uses Qwen3.5-4B (full base checkpoint approximately 4.66B parameters)
with a merged rank-16 LoRA and PhoWhisper-medium (approximately 0.764B) with
one merged ASR LoRA epoch. vLLM serves the agent in text-only mode over
container loopback with a 16,384-token cap. The agent transcribes audio and
history, chooses one tool at a time, and records the official ToolEnv calls
and final state hash. The two judge entry points share the inference module.

## Training reproduction

Use the official LifeMate TRAIN kit from the organizer. The preparation module
excludes the checked-in development and holdout IDs, leaving 1,158 requests
and 3,801 next-action or finish targets. Keep seed 42. See the pinned revisions below for the model sources and the main repository for evaluation.
Do not include restricted dataset files in the Docker image.

- Qwen3.5 base revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
- PhoWhisper base revision: 55a7e3eb6c906de891f8f06a107754427dd3be79.
- Prepare targets with training_code.prepare_lifemate_sft. Train Qwen3.5
  with training_code.train_lifemate_lora, model family qwen35, 12,288-token
  cap, and the epoch learning-rate schedule. The first run reached step 72
  before its 45-minute cap. To reproduce the measured artifact, use the
  saved step-72 adapter. Then initialize from it with a fresh optimizer,
  max-steps 78, and the same fitting data. Merge with
  training_code.merge_lifemate_qwen35.
- Train the ASR with training_code.train_lifemate_asr_lora, PhoWhisper
  selected explicitly, one epoch, rank 16, and the fitting audio only.
  Preserve its merged weights and processor together.

The fixed 48-example pilot scored 29.34. The 28.42 public result refers
to the earlier Qwen3.5 step-72 submission.

## Build inputs

Stage the two merged model directories under staged_models/agent and
staged_models/asr. Build from the repository root with
docker build -f judge_step150/Dockerfile -t zac2026-teamname:qwen35-step150 .
See docs/submit_qwen35_step150.md for the offline test and image export.
