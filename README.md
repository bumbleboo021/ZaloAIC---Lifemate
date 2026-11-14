# LifeMate · Zalo AI Challenge 2026

LifeMate turns a Vietnamese voice request and its conversation history into tool calls. We improved the public score from **13.55** to **38.97 reported** by fine-tuning the tool agent, switching to tuned PhoWhisper, and training Qwen3.5 through checkpoint 548. [Project history](WORKFLOW.md) · [Workflow diagram](docs/lifemate_overview.tex)

## What we built

**Vietnamese audio → PhoWhisper-medium → Qwen3.5-4B → official ToolEnv → JSONL.** The agent reads the request, previous turns, context, and installed tool schemas. It proposes one action at a time, receives the tool result, and exports the actual call trace and final state hash. vLLM serves the agent offline with thinking disabled. PhoWhisper uses greedy decoding.

| Candidate | Public score | Development score |
| --- | ---: | ---: |
| First Whisper turbo + Qwen3 baseline | 13.55 | — |
| Qwen3 LoRA step 362 + Whisper turbo | 23.91 | 23.10 |
| Qwen3 LoRA + tuned PhoWhisper | 28.16 | 33.59 |
| Qwen3.5 step 150 + tuned PhoWhisper | 32.00 | 32.77 |
| Qwen3.5 step 362 + tuned PhoWhisper | 36.00 | 37.80 |
| Qwen3.5 step 548 + tuned PhoWhisper | 38.97* | **40.07** |

Public scores were reported by the team. *The highest reported score was labeled step 538 in chat and the corresponding export series ends at step 548. Verify the uploaded filename before attributing 38.97 to that exact checkpoint.*

A reference transcript recovered **30 of 48 selected audio failures**. Beam search and a targeted transfer prompt lost to the greedy control in the pilot. Continuing with screened ASR transcripts improved the pilot but slipped to **37.84** on development. Flash Linear Attention and causal-conv1d cut the matched 60-update training run from **39.1 to 10.9 minutes**.

## Release candidates

- **Step 548** is the fallback. It completed a fitting pass over 1,158 official requests and led the fixed 232-request development comparison.
- **Full TRAIN** starts from step 548 and adds one pass over 5,086 clean targets from all 1,545 labeled TRAIN requests. It completed 636 more updates, reaching cumulative **step 1184**. The former development and holdout rows were included in fitting, so compare this candidate using the public leaderboard.
- The public submission is a 634-row JSONL. The separate judge submission is the frozen offline Docker image archive, with a Drive link and MD5. The timing notebook and the shell entry point share inference code. [Judge package](judge_qwen35/README.md) · [Build and upload guide](docs/submit_qwen35_judge.md)

## Repository layout

| Path | Purpose |
| --- | --- |
| `lifemate_baseline.py`, `lifemate_eval.py`, `lifemate_phase2.py` | Audio-to-tool pipeline, scorer, and evaluation loop |
| [`experiments/`](experiments/) | Model comparisons, the full-TRAIN run, and public export |
| [`training_code/`](training_code/) | Data preparation, LoRA training, merge, and backup |
| [`validation/`](validation/) | Fixed TRAIN development and holdout ID lists |
| [`judge_qwen35/`](judge_qwen35/) | Current Docker recipe, inference entry point, and reproduction notes |
| [`judge_step150/`](judge_step150/) | Earlier judge package and shared timing notebook/dependency inputs |
| [`docs/`](docs/) | Workflow diagram and submission instructions |

The root Dockerfile and prediction scripts are the earlier Qwen3 baseline. Use `judge_qwen35/Dockerfile` for the Qwen3.5 release. Train and run model evaluation on Molab. Docker Desktop builds the Linux image locally from the verified model files staged on Molab.

GitHub holds the source and documentation. The official kit, model weights, checkpoints, transcripts, predictions, build archives, credentials, and local logs live outside Git. Completed release artifacts are backed up in the team's private Hugging Face dataset.
