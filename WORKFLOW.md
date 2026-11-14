# LifeMate project workflow

This is the record of how we took a Vietnamese voice assistant from its first public submission (**13.55**) to **38.97 reported public**. It covers the decisions behind the runs, what slowed us down, and how we worked together. See [README.md](README.md) and [TODO.md](TODO.md) for operational details.

## Starting point

We began by reading the signed-in LifeMate track and the judge materials. Each sample starts with Vietnamese audio, conversation history, and the installed tools. The solution transcribes the user turns, gives Qwen the request and context, and lets it propose one action at a time. The official `ToolEnv` validates and executes those actions. The output records the actual calls and final state hash. An incorrect state can wipe out credit even when some calls are right.

You chose Molab for GPU work. We built the baseline with Whisper large-v3-turbo and Qwen3-4B, then created fixed TRAIN partitions for iteration: 1,158 fitting requests, 232 development requests, and 155 holdout requests. The 48-request pilot screened changes before the full development run. The 634 public requests were reserved for generating leaderboard predictions.

## Decisions that changed the result

| Stage | Decision and result |
| --- | --- |
| First baseline → Qwen3 LoRA | Started at **13.55 public**. Trained the agent on official next-action examples and used chronological history. Qwen3 step 362 with Whisper turbo reached **23.91 public**. |
| ASR fine-tuning | Kept the Qwen3 agent fixed and trained PhoWhisper-medium and Whisper turbo separately. Tuned PhoWhisper scored **33.59 development** vs. **32.57** for tuned turbo, and reached **28.16 public**. |
| Qwen3.5 trial → training milestones | Held tuned PhoWhisper fixed. The step-72 Qwen3.5 trial reached **28.42 public**. You proposed keeping checkpoints during training so we could export and compare them later. Step 150 reached **32.00 public**. Steps 150, 362, and 548 scored **32.77, 37.80, and 40.07** on the same 232-request development set. |
| Step 548 becomes the fallback | Public scores of **33.00, 36.00, and 38.97** were reported in submission order. The export labels are 232, 362, and 548. The reported labels for 33.00 and 38.97 were 252 and 538, so match those two scores to the uploaded filenames before citing exact checkpoint results. Step 548 completed the fitting pass and led the fixed development comparison. |
| Use all labeled TRAIN data | You asked to start a separate candidate from step 548 with a fresh optimizer and one pass over **5,086 targets from all 1,545 TRAIN requests**. This includes the earlier development and holdout rows, so use its public result to compare it with step 548. You will choose the Docker candidate after seeing that result. |

## Bottlenecks we found

- **Transcription errors changed the requested action.** Replacing the latest transcript with the official text recovered **30 of 48 selected zero-score cases**. That was a diagnostic on failures. On the separate 48-request audio pilot, beam-3 scored **36.76** vs. **40.54** for greedy, so we kept greedy PhoWhisper.
- **The agent could still pick the wrong argument with a correct transcript.** In 15 of the 18 remaining diagnostic failures, the first ordered call difference was an argument. Five involved the funding source for a batch of transfers. A prompt asking the agent to check the combined balance recovered **0 of 5** and lowered the pilot from **40.54 to 39.91**. Ordered comparisons can also flag valid alternate call orders.
- **A pilot gain did not carry over to development.** We screened all 1,158 fitting transcripts, accepting 41 variants that added 107 targets to 3,801 clean targets. The fast step-608 run reached **41.46** on the pilot, then fell to **37.84 development** vs. **40.07** for step 548. Wrong first tool choices rose from 38 to 45. The evidence favored keeping step 548 for submissions.
- **Training speed limited how many experiments we could run.** A Molab build of `causal-conv1d` initially failed at link time because the CUDA toolchain lacked the unversioned `libcudart.so` link. After fixing it, both `causal-conv1d` and `flash-linear-attention` passed CUDA forward and backward checks. The matched 60-update run fell from **39.1 to 10.9 minutes**, about **3.6× faster**.
- **Runtime expiry and Docker startup needed separate checks.** We saved adapters and reports to the private Hugging Face repo and kept resumable checkpoints with optimizer state when available. The step-150 judge image stopped during vLLM startup and the server log was inaccessible. The newer package prints the log tail in the judge-visible output. A CPU package check cannot verify GPU startup, latency, or notebook parity.

## How we worked together

You supplied portal and Molab access, the official kit location, and the judge guide. You set the constraints and made the release decisions: use Molab for training and inference, keep milestone checkpoints, back up completed outputs, and let the public result decide between step 548 and the full-TRAIN candidate. You handled leaderboard uploads and reported the scores.

I translated those decisions into versioned scripts, ran the GPU jobs on Molab, and gave you log paths and completion markers so you could check progress and tell me when a run finished. I compared each result with the saved control, validated the public JSONL, and saved checkpoints and aggregate results as appropriate. Raw transcripts and fitting targets stayed in the controlled workspace. We queued evaluations and backups behind training so they could run while the Molab runtime remained active.

For the Docker handoff, Molab stages the exact merged weights. The laptop pulls the checksum-verified files, then its local Docker Desktop daemon builds and exports the Linux image. This uses the laptop for packaging only. The judge package includes the inference entry point, timing notebook, model manifest, and training reproduction notes.

## Release status at this handoff

- **Step 548** remains the fallback. The Docker build started by transferring the merged weights from Molab. Check the build logs for `DOCKER_ARCHIVE_READY`.
- **Full TRAIN** is a separate candidate. Check its Molab log for `FULL_TRAIN_PUBLIC_READY`, which marks the validated public JSONL and adapter backup. Compare its leaderboard score with step 548 before choosing the Docker image.
- The public submission is a **JSONL**. The judge's **Test Solution** form takes the **Docker archive's Google Drive link and the 32-character MD5 of that exact archive**. The GPU startup and notebook checks still need judge-compatible Docker execution.

Related files: [project diagram](docs/lifemate_overview.tex), [Docker handoff](docs/submit_qwen35_judge.md), [judge package](judge_qwen35/README.md), and the [private Hugging Face backups](https://huggingface.co/datasets/bumbleboonq/zaic-lifemate).
