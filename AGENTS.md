# LifeMate handoff

Read this file first. It is the single operational handoff. Open only the source file or log needed for the current task. Treat `experiments/`, `judge_step150/`, and the project history as references to consult when a specific result, dependency, or reproduction step is needed.

## Current release

- **Step 1184 scored 40.72 public.** It is Qwen3.5-4B with tuned, greedy PhoWhisper-medium, offline vLLM, and the official ToolEnv. It adds one pass over all 1,545 TRAIN requests (5,086 targets, 636 updates) from step 548. Former development and holdout rows were included in fitting. Use those slices for diagnostics only with step 1184.
- The original 634-row public JSONL, adapter, metrics, and model manifest are in the [private Hugging Face release](https://huggingface.co/datasets/bumbleboonq/zaic-lifemate/tree/main/qwen35-full-train/from-step548). Original JSONL SHA-256: `3fb0745f18bb5acb2314ee81107aa676353832adc31cf35a85ae9df9c4ddd6b8`. Step 548 remains the fallback and the control for independent comparisons on the original holdout.
- **Docker:** the archived original under results/judge_qwen35/step1184/ (MD5 dcbe873264676fbad978c5d8bdbd0b9b) failed the judge test during vLLM startup. Triton could not compile its CUDA helper because Python.h was missing. The Docker recipe now installs python3.12-dev and compiles the helper in a build check. The replacement builds under results/judge_qwen35/step1184-pythonheaders/. Use its own DOCKER_ARCHIVE_READY marker and MD5. The weights and original 40.72 inference policy are preserved. Judge GPU startup and script/notebook parity remain to be tested.
- The user handles leaderboard JSONL uploads and the judge's Google Drive link plus MD5. Keep the archive, checksum, release manifest, logged build result, and backed-up checkpoints available. Confirm the user's choice before changing a release candidate.

## Experiment decisions

- Earlier diagnostics recovered 30 of 48 selected audio failures with reference transcripts. Beam-3 and the transfer-account prompt lost on the pilot. ASR-target augmentation improved the step-608 pilot and lost on full development. CUDA kernels sped up training, with no need to change inference.
- The first-turn retry changed zero of 634 predictions. Selective ASR reranking changed one prediction, changed no final state, and scored the same 40.72 publicly. Skip repeats of both runs.
- **First Molab batch:** device binding, first-tool suggestions, and a separate 48-clip ASR diagnostic after the public runs. Run and backup logs: `/marimo/results/lifemate-step1184-variants/{run,backup}.log`. [Private outputs](https://huggingface.co/datasets/bumbleboonq/zaic-lifemate/tree/main/qwen35-full-train/from-step548/experiments/step1184-variants-v1).
- Last checked on 15 November: device binding finished, changed two final states, and its JSONL and summary are backed up. First-tool suggestions was at 300/634 and had changed five final states. Public scores for those variants are unconfirmed.
- **Second Molab batch:** schema lookup hints, compound-request planning, completion check after lookup, and contextual ASR. It waits for the first batch and diagnostic. Run and backup logs: `/marimo/results/lifemate-step1184-variants-v2/{run,backup}.log`. [Private outputs](https://huggingface.co/datasets/bumbleboonq/zaic-lifemate/tree/main/qwen35-full-train/from-step548/experiments/step1184-variants-v2).
- Check `HF_VARIANT_BACKUP_READY` and the aggregate report for a validated JSONL. Prefer files with meaningful prediction changes. An identical SHA-256 is the same submission, and a changed state is not evidence of a gain without a score.

## Working constraints

- Run GPU workloads on Molab. Docker Desktop is used for build and export on the laptop. The current Molab runtime has no Docker engine for testing the exact image.
- Use the mounted official ToolEnv for tool execution and preserve its trace and final state hash. Evaluate each experimental change separately against a known control. Keep the judge image offline and requests under 30 seconds.
- Keep data, credentials, raw transcripts, weights, generated submissions, and Docker archives out of Git. Put verified shareable artifacts in the private Hugging Face dataset. Use runtime secrets for access. If the runtime changes, get the current pairing session from the user.
