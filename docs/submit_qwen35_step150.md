# Step-150 Qwen3.5 judge package

The **Test Solution** form takes a Google Drive link to a **gzip-compressed Docker image** and the **MD5 of that exact archive**. The judge runs the image against its mounted data. The leaderboard JSONL is a separate artifact. The screenshot shows one Test Solution upload per day. [Official Docker guide](https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf)

## 1. Freeze the candidate

- Use the completed Qwen3.5 continuation at **78 additional optimizer updates** after step 72. The molab metrics recorded `optimizer_steps: 78`. The verified adapter backup is in `results/lifemate_qwen35_150/`. After the milestone launcher runs, its copy is `continuation-milestones/training/milestones/checkpoint-150/`. The public result of **28.42** belongs to step 72. Record step 150's metrics separately.
- Use the merged **Qwen3.5-4B step-150** weights and tokenizer/chat template, with **tuned PhoWhisper-medium epoch-one** weights and processor. Record the base revisions, adapter checksums, merge commands, seed, dependencies, and score report. Package the merged inference weights once. Keep optimizer state and extra checkpoints in private backup storage.
- Include the training and merge source, the split ID lists, and reproduction instructions for both Qwen3.5 training segments and the ASR run. Document the fresh optimizer at the start of the 78-step continuation. Point the judge to authorized access for the official training kit. Keep competition audio, labels, transcripts, raw traces, and credentials out of the Drive archive.

## 2. Wire the judge entry points

**The root Dockerfile and prediction entry points currently target the older Qwen3/Whisper-turbo setup.** Use the separate [step-150 release recipe](../judge_step150/Dockerfile). Reuse the Qwen3.5 vLLM agent and ToolEnv integration in [the evaluator](../experiments/evaluate_lifemate_qwen35.py). Its experiment CLI handles the known TRAIN/public stages and is not the judge's private-split entry point.

The release needs these files under `/code`:

| Item | Required behavior |
| --- | --- |
| `predict.sh` and shared inference module | Start local vLLM with the frozen agent, warm it up, load tuned PhoWhisper offline, select the mounted split under `/data`, and write `/result/submission.jsonl`. Log model load and prediction times. |
| `predict_notebook.ipynb` | Run All sets seed 42, loads the same models, reads the same mounted data, and runs the shared inference. Write `/result/jupyter_submission.jsonl` with per-sample `time` in milliseconds. |
| `start_jupyter.sh` | Launch JupyterLab on port 9777 as shown in the guide. |
| `saved_models/agent/` and `saved_models/asr/` | Hold the full merged weights and their configs, tokenizers, and processors. |
| `training_code/`, `validation/`, `README.md`, pinned dependencies | Describe the model sizes, context limit, tool loop, fitting data, and how to reproduce the exact submitted weights. |

Use one official ToolEnv per sample. Keep the vLLM configuration used to evaluate the checkpoint, including thinking disabled, its Qwen3.5 tool parser, and the 16,384-token limit. Only the local loopback connection between the agent and vLLM is needed at inference. The Docker guide's script example and timing notebook both include the fields `id`, `tool_calls`, `state_hash`, and `time`. Compare the script and notebook outputs by ID, calls, and state hash while allowing timing to vary.

## 3. Choose a Docker host and verify the package

Use a Linux x86-64 GPU VM or bare-metal machine with SSH, permission to run the Docker daemon, NVIDIA Container Toolkit, and a driver compatible with the selected base image. Ask a teammate or lab for a host first. If renting, choose a single-GPU VM with roughly 100 GB of free disk for the model cache, staged weights, build layers, and archive. Select the judge GPU when available, and verify the driver and pricing before provisioning. A larger GPU can validate packaging and offline execution. It cannot establish the judge's memory or timing results.

A [DigitalOcean NVIDIA GPU Droplet](https://docs.digitalocean.com/products/droplets/getting-started/recommended-gpu-setup/) is one possible VM option. Its inference image includes Docker. Confirm its available GPU, driver, and container toolkit before buying. Notebook services or container-only rentals are suitable only if they provide access to a working Docker daemon with GPU passthrough.

Before copying weights, verify these commands on the host:

```bash
nvidia-smi
docker version
docker run --rm --gpus all nvidia/cuda:13.2.0-devel-ubuntu24.04 nvidia-smi
```


Use the organizer-confirmed GPU and CUDA environment. The downloaded Docker guide names an RTX 5090 with 32 GB and gives different CUDA wording on pages 4 and 7. Use its CUDA 13.2 base for the test unless the organizer confirms an alternative. Validate the pinned vLLM/PyTorch wheels against that environment. Molab has been used for model evaluation, and this image still needs a Docker-capable test host.

On the remote Docker host, clone the source and stage the two merged model directories as `judge_step150/staged_models/agent/` and `judge_step150/staged_models/asr/`. In the current molab workspace they are `/marimo/results/lifemate-qwen35-recovery/continuation-150/merged` and `/marimo/results/lifemate-qwen35-recovery/pho-merged`. Transfer them through access-controlled storage or reconstruct them on the build host using the pinned base models and adapter backups. The image recipe requires the staged directories and does not fetch model weights. Build from the repository root:

```bash
set -euo pipefail
image=zac2026-teamname:qwen35-step150
docker build --platform linux/amd64 -f judge_step150/Dockerfile -t "$image" .
mkdir -p judge-result
docker run --rm --gpus all --network none \
  -v "$DATA_ROOT:/data:ro" \
  -v "$PWD/judge-result:/result" \
  "$image" bash /code/predict.sh --limit 5
# Repeat without --limit 5 for all 634 public samples.
```

Set `DATA_ROOT` to the absolute path containing `public_test/` for the local test. Check all 634 IDs, output fields, model loading, and end-to-end latency under 30 seconds per sample. Run the timing notebook and check output parity. Check peak GPU memory with **both** models loaded. The larger molab GPU does not establish that the image fits the judge's GPU.

For reconstruction, run this from the repository root on the build host. Use Python 3.12 with torch 2.11.0, PEFT 0.18.1, and Transformers 5.22.2 for the agent merge. Use a separate ASR environment with torch 2.11.0, PEFT 0.18.1, and Transformers 4.57.1. The command verifies both adapter archives and checks the merged ASR hash against molab.

```bash
python -m judge_step150.stage_models \
  --agent-adapter /secure/qwen35-step150-adapter.zip \
  --asr-adapter /secure/phowhisper-medium-lora-epoch1.zip \
  --asr-python /path/to/asr-venv/bin/python
```

## Windows Docker Desktop route

Use this route to build the archive on Windows without running GPU inference locally. Open Docker Desktop in Linux containers mode. From the repository root, run:

```powershell
& .\judge_step150\build_on_windows.ps1
```

The script downloads the exact molab weights and verifies both SHA-256 values, builds the Linux image, checks the packaged files offline without a GPU, exports and reloads the image, then writes the archive and MD5 under results/lifemate_qwen35_150/judge/. The staging download is resumable while the molab runtime is available. Allow room for staged weights, image layers, and both archive forms. Use the judge's Test Solution slot for the GPU test.

## 4. Upload the exact archive

```bash
archive=zac2026_TeamName_qwen35_step150.tar.gz
docker save "$image" | gzip > "$archive"
gzip -t "$archive"
md5sum "$archive"
```

Load the archive on a second remote Docker host with `docker load -i "$archive"` and repeat the offline check if available. Upload the same `.tar.gz` to Google Drive, set **Anyone with the link → Viewer** per the guide, and confirm the link can download the file in a signed-out window. Paste the file link into **Google url** and the 32-character hash into **Checksum MD5**. If you recreate or recompress the archive, calculate a new MD5. Submit through **Test Solution** after the smoke checks pass.
