# ZaloAIC---Lifemate

## Run the baseline on molab

The baseline transcribes Vietnamese audio and conversation history with Whisper large-v3-turbo. Qwen3-4B selects actions from the installed tool schemas, and the official simulator produces the trace and final state hash. [Whisper model card](https://huggingface.co/openai/whisper-large-v3-turbo) · [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B)

1. Attach your molab GPU and open `molab_baseline.py` as a marimo notebook. Place `lifemate_baseline.py` in the notebook's working folder.
2. In the molab package manager, install `transformers==4.57.1`, `soundfile`, and `scipy`. Use molab's CUDA-enabled PyTorch.
3. Place the official kit in your authorized remote storage. Set **Dataset root** to the extracted directory containing `train/` and `public_test/`.
4. Click **Run on molab GPU**. The default is a five-sample TRAIN smoke run. The first run downloads the model weights in molab. Save the result from `results/train_submission.jsonl`.

[Molab setup and storage documentation](https://docs.marimo.io/guides/molab/)

You can also launch it in the molab runtime from the directory containing the scripts.

```bash
python lifemate_baseline.py --data-root train+public_test --split train \
  --limit 5 --include-time --output results/train_smoke.jsonl
```

For a full public run, set the sample limit to 0 in the notebook or omit `--limit` on the command line. By default each output line contains `id`, `tool_calls`, and `state_hash`. Use `--ids` to supply an explicit list of validation IDs and `--include-time` for timing diagnostics.

For offline execution, supply downloaded model directories with `--asr-path` and `--agent-path` and add `--offline`. The first version uses pretrained weights and greedy decoding. Its per-sample time budget is cooperative, so the actual timing on scoring hardware still needs verification. The first GPU run, scored TRAIN validation, and Docker verification remain open.
