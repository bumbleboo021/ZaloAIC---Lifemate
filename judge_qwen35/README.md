# ZAIC 2026 LifeMate solution

The included `/code/release.json` identifies the exact candidate and records the SHA-256 of every staged inference file. The solution uses Qwen3.5-4B (4.539B serialized weights in the step-548 snapshot, including the vision weights) and PhoWhisper-medium (0.764B serialized weights), approximately 5.303B serialized weights combined. Tied embedding tensors are stored once. vLLM serves the text agent with a 16,384-token context limit. PhoWhisper transcribes Vietnamese user turns with greedy decoding.

We pass earlier turns in chronological order along with the request, installed tool schemas and context. The agent proposes one call at a time using the local vLLM server with thinking disabled. It validates and executes through the mounted official ToolEnv, then reads its observations. One ToolEnv per request preserves the actual call trace and final state hash. The scripts time each request from input processing through output.

Step 548 is the verified fallback from the 1,158-request fitting split. It scored 40.07 on the 232-request development slice. The full-TRAIN candidate continues from step 548 on 5,086 targets over all 1,545 labeled TRAIN requests. PhoWhisper is shared across both. Neither release candidate uses synthetic data. See `/code/training_code/README.md` for training details and data access.

- Run `bash /code/predict.sh` with the official split mounted at `/data` and a writable `/result`. It writes `/result/submission.jsonl`.
- Run `bash /code/start_jupyter.sh` and open `predict_notebook.ipynb` on port 9777. Run All writes `/result/jupyter_submission.jsonl`. The notebook uses the same model loading and prediction functions as the script.
- All inference weights and tokenizer components are included. Inference uses a local loopback server and can run with container networking disabled.
- A startup failure prints a tail of the vLLM server log to the container output. The full log is in `/result/vllm-server.log`.
