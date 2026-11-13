# LifeMate reproduction

The selected solution uses Whisper large-v3-turbo and Qwen3-4B with the step-362 LoRA adapter. The model IDs and immutable revisions are in `fetch_models.py`. The image build fetches the two base models and copies the adapter files staged according to the repository README. The container merges the adapter locally at inference.

From the repository root on a remote GPU host, prepare fitting targets with `prepare_lifemate_sft.py` and train with `train_lifemate_lora.py`. The selected run used the official TRAIN kit, excluding the fixed development and holdout IDs. It retained 3,801 assistant targets from 1,158 requests with the 12,288-token cap. Rank 16, alpha 32, dropout 0.05, q/v projections, learning rate 1e-4, effective batch size 8, and seed 42 are set in the trainer. The 90-minute run ended at step 362 of the original 476-step schedule. Use the archived step-362 adapter to reproduce the released predictions, since the step reached by a time cap varies with GPU throughput.

Sources: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), and the [official LifeMate kit for registered teams](https://challenge.zalo.ai/portal/lifemate). The official simulator is provided by the mounted split. Build and verification steps are in the repository README.
