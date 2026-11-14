# Build with the repository root as the build context.
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates build-essential libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/lifemate
ENV PATH="/opt/lifemate/bin:$PATH"
WORKDIR /code
COPY judge_step150/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir uv && \
    uv pip install --python /opt/lifemate/bin/python --torch-backend=cu130 -r requirements.txt && \
    python -m pip check
# Use the kernel downloader directly so image builds do not require CUDA device detection.
RUN flashinfer download-cubin
COPY lifemate_baseline.py lifemate_eval.py lifemate_experiment.py lifemate_phase2.py lifemate_workflows.py ./
COPY predict.py ./predict.py
COPY experiments ./experiments
COPY training_code ./training_code
COPY validation ./validation
COPY judge_step150/predict.py ./judge_predict.py
COPY judge_step150/predict_notebook.ipynb judge_step150/README.md ./
COPY judge_step150/predict.sh ./predict.sh
COPY start_jupyter.sh ./
COPY judge_step150/staged_models ./saved_models
RUN test -s saved_models/agent/config.json && test -s saved_models/asr/config.json && \
    sed -i 's/\r$//' predict.sh start_jupyter.sh && chmod +x predict.sh start_jupyter.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
EXPOSE 9777
CMD ["/bin/bash", "/code/predict.sh"]
