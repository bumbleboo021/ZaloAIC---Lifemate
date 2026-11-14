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
RUN flashinfer download-cubin
RUN python -c "from importlib.metadata import version; assert version('flashinfer-python') == version('flashinfer-cubin') == '0.6.12'"
COPY lifemate_baseline.py lifemate_eval.py lifemate_experiment.py lifemate_phase2.py lifemate_workflows.py predict.py start_jupyter.sh ./
COPY experiments/__init__.py experiments/evaluate_lifemate_qwen35.py ./experiments/
COPY experiments/run_lifemate_qwen35_full_train.py ./experiments/
COPY training_code ./training_code
COPY validation ./validation
COPY judge_qwen35/README.md judge_qwen35/predict.sh judge_qwen35/judge_predict.py judge_qwen35/reproduce.md ./
COPY judge_step150/predict_notebook.ipynb ./
COPY judge_qwen35/reproduce.md ./training_code/README.md
COPY --from=weights agent ./saved_models/agent
COPY --from=weights asr ./saved_models/asr
COPY --from=weights release.json ./release.json
RUN test -s saved_models/agent/model.safetensors && test -s saved_models/asr/model.safetensors && \
    sed -i 's/\r$//' predict.sh start_jupyter.sh && chmod +x predict.sh start_jupyter.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
EXPOSE 9777
CMD ["/bin/bash", "/code/predict.sh"]
