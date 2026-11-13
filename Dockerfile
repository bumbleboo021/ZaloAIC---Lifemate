FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY requirements.txt ./
RUN python3 -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132 \
    && python3 -m pip install -r requirements.txt && python3 -m pip check
COPY training_code ./training_code
RUN python3 training_code/fetch_models.py --output /code/saved_models
COPY release_adapter/adapter_config.json release_adapter/adapter_model.safetensors /code/saved_models/adapter/
COPY lifemate_baseline.py predict.py predict.sh start_jupyter.sh predict_notebook.ipynb ./
RUN sed -i 's/\r$//' predict.sh start_jupyter.sh && chmod +x predict.sh start_jupyter.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 PYTHONHASHSEED=42 LIFEMATE_ADAPTER=/code/saved_models/adapter
EXPOSE 9777
CMD ["/bin/bash", "/code/predict.sh"]
