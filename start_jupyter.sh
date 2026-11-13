#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 PYTHONHASHSEED=42
exec jupyter lab --port=9777 --ip=0.0.0.0 --IdentityProvider.token="${JUPYTER_TOKEN:-zac2026}" --allow-root --no-browser
