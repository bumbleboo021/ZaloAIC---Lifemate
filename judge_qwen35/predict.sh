#!/usr/bin/env bash
set -euo pipefail
cd /code
exec python -u judge_predict.py "$@"
