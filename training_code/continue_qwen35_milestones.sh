#!/usr/bin/env bash
# Run on molab after the current continuation and its evaluations finish.
set -euo pipefail

root=${1:-/marimo/results/lifemate-qwen35-recovery}
current="$root/continuation-150"
next="$root/continuation-milestones"
source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$root/.merge-venv/bin/python"

grep -qx CONTINUATION_READY "$current/run.log" || {
  echo "Current workflow has not finished. Check $current/run.log." >&2
  exit 1
}
checkpoint="$current/training/checkpoint-78"
"$python" - "$checkpoint" "$current/training/training_metrics.json" <<'PY'
import json
from pathlib import Path
import sys

checkpoint, metrics = map(Path, sys.argv[1:])
for name in ("adapter_model.safetensors", "trainer_state.json", "optimizer.pt",
             "scheduler.pt", "rng_state.pth"):
    if not (checkpoint / name).is_file():
        raise SystemExit(f"Missing resume file: {checkpoint / name}")
state = json.loads((checkpoint / "trainer_state.json").read_text())
if state["global_step"] != 78 or json.loads(metrics.read_text())["optimizer_steps"] != 78:
    raise SystemExit("Expected 78 completed updates in the current continuation.")
PY

umask 077
mkdir "$next"
mkdir -p "$next/training/milestones"
cp -a "$checkpoint" "$next/training/milestones/checkpoint-150"
echo "MILESTONE_READY 150"
unset PYTHONHOME VIRTUAL_ENV PYTHONUSERBASE PYTHONSAFEPATH
export PYTHONPATH="$source" USE_TF=0
if [[ -n "${LIFEMATE_HF_BACKUP_REPO:-}" ]]; then
  : "${LIFEMATE_HF_BACKUP_RUN:?Set a unique run folder for checkpoint backups}"
  : "${HF_TOKEN:?Set HF_TOKEN in the protected Molab environment}"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
cd "$source"
"$python" -u -m training_code.train_lifemate_lora \
  --fitting "$current/fitting.private.jsonl" --agent "$root/qwen-base" \
  --output "$next/training" --model-family qwen35 --resume "$checkpoint" \
  --max-minutes 360 --step-offset 72 --milestone-steps 232 362 \
  >"$next/training.log" 2>&1
"$python" - "$next/training/training_metrics.json" <<'PY'
import json
from pathlib import Path
import sys

metrics = json.loads(Path(sys.argv[1]).read_text())
if metrics["epoch"] < 0.999:
    raise SystemExit(f"Fitting pass stopped early: {metrics}")
print("MILESTONE_TRAINING_READY", metrics["cumulative_updates"])
PY
