"""Run the step-548 PhoWhisper augmentation experiment on Molab."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from experiments.compare_lifemate_qwen35_asr import save
from experiments.evaluate_lifemate_asr import paired, read_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recovery", type=Path)
    parser.add_argument("--archive-url", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.recovery.resolve()
    source = Path(__file__).resolve().parents[1]
    run = root / "qwen35-step548-asr-mix"
    run.mkdir(exist_ok=True)
    infer = root / ".public-inference-venv/bin/python"
    train = root / ".merge-venv/bin/python"
    asr = root / "pho-merged"
    base = root / "qwen-base"
    data = root / "development-kit"
    control = root / "milestone-submissions/merged-step548"
    checkpoint = root / "restored-milestones/step548/checkpoint-476"

    def execute(python, module, *arguments):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source, data / "train"))),
                   USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HOME=str(root / ".public-hf-cache"))
        for key in ("LIFEMATE_HF_BACKUP_REPO", "LIFEMATE_HF_BACKUP_RUN"):
            env.pop(key, None)
        subprocess.run([str(python), "-u", "-m", module, *map(str, arguments)],
                       env=env, cwd=source, check=True)

    execute(infer, "training_code.prepare_lifemate_qwen35_asr_mix",
            "--archive-url", args.archive_url, "--agent", control, "--asr", asr,
            "--output", run / "preparation", "--validation", source / "validation")
    execute(train, "training_code.train_lifemate_lora",
            "--fitting", run / "preparation/mixed.private.jsonl", "--agent", base,
            "--init-adapter", checkpoint, "--output", run / "training",
            "--model-family", "qwen35", "--step-offset", "548", "--max-steps", "60",
            "--max-minutes", "55", "--milestone-steps", "568", "588", "608")

    merged = run / "merged"
    execute(train, "training_code.merge_lifemate_qwen35", base, run / "training/adapter", merged)
    for file in base.iterdir():
        target = merged / file.name
        if (file.is_file() and file.suffix in {".json", ".jinja", ".txt"}
                and file.name != "model.safetensors.index.json" and not target.exists()):
            shutil.copyfile(file, target)

    summary = {"checkpoint": 548,
               "preparation": read_report(run / "preparation/preparation_summary.json"),
               "training": read_report(run / "training/training_metrics.json")}
    controls = {
        "pilot": root / "qwen35-step548-asr-beam3/pilot/greedy_score.json",
        "development": root / "milestone-development/step548/development_score.json",
    }
    for stage, baseline in controls.items():
        execute(infer, "experiments.evaluate_lifemate_qwen35", "--data-root", data,
                "--agent", merged, "--asr", asr, "--output", run / stage,
                "--stage", stage, "--port", "18087")
        outcome = paired(read_report(run / stage / f"{stage}_score.json"), read_report(baseline))
        summary[stage] = outcome
        false_key = "calls_when_empty_expected"
        passed = (outcome["delta_points"] > 0 and not outcome["at_or_over_30s"]
                  and outcome["candidate_failures"].get(false_key, 0)
                  <= outcome["baseline_failures"].get(false_key, 0))
        summary["outcome"] = ("development_improved" if stage == "development" and passed
                              else f"{stage}_passed" if passed else f"{stage}_gate_failed")
        save(run, summary)
        if not passed:
            break
    print("QWEN35_ASR_MIX_READY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
