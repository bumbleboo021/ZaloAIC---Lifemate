"""Run the gated Qwen3.5 challenger from smoke test through public export."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "agent", "asr", "fitting", "train-python", "output", "baseline"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, help="Reuse a successful smoke summary")
    parser.add_argument("--max-train-minutes", type=int, default=45)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir()
    source = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(source), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               USE_TF="0")
    from lifemate_experiment import write_json
    from experiments.evaluate_lifemate_asr import paired, read_report

    def run(name, command, seconds):
        print("QWEN35_START", name, flush=True)
        with (args.output / f"{name}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                           timeout=seconds, check=True)

    def evaluate(name, stage, agent):
        run(name, [sys.executable, "-u", "-m", "experiments.evaluate_lifemate_qwen35",
                   "--data-root", str(args.data_root), "--asr", str(args.asr),
                   "--agent", str(agent), "--output", str(args.output / name),
                   "--stage", stage], 5400 if stage == "public" else 2400)
        return read_report(args.output / name / "summary.json")

    smoke = read_report(args.smoke) if args.smoke else evaluate("smoke", "smoke", args.agent)
    if not smoke.get("samples") or smoke["at_or_over_30s"] or smoke["stops"].get("time_budget"):
        raise ValueError("The Qwen3.5 smoke test failed.")
    summary = {"smoke": smoke}
    write_json(args.output / "summary.json", summary)
    train_dir = args.output / "training"
    merged = args.output / "merged"
    run("training", [str(args.train_python), "-u", "-m", "training_code.train_lifemate_lora",
                     "--fitting", str(args.fitting), "--agent", str(args.agent),
                     "--output", str(train_dir), "--max-minutes", str(args.max_train_minutes),
                     "--model-family", "qwen35"], 60 * (args.max_train_minutes + 15))
    run("merge", [str(args.train_python), "-u", "-m", "training_code.merge_lifemate_qwen35",
                  str(args.agent), str(train_dir / "adapter"), str(merged)], 900)
    for config in args.agent.iterdir():
        if config.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if config.is_file() and config.suffix in {".json", ".jinja", ".txt"}:
            destination = merged / config.name
            if not destination.exists():
                shutil.copy2(config, destination)
    summary["training"] = read_report(train_dir / "training_metrics.json")
    write_json(args.output / "summary.json", summary)
    evaluate("pilot", "pilot", merged)
    pilot = read_report(args.output / "pilot/pilot_score.json")
    summary["pilot"] = paired(pilot, read_report(args.baseline / "tuned_score.json"))
    write_json(args.output / "summary.json", summary)
    if pilot["at_or_over_30s"] or pilot["stops"].get("time_budget"):
        raise ValueError("The trained pilot did not meet the sample deadline.")
    evaluate("development", "development", merged)
    dev = read_report(args.output / "development/development_score.json")
    summary["development"] = paired(
        dev, read_report(args.baseline / "development/tuned_score.json")
    )
    write_json(args.output / "summary.json", summary)
    if not dev["at_or_over_30s"] and not dev["stops"].get("time_budget"):
        summary["public"] = evaluate("public", "public", merged)
        write_json(args.output / "summary.json", summary)
    print("QWEN35_CHALLENGER_READY", flush=True)


if __name__ == "__main__":
    main()

