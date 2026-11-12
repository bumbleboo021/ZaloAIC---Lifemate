"""Marimo controls for the LifeMate baseline."""

import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import subprocess
    import sys
    from pathlib import Path
    return Path, mo, subprocess, sys


@app.cell
def _(mo):
    mo.md("# LifeMate GPU baseline")
    return


@app.cell
def _(mo):
    data_root = mo.ui.text(value="train+public_test", label="Dataset root")
    split = mo.ui.dropdown(
        options=["train", "public_test"], value="train", label="Split",
    )
    limit = mo.ui.number(start=0, stop=1545, value=5, label="Sample limit (0 for full split)")
    output_dir = mo.ui.text(value="results", label="Output directory")
    mo.vstack([data_root, split, limit, output_dir])
    return data_root, limit, output_dir, split


@app.cell
def _(mo):
    run_baseline = mo.ui.run_button(label="Run on molab GPU")
    run_baseline
    return (run_baseline,)


@app.cell
def _(Path, data_root, limit, mo, output_dir, run_baseline, split, subprocess, sys):
    mo.stop(not run_baseline.value)
    _script = Path.cwd() / "lifemate_baseline.py"
    _output = Path(output_dir.value) / f"{split.value}_submission.jsonl"
    _command = [
        sys.executable, str(_script),
        "--data-root", data_root.value,
        "--split", split.value,
        "--output", str(_output),
    ]
    if limit.value:
        _command.extend(["--limit", str(int(limit.value))])
    subprocess.run(_command, check=True)
    mo.md(f"Saved `{_output}`.")
    return


if __name__ == "__main__":
    app.run()
