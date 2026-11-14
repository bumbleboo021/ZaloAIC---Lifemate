"""Verify both CUDA training kernels and repeat the 60-update LifeMate run."""
import argparse
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time


def verify():
    import torch
    from transformers.utils import import_utils

    import_utils._torchvision_available = False
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    def active(wrapper, module):
        while wrapper is not None:
            closure = dict(zip(wrapper.__code__.co_freevars,
                               (cell.cell_contents for cell in (wrapper.__closure__ or ()))) )
            selected = closure.get("implementation")
            if selected is not None:
                return selected.__module__.startswith(module)
            wrapper = getattr(wrapper, "__wrapped__", None)
        return False

    assert active(qwen.torch_chunk_gated_delta_rule, "fla"), "FLA fallback is active"
    assert active(qwen.causal_conv1d_fn, "causal_conv1d"), "Convolution fallback is active"

    torch.manual_seed(42)
    values = torch.randn(1, 64, 256, device="cuda", dtype=torch.bfloat16)
    conv_input = values.transpose(1, 2).detach().requires_grad_(True)
    weight = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    conv = qwen.causal_conv1d_fn(conv_input, weight, activation="silu")
    conv.float().square().mean().backward()
    assert torch.isfinite(weight.grad).all()
    q, k, v = [torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    gate = -torch.rand(1, 128, 4, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, 128, 4, device="cuda", dtype=torch.bfloat16)
    output, _ = qwen.torch_chunk_gated_delta_rule(
        q, k, v, g=gate, beta=beta, use_qk_l2norm_in_kernel=True
    )
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    print("FAST_KERNEL_SMOKE_READY", flush=True)
    torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    args, training_args = parser.parse_known_args()
    verify()
    if args.smoke_only:
        return
    from transformers import TrainerCallback
    import transformers

    steps = []
    class Timing(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            import torch
            torch.cuda.synchronize()
            steps.append(time.monotonic())

    original = transformers.Trainer

    class TimedTrainer(original):
        def __init__(self, *args, **kwargs):
            kwargs["callbacks"] = [*kwargs.get("callbacks", ()), Timing()]
            super().__init__(*args, **kwargs)

    transformers.Trainer = TimedTrainer
    sys.argv = ["train_lifemate_lora", *training_args]
    runpy.run_module("training_code.train_lifemate_lora", run_name="__main__")
    import torch
    output = Path(training_args[training_args.index("--output") + 1])
    intervals = [right - left for left, right in zip(steps[4:], steps[5:])]
    metrics = {"updates": len(steps), "median_seconds_per_update": statistics.median(intervals)
               if intervals else None, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    (output / "fast_kernel_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("FAST_TRAINING_READY", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
