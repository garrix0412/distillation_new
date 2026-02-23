"""
End-to-end distillation pipeline runner.

Executes the full pipeline in order:
  1. Train mock teacher
  2. Train probe heads
  3. Stage 1 KD (CNN + RNN feature KD — needed by Stage 2 and ablations)
  4. Stage 1 v2 KD (three-signal: CNN + RNN + readout feature KD)
  5. Stage 2 KD (fused logits, init from Stage 1)
  6. Ablation experiments

Supports resuming from any step with --from N.

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --from 3     # Resume from step 3
    python run_pipeline.py --dry-run    # Print steps without running
"""

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    {
        "name": "Train mock teacher",
        "cmd": [sys.executable, "train.py", "--config", "configs/mock_teacher_d3.yaml"],
        "checkpoint": "checkpoints/mock_teacher_d3/best_model.pt",
    },
    {
        "name": "Train probe heads",
        "cmd": [sys.executable, "train_probes.py", "--config", "configs/mock_teacher_d3.yaml"],
        "checkpoint": "checkpoints/mock_teacher_d3/probe_heads.pt",
    },
    {
        "name": "Stage 1 KD (CNN + RNN feature KD)",
        "cmd": [sys.executable, "train_distill.py", "--config", "configs/stage1_kd_d3.yaml"],
        "checkpoint": "checkpoints/stage1_kd_d3/best_model.pt",
    },
    {
        "name": "Stage 1 v2 KD (CNN + RNN + readout feature KD)",
        "cmd": [sys.executable, "train_distill.py", "--config", "configs/stage1_v2_kd_d3.yaml"],
        "checkpoint": "checkpoints/stage1_v2_kd_d3/best_model.pt",
    },
    {
        "name": "Stage 2 KD (fused logits, init from Stage 1)",
        "cmd": [sys.executable, "train_distill.py", "--config", "configs/stage2_kd_d3.yaml"],
        "checkpoint": "checkpoints/stage2_kd_d3/best_model.pt",
    },
    {
        "name": "Ablation experiments",
        "cmd": [sys.executable, "run_ablations.py"],
        "checkpoint": None,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run full distillation pipeline")
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Start from step N (1-indexed, default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print pipeline steps without executing",
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print("Distillation Pipeline")
    print(f"{'='*60}")

    for i, step in enumerate(STEPS, 1):
        status = "SKIP" if i < args.from_step else "RUN"
        print(f"  Step {i}: {step['name']} [{status}]")
    print()

    if args.dry_run:
        print("Dry run — no commands executed.")
        return

    for i, step in enumerate(STEPS, 1):
        if i < args.from_step:
            continue

        # Skip if checkpoint already exists
        if step["checkpoint"] and Path(step["checkpoint"]).exists():
            print(f"[Step {i}/{len(STEPS)}] {step['name']} — checkpoint exists, skipping")
            continue

        print(f"[Step {i}/{len(STEPS)}] {step['name']}")
        print(f"  Command: {' '.join(step['cmd'])}")

        result = subprocess.run(step["cmd"])
        if result.returncode != 0:
            print(f"\nERROR: Step {i} failed (exit code {result.returncode})")
            print(f"Fix the issue and resume with: python run_pipeline.py --from {i}")
            sys.exit(1)

        print(f"  Done.\n")

    print(f"{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
