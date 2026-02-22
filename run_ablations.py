"""
Ablation experiment runner for the distillation pipeline.

Runs a matrix of experiments with different distillation signal combinations
and aggregates results into a comparison table.

Group A: Feature signal ablation (plan.md section 5.1)
Group B: Fusion and two-stage ablation (plan.md section 5.2)

Usage:
    python run_ablations.py                  # Run all ablations
    python run_ablations.py --group A        # Run only Group A
    python run_ablations.py --group B        # Run only Group B
    python run_ablations.py --summary-only   # Just print results table
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


# Base config shared by all experiments
BASE_CONFIG = {
    "data": {
        "distance": 3,
        "rounds": 5,
        "noise_strength": 0.005,
        "snr": 10.0,
        "use_soft": True,
        "num_train": 200000,
        "num_val": 20000,
        "batch_size": 512,
        "seed": 42,
    },
    "model": {
        "size": "small",
        "rnn_type": "gru",
    },
    "training": {
        "epochs": 30,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "warmup_steps": 200,
        "grad_clip": 1.0,
        "device": "auto",
    },
    "logging": {
        "log_interval": 50,
        "eval_interval": 1,
    },
}

TEACHER_CHECKPOINT = "checkpoints/mock_teacher_d3/best_model.pt"
PROBE_HEADS_PATH = "checkpoints/mock_teacher_d3/probe_heads.pt"
STAGE1_CHECKPOINT = "checkpoints/stage1_kd_d3/best_model.pt"

# ── Group A: Feature signal ablation ──
# Tests which distillation signals contribute and whether they are complementary.
GROUP_A_EXPERIMENTS = {
    "abl_cnn_only": {
        "description": "CNN feature KD only (spatial signal)",
        "script": "train_distill.py",
        "teacher": {"checkpoint": TEACHER_CHECKPOINT},
        "distillation": {
            "alpha": 0.3, "beta": 0.3, "gamma_cnn": 0.4, "gamma_rnn": 0.0,
            "temperature": 2.0, "feature_loss_type": "mse",
        },
    },
    "abl_rnn_only": {
        "description": "RNN feature KD only (temporal signal)",
        "script": "train_distill.py",
        "teacher": {"checkpoint": TEACHER_CHECKPOINT},
        "distillation": {
            "alpha": 0.3, "beta": 0.3, "gamma_cnn": 0.0, "gamma_rnn": 0.4,
            "temperature": 2.0, "feature_loss_type": "mse",
        },
    },
    "abl_cnn_rnn": {
        "description": "CNN + RNN feature KD (spatial + temporal)",
        "script": "train_distill.py",
        "teacher": {"checkpoint": TEACHER_CHECKPOINT},
        "distillation": {
            "alpha": 0.3, "beta": 0.3, "gamma_cnn": 0.2, "gamma_rnn": 0.2,
            "temperature": 2.0, "feature_loss_type": "mse",
        },
    },
    "abl_response_only": {
        "description": "Response KD only (readout signal, no feature KD)",
        "script": "train_distill.py",
        "teacher": {"checkpoint": TEACHER_CHECKPOINT},
        "distillation": {
            "alpha": 0.3, "beta": 0.7, "gamma_cnn": 0.0, "gamma_rnn": 0.0,
            "temperature": 2.0, "feature_loss_type": "mse",
        },
    },
}

# ── Group B: Fusion and two-stage ablation ──
# Tests whether fused logits and two-stage training are necessary.
# Fair comparisons:
#   stage2_kd_d3 vs abl_stage2_response_only → effect of fused logits (same init + lr)
#   abl_fused_only vs stage2_kd_d3 → effect of Stage 1 init (fused_only is from scratch)
GROUP_B_EXPERIMENTS = {
    "abl_fused_only": {
        "description": "Fused logit KD from scratch (no Stage 1 init)",
        "script": "train_distill.py",
        "teacher": {
            "checkpoint": TEACHER_CHECKPOINT,
            "probe_heads": PROBE_HEADS_PATH,
        },
        "distillation": {
            "alpha": 0.3, "beta": 0.1, "gamma_cnn": 0.0, "gamma_rnn": 0.0,
            "gamma_fused": 0.6, "temperature": 2.0,
        },
    },
    "abl_stage2_response_only": {
        "description": "Stage 2 with response KD only (no fused logits)",
        "script": "train_distill.py",
        "teacher": {
            "checkpoint": TEACHER_CHECKPOINT,
        },
        "distillation": {
            "alpha": 0.3, "beta": 0.7, "gamma_cnn": 0.0, "gamma_rnn": 0.0,
            "gamma_fused": 0.0, "temperature": 2.0,
        },
        "init_checkpoint": STAGE1_CHECKPOINT,
        "training": {"learning_rate": 0.0005, "warmup_steps": 100},
    },
}

# Pre-existing experiments (already trained, just read results)
EXISTING_EXPERIMENTS = {
    "scratch_d3": {
        "description": "No distillation (scratch training)",
    },
    "baseline_kd_d3": {
        "description": "Baseline KD (final logit only)",
    },
    "stage1_kd_d3": {
        "description": "Stage 1 multi-signal KD (=cnn+rnn+response)",
    },
    "stage2_kd_d3": {
        "description": "Stage 1 → Stage 2 (full two-stage pipeline)",
    },
}


def build_config(name, experiment):
    """Build full config dict for an experiment."""
    config = {}
    # Deep copy base config
    for section, values in BASE_CONFIG.items():
        config[section] = dict(values)

    # Add teacher config
    if "teacher" in experiment:
        config["teacher"] = dict(experiment["teacher"])

    # Add distillation config
    if "distillation" in experiment:
        config["distillation"] = dict(experiment["distillation"])

    # Add init_checkpoint if specified
    if "init_checkpoint" in experiment:
        config["model"]["init_checkpoint"] = experiment["init_checkpoint"]

    # Apply training overrides if specified
    if "training" in experiment:
        for k, v in experiment["training"].items():
            config["training"][k] = v

    # Set save directory
    config["logging"]["save_dir"] = f"checkpoints/{name}"

    return config


def run_experiment(name, experiment):
    """Run a single experiment."""
    save_dir = Path(f"checkpoints/{name}")
    history_file = save_dir / "history.json"

    # Skip if already completed
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        expected_epochs = BASE_CONFIG["training"]["epochs"]
        if len(history) >= expected_epochs:
            print(f"  [SKIP] {name}: already completed ({len(history)} epochs)")
            return True

    # Build and save config
    config = build_config(name, experiment)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "ablation_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    script = experiment.get("script", "train_distill.py")
    cmd = [sys.executable, script, "--config", str(config_path)]
    print(f"  [RUN] {name}: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] {name}")
        print(result.stderr[-500:] if result.stderr else "No stderr")
        return False

    print(f"  [DONE] {name}")
    return True


def load_results(name):
    """Load training results for an experiment."""
    history_file = Path(f"checkpoints/{name}/history.json")
    if not history_file.exists():
        return None

    with open(history_file) as f:
        history = json.load(f)

    if not history:
        return None

    # Find best epoch by val_ler
    best = min(history, key=lambda x: x["val_ler"])
    return {
        "best_val_ler": best["val_ler"],
        "best_epoch": best["epoch"],
        "ler_per_round": best["ler_per_round"],
        "best_val_acc": best["val_accuracy"],
        "final_val_ler": history[-1]["val_ler"],
        "n_epochs": len(history),
    }


def print_summary():
    """Print comparison table of all experiments."""
    # Collect all experiment names
    all_experiments = {}
    all_experiments.update(EXISTING_EXPERIMENTS)
    all_experiments.update(GROUP_A_EXPERIMENTS)
    all_experiments.update(GROUP_B_EXPERIMENTS)

    # Load teacher LER as reference
    teacher_results = load_results("mock_teacher_d3")

    print("\n" + "=" * 90)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 90)

    if teacher_results:
        print(f"Teacher reference: val_ler={teacher_results['best_val_ler']:.4f} "
              f"(epoch {teacher_results['best_epoch']})")
    print()

    # Header
    print(f"{'Experiment':<25} {'Description':<40} {'Best LER':>9} {'Epoch':>6} "
          f"{'LER/round':>10}")
    print("-" * 90)

    for name, exp in all_experiments.items():
        results = load_results(name)
        desc = exp.get("description", "")[:38]
        if results:
            print(f"{name:<25} {desc:<40} {results['best_val_ler']:>9.4f} "
                  f"{results['best_epoch']:>6d} {results['ler_per_round']:>10.6f}")
        else:
            print(f"{name:<25} {desc:<40} {'N/A':>9} {'N/A':>6} {'N/A':>10}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--group", type=str, choices=["A", "B", "all"], default="all",
                        help="Which experiment group to run")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print results table, don't run experiments")
    args = parser.parse_args()

    if args.summary_only:
        print_summary()
        return

    # Pre-flight checks: ensure prerequisite checkpoints exist
    missing = []
    if not Path(TEACHER_CHECKPOINT).exists():
        missing.append(f"Teacher: {TEACHER_CHECKPOINT}")
    if args.group in ("B", "all"):
        if not Path(STAGE1_CHECKPOINT).exists():
            missing.append(f"Stage 1: {STAGE1_CHECKPOINT}")
        if not Path(PROBE_HEADS_PATH).exists():
            missing.append(f"Probe heads: {PROBE_HEADS_PATH}")
    if missing:
        print("ERROR: Required checkpoints not found:")
        for m in missing:
            print(f"  - {m}")
        print("Train the prerequisite models first.")
        sys.exit(1)

    experiments = {}
    if args.group in ("A", "all"):
        experiments.update(GROUP_A_EXPERIMENTS)
    if args.group in ("B", "all"):
        experiments.update(GROUP_B_EXPERIMENTS)

    print(f"Running {len(experiments)} ablation experiments...")
    print()

    for name, exp in experiments.items():
        desc = exp.get("description", "")
        print(f"--- {name}: {desc} ---")
        success = run_experiment(name, exp)
        if not success:
            print(f"  Warning: {name} failed, continuing with remaining experiments")
        print()

    # Print summary table
    print_summary()


if __name__ == "__main__":
    main()
