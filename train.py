"""
Unified training script for surface code neural decoder.

Supports:
- Scratch training (no distillation, cross-entropy only)
- Knowledge distillation (future: baseline KD, Stage 1, Stage 2)

Usage:
    python train.py --config configs/scratch_d3.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data.dataset import create_dataloaders
from evaluation.metrics import evaluate_model, compute_ler_per_round
from models.student import create_student


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def train_one_epoch(
    model, train_loader, optimizer, scheduler, criterion, device, epoch, config
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    log_interval = config["logging"]["log_interval"]

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(inputs).squeeze(-1)  # [batch]
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping
        grad_clip = config["training"].get("grad_clip", 0)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"loss={avg_loss:.4f} acc={acc:.4f} lr={lr:.2e}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def build_scheduler(optimizer, config, total_steps):
    sched_type = config["training"].get("scheduler", "none")
    warmup_steps = config["training"].get("warmup_steps", 0)

    if sched_type == "cosine":
        # Cosine annealing with warm-up
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 3)
    return None


def main():
    parser = argparse.ArgumentParser(description="Train surface code decoder")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    # Create data loaders
    print("Generating training data...")
    t0 = time.time()
    train_loader, val_loader = create_dataloaders(
        distance=config["data"]["distance"],
        rounds=config["data"]["rounds"],
        num_train=config["data"]["num_train"],
        num_val=config["data"]["num_val"],
        noise_strength=config["data"]["noise_strength"],
        snr=config["data"]["snr"],
        batch_size=config["data"]["batch_size"],
        use_soft=config["data"]["use_soft"],
        seed=config["data"]["seed"],
    )
    print(f"Data generated in {time.time()-t0:.1f}s")
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} samples")

    # Create model
    model = create_student(
        distance=config["data"]["distance"],
        size=config["model"]["size"],
        rnn_type=config["model"]["rnn_type"],
        use_soft=config["data"]["use_soft"],
    )
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model: {config['model']['size']} ({config['model']['rnn_type']}), {n_params:,} parameters")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = build_scheduler(optimizer, config, total_steps)

    # Save directory
    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Training loop
    best_val_ler = 1.0
    history = []

    print(f"\n{'='*60}")
    print(f"Starting training for {config['training']['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, config["training"]["epochs"] + 1):
        t_epoch = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch, config
        )

        # Evaluate
        eval_interval = config["logging"].get("eval_interval", 1)
        if epoch % eval_interval == 0 or epoch == config["training"]["epochs"]:
            val_metrics = evaluate_model(model, val_loader, device)
            ler_per_round = compute_ler_per_round(
                val_metrics["ler"], config["data"]["rounds"]
            )

            epoch_time = time.time() - t_epoch
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_ler={val_metrics['ler']:.4f} "
                f"ler/round={ler_per_round:.6f} | "
                f"{epoch_time:.1f}s"
            )

            # Save best model
            if val_metrics["ler"] < best_val_ler:
                best_val_ler = val_metrics["ler"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  -> New best model saved (val_ler={best_val_ler:.4f})")

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_ler": val_metrics["ler"],
                    "ler_per_round": ler_per_round,
                }
            )

    # Save final model and history
    torch.save(
        {
            "epoch": config["training"]["epochs"],
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        save_dir / "final_model.pt",
    )

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Best val LER: {best_val_ler:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
