"""
知识蒸馏训练脚本。

支持：
- Baseline KD（任务 1）：仅响应 KD（soft logit 匹配）
- Stage 1 KD（任务 2）：多信号 KD（响应 + CNN 特征 + RNN 特征）
- Stage 2 KD（任务 4）：Fused logits KD

用法：
    python train_distill.py --config configs/baseline_kd_d3.yaml
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data.dataset import create_dataloaders
from distillation.losses import DistillationLoss
from evaluation.metrics import evaluate_model, compute_ler_per_round
from models.student import create_student
from models.teacher import load_mock_teacher, load_mock_teacher_with_probes


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


def train_one_epoch_kd(
    student, teacher, train_loader, optimizer, scheduler,
    criterion, device, epoch, config,
):
    student.train()
    total_loss = 0.0
    correct = 0
    total = 0
    loss_accum = {}
    log_interval = config["logging"]["log_interval"]

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        # Teacher 前向（无梯度）
        teacher_logits, teacher_intermediates = teacher.forward_with_intermediates(inputs)

        # Student 前向（带中间表征）
        student_logits, student_intermediates = student(inputs, return_intermediates=True)

        # 组合损失
        loss, loss_dict = criterion(
            student_logits=student_logits,
            labels=labels,
            teacher_logits=teacher_logits,
            student_intermediates=student_intermediates,
            teacher_intermediates=teacher_intermediates,
        )

        loss.backward()

        grad_clip = config["training"].get("grad_clip", 0)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(student.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = (student_logits.squeeze(-1) > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 累计各损失分量
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / total
            lr = optimizer.param_groups[0]["lr"]
            loss_str = " ".join(
                f"{k}={v/(batch_idx+1):.4f}" for k, v in loss_accum.items() if k != "total"
            )
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"loss={avg_loss:.4f} acc={acc:.4f} lr={lr:.2e} | {loss_str}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    avg_losses = {k: v / len(train_loader) for k, v in loss_accum.items()}
    return {"loss": avg_loss, "accuracy": accuracy, "loss_components": avg_losses}


def build_scheduler(optimizer, config, total_steps):
    import math
    sched_type = config["training"].get("scheduler", "none")
    warmup_steps = config["training"].get("warmup_steps", 0)

    if sched_type == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return None


def main():
    parser = argparse.ArgumentParser(description="Train with knowledge distillation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    # 创建数据加载器
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

    # 加载 Teacher
    print("Loading Teacher model...")
    probe_heads_path = config.get("teacher", {}).get("probe_heads", None)
    if probe_heads_path:
        teacher = load_mock_teacher_with_probes(
            checkpoint_path=config["teacher"]["checkpoint"],
            probe_heads_path=probe_heads_path,
            distance=config["data"]["distance"],
            device=device,
        )
        print("Teacher loaded with probe heads (fused logits enabled)")
    else:
        teacher = load_mock_teacher(
            checkpoint_path=config["teacher"]["checkpoint"],
            distance=config["data"]["distance"],
            device=device,
        )
    teacher_dim = teacher.hidden_dim
    print(f"Teacher hidden_dim={teacher_dim}")

    # 评估 teacher 展示基线
    teacher_metrics = evaluate_model(teacher.model, val_loader, device)
    print(f"Teacher val LER: {teacher_metrics['ler']:.4f}")

    # 创建 Student
    student = create_student(
        distance=config["data"]["distance"],
        size=config["model"]["size"],
        rnn_type=config["model"]["rnn_type"],
        use_soft=config["data"]["use_soft"],
    )

    # 可选：从 Stage 1 checkpoint 初始化（用于 Stage 2）
    init_checkpoint = config.get("model", {}).get("init_checkpoint", None)
    if init_checkpoint:
        print(f"Loading student initialization from {init_checkpoint}")
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])

    student = student.to(device)
    student_dim = student.hidden_dim
    student_readout_dim = student.readout_dim
    teacher_readout_dim = teacher.readout_dim
    print(f"Student: {config['model']['size']} ({config['model']['rnn_type']}), "
          f"hidden_dim={student_dim}, readout_dim={student_readout_dim}, "
          f"{student.count_parameters():,} params")

    # 构建蒸馏损失
    kd_config = config.get("distillation", {})
    criterion = DistillationLoss(
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        alpha=kd_config.get("alpha", 0.5),
        beta=kd_config.get("beta", 0.5),
        gamma_cnn=kd_config.get("gamma_cnn", 0.0),
        gamma_rnn=kd_config.get("gamma_rnn", 0.0),
        gamma_readout=kd_config.get("gamma_readout", 0.0),
        gamma_fused=kd_config.get("gamma_fused", 0.0),
        temperature=kd_config.get("temperature", 1.0),
        feature_loss_type=kd_config.get("feature_loss_type", "mse"),
        student_readout_dim=student_readout_dim,
        teacher_readout_dim=teacher_readout_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(criterion.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = build_scheduler(optimizer, config, total_steps)

    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    best_val_ler = 1.0
    history = []

    print(f"\n{'='*60}")
    print(f"Starting KD training for {config['training']['epochs']} epochs")
    print(f"Loss weights: alpha={kd_config.get('alpha',0.5)} "
          f"beta={kd_config.get('beta',0.5)} "
          f"gamma_cnn={kd_config.get('gamma_cnn',0.0)} "
          f"gamma_rnn={kd_config.get('gamma_rnn',0.0)} "
          f"gamma_readout={kd_config.get('gamma_readout',0.0)} "
          f"gamma_fused={kd_config.get('gamma_fused',0.0)}")
    print(f"{'='*60}\n")

    for epoch in range(1, config["training"]["epochs"] + 1):
        t_epoch = time.time()

        train_metrics = train_one_epoch_kd(
            student, teacher, train_loader, optimizer, scheduler,
            criterion, device, epoch, config,
        )

        eval_interval = config["logging"].get("eval_interval", 1)
        if epoch % eval_interval == 0 or epoch == config["training"]["epochs"]:
            val_metrics = evaluate_model(student, val_loader, device)
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

            if val_metrics["ler"] < best_val_ler:
                best_val_ler = val_metrics["ler"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  -> New best model saved (val_ler={best_val_ler:.4f})")

            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_ler": val_metrics["ler"],
                "ler_per_round": ler_per_round,
                "loss_components": train_metrics.get("loss_components", {}),
            })

    torch.save(
        {"epoch": config["training"]["epochs"],
         "model_state_dict": student.state_dict(), "config": config},
        save_dir / "final_model.pt",
    )
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"KD Training complete.")
    print(f"Teacher val LER: {teacher_metrics['ler']:.4f}")
    print(f"Student best val LER: {best_val_ler:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
