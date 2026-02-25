"""
在冻结的 Teacher 中间特征上训练辅助 probe heads。

Probe heads 学习从 Teacher 的 CNN 特征和解码器状态预测逻辑错误。
训练完成后，用于为 Stage 2 蒸馏生成 fused logits。

用法：
    python train_probes.py --config configs/baseline_kd_d3.yaml
    python train_probes.py --config configs/baseline_kd_d3.yaml --save_dir checkpoints/probes/
    python train_probes.py --config configs/baseline_kd_d3.yaml --epochs 20 --lr 0.0005

只使用 YAML 中的 data 和 teacher 部分，忽略 model/distillation/training/logging。
因此可以直接复用任何 KD 配置文件。
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data.dataset import create_dataloaders
from distillation.probe_heads import ProbeHeadSet
from models.teacher import load_teacher


def main():
    parser = argparse.ArgumentParser(description="Train probe heads on teacher features")
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML 配置路径，包含 data + teacher 部分",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="probe_heads.pt 保存目录（默认从 teacher checkpoint 路径推导）",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 从 YAML 读配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    teacher_cfg = config.get("teacher", {})

    # 用统一入口加载冻结的 teacher
    print("Loading frozen Teacher...")
    teacher = load_teacher(teacher_cfg, distance=data_cfg["distance"], device=device)
    teacher_dim = teacher.hidden_dim
    print(f"Teacher hidden_dim: {teacher_dim}")

    # 创建数据
    online = data_cfg.get("online", False)
    print(f"Generating data (online={online})...")
    t0 = time.time()
    train_loader, val_loader = create_dataloaders(
        distance=data_cfg["distance"],
        rounds=data_cfg["rounds"],
        num_train=data_cfg["num_train"],
        num_val=data_cfg["num_val"],
        noise_strength=data_cfg["noise_strength"],
        snr=data_cfg["snr"],
        batch_size=data_cfg["batch_size"],
        use_soft=data_cfg["use_soft"],
        seed=data_cfg["seed"],
        online=online,
    )
    print(f"Data generated in {time.time()-t0:.1f}s")

    # 创建 probe heads
    probe_heads = ProbeHeadSet(teacher_dim=teacher_dim).to(device)
    n_params = sum(p.numel() for p in probe_heads.parameters())
    print(f"Probe heads: {n_params} parameters")

    # 优化器和损失
    optimizer = torch.optim.Adam(probe_heads.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # 训练循环
    print(f"\nTraining probe heads for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Online 模式：每个 epoch 重新采样训练数据
        if online:
            t_resample = time.time()
            train_loader.dataset.set_epoch(epoch)
            print(f"  [Online] Epoch {epoch}: resampled training data in {time.time()-t_resample:.1f}s")

        probe_heads.train()
        total_loss_cnn = 0.0
        total_loss_rnn = 0.0
        n_batches = 0

        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # 获取 teacher 中间表征（无梯度）
            _, intermediates = teacher.forward_with_intermediates(inputs)

            # Probe head 前向（有梯度）
            # 使用倒数第二轮，以避免边界轮映射问题
            cnn_last = intermediates["cnn_features"][:, -2, :, :]
            rnn_last = intermediates["decoder_states"][:, -2, :, :]

            z_cnn = probe_heads.cnn_probe(cnn_last).squeeze(-1)
            z_rnn = probe_heads.rnn_probe(rnn_last).squeeze(-1)

            loss_cnn = criterion(z_cnn, labels)
            loss_rnn = criterion(z_rnn, labels)
            loss = loss_cnn + loss_rnn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_cnn += loss_cnn.item()
            total_loss_rnn += loss_rnn.item()
            n_batches += 1

        # 验证
        probe_heads.eval()
        correct_cnn = correct_rnn = correct_fused = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                _, intermediates = teacher.forward_with_intermediates(inputs)
                fused_out = probe_heads(intermediates)

                z_cnn = fused_out["cnn_logits"].squeeze(-1)
                z_rnn = fused_out["rnn_logits"].squeeze(-1)
                z_fused = fused_out["fused_logits"].squeeze(-1)

                correct_cnn += ((z_cnn > 0).float() == labels).sum().item()
                correct_rnn += ((z_rnn > 0).float() == labels).sum().item()
                correct_fused += ((z_fused > 0).float() == labels).sum().item()
                total += labels.size(0)

        acc_cnn = correct_cnn / total
        acc_rnn = correct_rnn / total
        acc_fused = correct_fused / total

        print(
            f"Epoch {epoch:2d} | "
            f"loss_cnn={total_loss_cnn/n_batches:.4f} "
            f"loss_rnn={total_loss_rnn/n_batches:.4f} | "
            f"val_acc: cnn={acc_cnn:.4f} rnn={acc_rnn:.4f} fused={acc_fused:.4f}"
        )

    # 确定保存路径
    if args.save_dir:
        save_dir = args.save_dir
    else:
        # 从 teacher checkpoint 路径推导
        save_dir = str(Path(teacher_cfg["checkpoint"]).parent)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "probe_heads.pt")

    torch.save(
        {"probe_heads_state_dict": probe_heads.state_dict(), "teacher_dim": teacher_dim},
        save_path,
    )
    print(f"\nProbe heads saved to: {save_path}")
    print(f"Final val accuracy: cnn={acc_cnn:.4f} rnn={acc_rnn:.4f} fused={acc_fused:.4f}")


if __name__ == "__main__":
    main()
