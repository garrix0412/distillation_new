"""
在冻结的 Teacher 中间特征上训练辅助 probe heads。

Probe heads 学习从 Teacher 的 CNN 特征和解码器状态预测逻辑错误。
训练完成后，用于为 Stage 2 蒸馏生成 fused logits。

用法：
    python train_probes.py --teacher_checkpoint checkpoints/mock_teacher_d3/best_model.pt
"""

import argparse
import time

import torch
import torch.nn as nn
import yaml

from data.dataset import create_dataloaders
from distillation.probe_heads import ProbeHeadSet
from models.teacher import load_mock_teacher


def main():
    parser = argparse.ArgumentParser(description="Train probe heads on teacher features")
    parser.add_argument(
        "--teacher_checkpoint", type=str,
        default="checkpoints/mock_teacher_d3/best_model.pt",
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

    # 加载 teacher 配置以获取数据参数
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # 加载冻结的 teacher
    print("Loading frozen Teacher...")
    teacher = load_mock_teacher(
        checkpoint_path=args.teacher_checkpoint,
        distance=config["data"]["distance"],
        device=device,
    )
    teacher_dim = teacher.hidden_dim
    print(f"Teacher hidden_dim: {teacher_dim}")

    # 创建数据（与 teacher 训练相同）
    print("Generating data...")
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

    # 保存 probe heads
    save_path = args.teacher_checkpoint.replace("best_model.pt", "probe_heads.pt")
    torch.save(
        {"probe_heads_state_dict": probe_heads.state_dict(), "teacher_dim": teacher_dim},
        save_path,
    )
    print(f"\nProbe heads saved to: {save_path}")
    print(f"Final val accuracy: cnn={acc_cnn:.4f} rnn={acc_rnn:.4f} fused={acc_fused:.4f}")


if __name__ == "__main__":
    main()
