"""
量子纠错解码器的评估指标。

主要指标：逻辑错误率（LER）— 每增加一轮纠错，
解码器失败的实验比例。
"""

import numpy as np
import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算二分类准确率。

    Args:
        logits: [N, 1] 或 [N] 原始 logits
        labels: [N] 二值标签

    Returns:
        准确率，浮点数，范围 [0, 1]。
    """
    logits = logits.squeeze(-1)
    preds = (logits > 0).float()
    return (preds == labels).float().mean().item()


def compute_logical_error_rate(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    计算逻辑错误率（LER）。

    LER = 解码器预测错误的样本比例。
    对于二分类即 1 - 准确率。

    根据 AlphaQubit 论文（公式 2），更精确的逐轮 LER epsilon
    由总错误率 E(n) 推导：
        E(n) = 1/2 * (1 - (1 - 2*epsilon)^n)

    但在训练评估中，我们使用更简单的定义：
        LER = mean(prediction != label)

    Args:
        logits: [N, 1] 或 [N] 原始 logits
        labels: [N] 二值标签

    Returns:
        LER 浮点数。
    """
    return 1.0 - compute_accuracy(logits, labels)


def compute_ler_per_round(
    total_error_rate: float,
    n_rounds: int,
) -> float:
    """
    从总错误率 E(n) 计算逐轮 LER epsilon。

    来自 AlphaQubit 论文公式 (4)：
        epsilon = 1/2 * (1 - (1 - 2*E(n))^(1/n))

    Args:
        total_error_rate: E(n)，失败实验的比例。
        n_rounds: 纠错轮数。

    Returns:
        逐轮 LER epsilon。
    """
    if total_error_rate >= 0.5:
        return 0.5  # 最大错误率
    if total_error_rate <= 0:
        return 0.0

    fidelity = 1.0 - 2.0 * total_error_rate
    epsilon = 0.5 * (1.0 - np.power(fidelity, 1.0 / n_rounds))
    return float(epsilon)


def compute_error_suppression_ratio(
    ler_d1: float,
    ler_d2: float,
    d1: int,
    d2: int,
) -> float:
    """
    计算两个码距之间的错误抑制比 Lambda。

    Lambda = LER(d) / LER(d+2)，沿用 AlphaQubit 论文。
    值 > 1 表示随码距增大错误被抑制。

    Args:
        ler_d1: 较小码距 d1 的 LER。
        ler_d2: 较大码距 d2 的 LER。
        d1: 较小码距。
        d2: 较大码距。

    Returns:
        错误抑制比。
    """
    if ler_d2 <= 0:
        return float("inf")
    return ler_d1 / ler_d2


@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu"):
    """
    在数据集上评估解码器模型。

    Args:
        model: 解码器模型。
        dataloader: 提供 (inputs, labels) 的 DataLoader。
        device: 评估设备。

    Returns:
        包含以下键的字典：
            'accuracy': 分类准确率
            'ler': 逻辑错误率
            'loss': 平均交叉熵损失
            'n_samples': 评估的样本数
    """
    model.eval()
    all_logits = []
    all_labels = []

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0

    for inputs, labels in dataloader:
        # 移至设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        logits = model(inputs)
        logits = logits.squeeze(-1)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy = compute_accuracy(all_logits, all_labels)
    ler = compute_logical_error_rate(all_logits, all_labels)
    mean_loss = total_loss / max(n_batches, 1)

    return {
        "accuracy": accuracy,
        "ler": ler,
        "loss": mean_loss,
        "n_samples": len(all_labels),
    }
