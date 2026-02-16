"""
Evaluation metrics for quantum error correction decoders.

Primary metric: Logical Error Rate (LER) - the fraction of experiments
in which the decoder fails for each additional error-correction round.
"""

import numpy as np
import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute binary classification accuracy.

    Args:
        logits: [N, 1] or [N] raw logits
        labels: [N] binary labels

    Returns:
        Accuracy as a float in [0, 1].
    """
    logits = logits.squeeze(-1)
    preds = (logits > 0).float()
    return (preds == labels).float().mean().item()


def compute_logical_error_rate(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute the Logical Error Rate (LER).

    LER = fraction of samples where decoder prediction is wrong.
    This is 1 - accuracy for binary classification.

    For a more precise LER following the AlphaQubit paper (Eq. 2),
    the per-round LER epsilon is derived from the total error rate E(n):
        E(n) = 1/2 * (1 - (1 - 2*epsilon)^n)

    But for training evaluation, we use the simpler definition:
        LER = mean(prediction != label)

    Args:
        logits: [N, 1] or [N] raw logits
        labels: [N] binary labels

    Returns:
        LER as a float.
    """
    return 1.0 - compute_accuracy(logits, labels)


def compute_ler_per_round(
    total_error_rate: float,
    n_rounds: int,
) -> float:
    """
    Compute per-round LER epsilon from total error rate E(n).

    From AlphaQubit paper Eq. (4):
        epsilon = 1/2 * (1 - (1 - 2*E(n))^(1/n))

    Args:
        total_error_rate: E(n), fraction of failed experiments.
        n_rounds: Number of error-correction rounds.

    Returns:
        Per-round LER epsilon.
    """
    if total_error_rate >= 0.5:
        return 0.5  # Maximum error rate
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
    Compute error suppression ratio Lambda between two code distances.

    Lambda = LER(d) / LER(d+2), following the AlphaQubit paper.
    A value > 1 indicates error suppression with increasing distance.

    Args:
        ler_d1: LER at code distance d1 (smaller distance).
        ler_d2: LER at code distance d2 (larger distance).
        d1: Smaller code distance.
        d2: Larger code distance.

    Returns:
        Error suppression ratio.
    """
    if ler_d2 <= 0:
        return float("inf")
    return ler_d1 / ler_d2


@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluate a decoder model on a dataset.

    Args:
        model: The decoder model.
        dataloader: DataLoader providing (inputs, labels).
        device: Device to evaluate on.

    Returns:
        dict with:
            'accuracy': Classification accuracy
            'ler': Logical error rate
            'loss': Mean cross-entropy loss
            'n_samples': Number of samples evaluated
    """
    model.eval()
    all_logits = []
    all_labels = []

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0

    for inputs, labels in dataloader:
        # Move to device
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
