"""
Surface code 解码数据的 PyTorch Dataset 和 DataLoader。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .stim_generator import generate_surface_code_data


class SurfaceCodeDataset(Dataset):
    """
    Surface code memory 实验数据的 PyTorch Dataset。

    每个样本包含逐轮、逐 stabilizer 的输入，
    以及一个指示是否发生逻辑错误的二值标签。
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        num_samples: int,
        noise_strength: float = 0.001,
        snr: float = 10.0,
        use_soft: bool = True,
        seed: int = 42,
    ):
        self.distance = distance
        self.rounds = rounds
        self.use_soft = use_soft

        data = generate_surface_code_data(
            distance=distance,
            rounds=rounds,
            num_shots=num_samples,
            noise_strength=noise_strength,
            snr=snr,
            use_soft=use_soft,
            seed=seed,
        )

        self.n_stabilizers = data["n_stabilizers"]

        # 存储为张量
        # detection_events: [N, rounds, n_stab] 二值
        self.detection_events = torch.from_numpy(data["detection_events"])
        # labels: [N] 二值（是否发生逻辑错误）
        self.labels = torch.from_numpy(data["logical_observables"])

        if use_soft:
            # soft_events: [N, rounds, n_stab] 后验概率
            self.soft_events = torch.from_numpy(data["soft_events"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            inputs: 字典
                'detection_events': [rounds, n_stab] 二值 float32
                'soft_events': [rounds, n_stab] float32（如果 use_soft）
            label: float32 标量
        """
        inputs = {
            "detection_events": self.detection_events[idx],
        }
        if self.use_soft:
            inputs["soft_events"] = self.soft_events[idx]

        return inputs, self.labels[idx]


def create_dataloaders(
    distance: int,
    rounds: int,
    num_train: int = 100000,
    num_val: int = 10000,
    noise_strength: float = 0.001,
    snr: float = 10.0,
    batch_size: int = 256,
    use_soft: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader。

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = SurfaceCodeDataset(
        distance=distance,
        rounds=rounds,
        num_samples=num_train,
        noise_strength=noise_strength,
        snr=snr,
        use_soft=use_soft,
        seed=seed,
    )

    val_dataset = SurfaceCodeDataset(
        distance=distance,
        rounds=rounds,
        num_samples=num_val,
        noise_strength=noise_strength,
        snr=snr,
        use_soft=use_soft,
        seed=seed + 10000,  # 验证集使用不同的种子
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
