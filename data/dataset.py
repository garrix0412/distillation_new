"""
Surface code 解码数据的 PyTorch Dataset 和 DataLoader。

支持两种数据生成模式：
- offline（默认）：初始化时一次性生成所有数据，每个 epoch 复用
- online：每个 epoch 通过 set_epoch() 重新采样全新数据
"""

import torch
from torch.utils.data import Dataset, DataLoader

from .stim_generator import (
    generate_surface_code_data,
    prepare_surface_code_context,
    sample_from_context,
)


class SurfaceCodeDataset(Dataset):
    """
    Surface code memory 实验数据的 PyTorch Dataset。

    每个样本包含逐轮、逐 stabilizer 的输入，
    以及一个指示是否发生逻辑错误的二值标签。

    Args:
        online: 如果为 True，每个 epoch 调用 set_epoch() 重新采样数据。
                验证集应始终使用 online=False 以保证评估公平性。
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
        online: bool = False,
    ):
        self.distance = distance
        self.rounds = rounds
        self.use_soft = use_soft
        self.num_samples = num_samples
        self.noise_strength = noise_strength
        self.snr = snr
        self.base_seed = seed
        self.online = online

        if online:
            # 一次性预计算电路和排列，缓存 context
            self._context = prepare_surface_code_context(
                distance, rounds, noise_strength
            )
            self.n_stabilizers = self._context["n_stabilizers"]
            # 用 base_seed 生成初始数据（epoch 0）
            self._load_data(seed)
        else:
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
            self._store_tensors(data)

    def _store_tensors(self, data: dict):
        """将 numpy 数据字典转为 torch 张量并存储。"""
        # detection_events: [N, rounds, n_stab] 二值
        self.detection_events = torch.from_numpy(data["detection_events"])
        # labels: [N] 二值（是否发生逻辑错误）
        self.labels = torch.from_numpy(data["logical_observables"])
        if self.use_soft:
            # soft_events: [N, rounds, n_stab] 后验概率
            self.soft_events = torch.from_numpy(data["soft_events"])

    def _load_data(self, seed: int):
        """从 context 采样数据并存储为张量（online 模式使用）。"""
        data = sample_from_context(
            self._context,
            num_shots=self.num_samples,
            snr=self.snr,
            use_soft=self.use_soft,
            seed=seed,
        )
        self._store_tensors(data)

    def set_epoch(self, epoch: int):
        """
        重新生成训练数据（仅 online 模式生效）。

        使用 seed = base_seed + epoch，确保：
        - 不同 epoch 数据不同
        - 相同 epoch 可确定性复现（支持断点恢复）

        Args:
            epoch: 当前 epoch 编号（1-based）。
        """
        if not self.online:
            return
        seed = self.base_seed + epoch
        self._load_data(seed)

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
    online: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader。

    Args:
        online: 训练集是否启用 online 模式（每 epoch 重新采样）。
                验证集始终使用 offline 模式以保证评估公平性。

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
        online=online,
    )

    val_dataset = SurfaceCodeDataset(
        distance=distance,
        rounds=rounds,
        num_samples=num_val,
        noise_strength=noise_strength,
        snr=snr,
        use_soft=use_soft,
        seed=seed + 10000,  # 验证集使用不同的种子
        online=False,  # 验证集始终 offline
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
