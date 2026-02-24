"""
Readout 网络模块。

接收最终解码器状态，预测是否发生了逻辑错误。
AlphaQubit readout 的简化版本。
"""

import torch
import torch.nn as nn


class ReadoutNetwork(nn.Module):
    """
    将解码器状态映射到逻辑错误预测的 readout 网络。

    AlphaQubit readout 的简化版本：
    1. 线性投影（降维）
    2. 在 stabilizer 上做均值池化
    3. 小型残差网络
    4. 最终 logit 输出

    AlphaQubit 使用 scatter → 2x2 conv → 行/列池化 → ResNet。
    我们的简化版本直接对所有 stabilizer 池化，
    更适合 FPGA 且对小码距足够。
    """

    def __init__(
        self,
        hidden_dim: int,
        readout_dim: int = 32,
        n_layers: int = 2,
        distance: int = 3,
    ):
        """
        Args:
            hidden_dim: 每个 stabilizer 的输入维度。
            readout_dim: readout 处理的内部维度。
            n_layers: 残差层数。
            distance: 码距（用于逐线预测）。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_dim = readout_dim
        self.distance = distance

        # 从 hidden_dim 投影到 readout_dim
        self.proj = nn.Linear(hidden_dim, readout_dim)

        # 残差模块
        self.res_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.res_blocks.append(
                nn.Sequential(
                    nn.LayerNorm(readout_dim),
                    nn.Linear(readout_dim, readout_dim),
                    nn.ReLU(),
                    nn.Linear(readout_dim, readout_dim),
                )
            )

        # 最终分类头
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, 1),
        )

    def forward(self, decoder_state, return_features=False):
        """
        Args:
            decoder_state: [batch, n_stab, hidden_dim] 最终解码器状态
            return_features: 若为 True，同时返回中间特征

        Returns:
            logits: [batch, 1] P(逻辑错误) 的 logit
            features: [batch, readout_dim]（仅当 return_features=True 时）
        """
        # 投影到 readout 维度
        h = self.proj(decoder_state)  # [batch, n_stab, readout_dim]

        # 在 stabilizer 上均值池化
        h = h.mean(dim=1)  # [batch, readout_dim]

        # 残差模块
        for block in self.res_blocks:
            h = h + block(h)

        features = h

        # 最终 logit
        logits = self.head(h)  # [batch, 1]

        if return_features:
            return logits, features
        return logits
