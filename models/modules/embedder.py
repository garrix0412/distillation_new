"""
Stabilizer 嵌入器模块。

将每个 stabilizer 的输入特征（检测事件、soft readout）
映射为隐藏表征向量。

"""

import torch
import torch.nn as nn


class StabilizerEmbedder(nn.Module):
    """
    将每个 stabilizer 的输入嵌入为隐藏表征。

    对每个 stabilizer 的每一轮，接收输入特征
    （检测事件、soft readout 概率）并生成
    维度为 `hidden_dim` 的隐藏向量。

    架构：
    - 对每个输入特征分别做线性投影
    - 将投影结果求和
    - 加上可学习的 stabilizer 索引嵌入
    - 通过残差模块
    """

    def __init__(
        self,
        n_stabilizers: int,
        hidden_dim: int,
        n_input_features: int = 2,
        use_soft: bool = True,
    ):
        """
        Args:
            n_stabilizers: stabilizer 数量（d^2 - 1）。
            hidden_dim: 每个 stabilizer 输出嵌入的维度。
            n_input_features: 每个 stabilizer 的输入特征数。
                use_soft=True 时为 2（detection_event, soft_event），
                use_soft=False 时为 1（detection_event）。
            use_soft: 是否提供 soft readout 作为输入。
        """
        super().__init__()
        self.n_stabilizers = n_stabilizers
        self.hidden_dim = hidden_dim
        self.use_soft = use_soft
        self.n_input_features = 2 if use_soft else 1

        # 对每个输入特征的独立线性投影
        # 每个将标量映射到 hidden_dim
        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(self.n_input_features)]
        )

        # 可学习的 stabilizer 索引嵌入
        self.stabilizer_embedding = nn.Embedding(n_stabilizers, hidden_dim)

        # 用于混合的残差模块
        self.res_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, detection_events, soft_events=None):
        """
        Args:
            detection_events: [batch, n_stabilizers] 单轮二值检测事件
            soft_events: [batch, n_stabilizers] 单轮 soft 概率（可选）

        Returns:
            embeddings: [batch, n_stabilizers, hidden_dim]
        """
        batch_size = detection_events.shape[0]

        # 投影每个特征：[batch, n_stab] -> [batch, n_stab, 1] -> [batch, n_stab, hidden_dim]
        features = [self.feature_projections[0](detection_events.unsqueeze(-1))]

        if self.use_soft and soft_events is not None:
            features.append(self.feature_projections[1](soft_events.unsqueeze(-1)))

        # 投影求和：[batch, n_stab, hidden_dim]
        h = torch.stack(features, dim=0).sum(dim=0)

        # 加上 stabilizer 索引嵌入
        stab_indices = torch.arange(self.n_stabilizers, device=h.device)
        stab_embed = self.stabilizer_embedding(stab_indices)  # [n_stab, hidden_dim]
        h = h + stab_embed.unsqueeze(0)  # 在 batch 维度广播

        # 残差模块
        h = h + self.res_block(h)

        return h
