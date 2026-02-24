"""
Fused logits 蒸馏用辅助 probe heads。

轻量级 readout 头，将 Teacher 中间表征
（CNN 特征、解码器状态）映射到 logit 空间。
在冻结的 Teacher 特征上训练，用于构造融合软目标。

每个 probe head：在 stabilizer 上均值池化 → Linear → logit。
"""

import torch
import torch.nn as nn


class AuxiliaryProbeHead(nn.Module):
    """
    将 per-stabilizer 特征映射为单个 logit 的轻量级 probe。

    取最后一轮的中间表征，生成预测 P(逻辑错误) 的 logit，
    类似于主 readout 但简单得多（池化后仅一个线性层）。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        """
        Args:
            features: [batch, n_stab, hidden_dim] 最后一轮的特征

        Returns:
            logits: [batch, 1]
        """
        pooled = features.mean(dim=1)  # [batch, hidden_dim]
        return self.head(pooled)  # [batch, 1]


class ProbeHeadSet(nn.Module):
    """
    CNN 和 RNN 中间特征的 probe head 集合。

    从 teacher 中间表征产生三路 logit 流：
    - z_cnn: 来自 CNN 特征（最后一轮）
    - z_rnn: 来自解码器状态（最后一轮）
    - z_final: 来自 teacher 的 readout（直接透传，不在此处计算）

    融合 logit：z_fuse = (z_cnn + z_rnn + z_final) / 3
    """

    def __init__(self, teacher_dim: int):
        super().__init__()
        self.cnn_probe = AuxiliaryProbeHead(teacher_dim)
        self.rnn_probe = AuxiliaryProbeHead(teacher_dim)

    def forward(self, teacher_intermediates):
        """
        Args:
            teacher_intermediates: TeacherWrapper.forward_with_intermediates() 返回的字典，
                必须包含 'cnn_features'、'decoder_states'、'readout_logits'。

        Returns:
            包含 'cnn_logits'、'rnn_logits'、'fused_logits' 的字典
        """
        # 倒数第二轮特征：[batch, n_stab, hidden_dim]
        # 使用 -2 而非 -1，以避免边界轮 fallback 映射的问题
        cnn_last = teacher_intermediates["cnn_features"][:, -2, :, :]
        rnn_last = teacher_intermediates["decoder_states"][:, -2, :, :]
        z_final = teacher_intermediates["readout_logits"]

        z_cnn = self.cnn_probe(cnn_last)
        z_rnn = self.rnn_probe(rnn_last)
        z_fused = (z_cnn + z_rnn + z_final) / 3.0

        return {
            "cnn_logits": z_cnn,
            "rnn_logits": z_rnn,
            "fused_logits": z_fused,
        }
