"""
知识蒸馏用 Teacher 模型接口。

本模块提供统一接口，用于在蒸馏管线中加载和使用 Teacher 模型
（Mock Teacher 或真实的 AlphaQubit 复现模型）。

Teacher 必须在预定义的 hook 点暴露中间表征：
  - Hook A: CNN/Transformer 特征（空间，逐轮）
  - Hook B: 解码器状态（时序，逐轮）
  - Hook C: Readout logits（决策）
  - Hook D: 最终 logits（输出）
"""

import torch
import torch.nn as nn

from .student import StudentDecoder, create_student
from distillation.probe_heads import ProbeHeadSet


class TeacherWrapper(nn.Module):
    """
    加载已训练模型并提供 Teacher 接口的封装器。

    对于 Mock Teacher，封装一个较大的 StudentDecoder。
    对于真实 AlphaQubit Teacher，应替换为
    对实际 AlphaQubit 模型的封装（待 Teacher 模型就绪后实现）。

    核心接口是 `forward_with_intermediates()`，
    同时返回预测结果和用于 KD 的内部表征。
    """

    def __init__(self, model: nn.Module, probe_heads: ProbeHeadSet = None):
        super().__init__()
        self.model = model
        self.probe_heads = probe_heads
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 同时冻结 probe heads（已预训练）
        if self.probe_heads is not None:
            for param in self.probe_heads.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, inputs):
        """标准前向传播（仅推理）。"""
        self.model.eval()
        return self.model(inputs)

    @torch.no_grad()
    def forward_with_intermediates(self, inputs):
        """
        返回中间表征的前向传播，用于知识蒸馏。

        Args:
            inputs: 字典，包含 'detection_events' 和可选的 'soft_events'

        Returns:
            logits: [batch, 1]
            intermediates: 字典
                'cnn_features': [batch, rounds, n_stab, hidden_dim]
                    每轮 CNN 输出（Hook A：空间特征）
                'decoder_states': [batch, rounds, n_stab, hidden_dim]
                    每轮解码器状态（Hook B：时序特征）
                'readout_features': [batch, readout_dim]
                    Readout 内部特征
                'readout_logits': [batch, 1]
                    输出 logits（Hook C/D）
        """
        self.model.eval()
        logits, intermediates = self.model(inputs, return_intermediates=True)
        # 分离所有中间表征，确保梯度不会流向 teacher
        intermediates = {
            k: v.detach() for k, v in intermediates.items()
        }
        # 若 probe heads 可用，计算 fused logits
        if self.probe_heads is not None:
            self.probe_heads.eval()
            fused_outputs = self.probe_heads(intermediates)
            for k, v in fused_outputs.items():
                intermediates[k] = v.detach()
        return logits.detach(), intermediates

    @property
    def hidden_dim(self):
        return self.model.hidden_dim

    @property
    def readout_dim(self):
        return self.model.readout_dim


def load_mock_teacher(
    checkpoint_path: str,
    distance: int,
    device: str = "cpu",
) -> TeacherWrapper:
    """
    从 checkpoint 加载已训练的 Mock Teacher。

    Args:
        checkpoint_path: 已保存模型 checkpoint 的路径。
        distance: 码距。
        device: 加载模型的设备。

    Returns:
        参数已冻结的 TeacherWrapper。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    model = create_student(
        distance=distance,
        size=config["model"]["size"],
        rnn_type=config["model"]["rnn_type"],
        use_soft=config["data"]["use_soft"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return TeacherWrapper(model)


def load_mock_teacher_with_probes(
    checkpoint_path: str,
    probe_heads_path: str,
    distance: int,
    device: str = "cpu",
) -> TeacherWrapper:
    """
    加载带有预训练 probe heads 的 Mock Teacher，用于 fused logits。

    Args:
        checkpoint_path: 已保存 teacher 模型 checkpoint 的路径。
        probe_heads_path: 已保存 probe heads checkpoint 的路径。
        distance: 码距。
        device: 加载模型的设备。

    Returns:
        带有冻结参数和 probe heads 的 TeacherWrapper。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    model = create_student(
        distance=distance,
        size=config["model"]["size"],
        rnn_type=config["model"]["rnn_type"],
        use_soft=config["data"]["use_soft"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    probe_ckpt = torch.load(probe_heads_path, map_location=device, weights_only=False)
    probe_heads = ProbeHeadSet(teacher_dim=model.hidden_dim)
    probe_heads.load_state_dict(probe_ckpt["probe_heads_state_dict"])
    probe_heads = probe_heads.to(device)

    return TeacherWrapper(model, probe_heads=probe_heads)
