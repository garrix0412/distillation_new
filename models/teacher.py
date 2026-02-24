"""
知识蒸馏用 Teacher 模型接口。

本模块提供统一接口，用于在蒸馏管线中加载和使用 Teacher 模型
（Mock Teacher 或真实的 AlphaQubit 复现模型）。

Teacher 必须在预定义的 hook 点暴露中间表征：
  - Hook A: CNN/Transformer 特征（空间，逐轮）
  - Hook B: 解码器状态（时序，逐轮）
  - Hook C: Readout logits（决策）
  - Hook D: 最终 logits（输出）

接口规范：
  所有 Teacher 适配器必须继承 TeacherAdapter 抽象基类，
  实现 forward_with_intermediates() 方法以及 hidden_dim / readout_dim 属性。
  使用 load_teacher(teacher_config, distance, device) 统一加载。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .student import StudentDecoder, create_student
from distillation.probe_heads import ProbeHeadSet


class TeacherAdapter(ABC):
    """
    Teacher 模型的抽象基类接口。

    任何外部 Teacher 模型（如真实 AlphaQubit）需实现此接口以接入蒸馏管线。
    Mock Teacher（TeacherWrapper）已实现此接口。

    接口契约:
        forward_with_intermediates(inputs) -> (logits, intermediates)

        inputs: dict，至少包含以下键之一：
            'detection_events': Tensor [batch, rounds, n_stab]  硬检测事件
            'soft_events':      Tensor [batch, rounds, n_stab]  软测量后验（可选）

        返回值:
            logits: Tensor [batch, 1]  预测逻辑错误概率的 logit
            intermediates: dict，包含以下键：
                'cnn_features':     Tensor [batch, rounds, n_stab, hidden_dim]
                    每轮 CNN/空间混合后的特征（Hook A）
                'decoder_states':   Tensor [batch, rounds, n_stab, hidden_dim]
                    每轮 RNN/时序状态（Hook B）
                'readout_features': Tensor [batch, readout_dim]
                    Readout 网络的内部特征
                'readout_logits':   Tensor [batch, 1]
                    最终输出 logits（Hook C/D）
                （可选）'fused_cnn_logits': Tensor [batch, 1]
                    Probe head 的 CNN fused logit（仅 mock teacher with probes）
                （可选）'fused_rnn_logits': Tensor [batch, 1]
                    Probe head 的 RNN fused logit（仅 mock teacher with probes）

    属性:
        hidden_dim: int  Teacher 的隐藏维度（用于特征 KD 投影层）
        readout_dim: int  Teacher 的 readout 维度（用于 readout 特征 KD 投影层）
    """

    @abstractmethod
    def forward_with_intermediates(self, inputs: dict):
        """
        返回中间表征的前向传播，用于知识蒸馏。

        Args:
            inputs: 字典，包含 'detection_events' 和可选的 'soft_events'

        Returns:
            logits: Tensor [batch, 1]
            intermediates: 字典（详见类文档）
        """
        ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Teacher 的隐藏维度，用于 CNN/RNN 特征 KD 的投影层匹配。"""
        ...

    @property
    @abstractmethod
    def readout_dim(self) -> int:
        """Teacher 的 readout 维度，用于 readout 特征 KD 的投影层匹配。"""
        ...


class TeacherWrapper(nn.Module, TeacherAdapter):
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


class AlphaQubitAdapter(nn.Module, TeacherAdapter):
    """
    真实 AlphaQubit 模型的适配器模板。

    组员需复制此类并填写实现。主要需要完成：
    1. __init__: 加载 AlphaQubit 模型权重
    2. forward_with_intermediates: 将 AlphaQubit 的输出映射到蒸馏管线的标准格式
    3. hidden_dim / readout_dim: 返回模型对应维度

    配置格式（在 YAML 的 teacher 部分）：
        teacher:
          type: "alphaqubit"
          checkpoint: "path/to/alphaqubit_checkpoint.pt"
          hidden_dim: 256     # AlphaQubit 的隐藏维度
          readout_dim: 128    # AlphaQubit 的 readout 维度

    使用示例：
        teacher = load_teacher(config["teacher"], distance=3, device="cuda")
        logits, intermediates = teacher.forward_with_intermediates(inputs)
    """

    def __init__(self, checkpoint_path: str, hidden_dim: int, readout_dim: int,
                 distance: int, device: str = "cpu"):
        """
        初始化 AlphaQubit 适配器。

        Args:
            checkpoint_path: AlphaQubit 模型 checkpoint 路径
            hidden_dim: 模型隐藏维度（用于特征 KD 投影匹配）
            readout_dim: 模型 readout 维度（用于 readout 特征 KD 投影匹配）
            distance: 表面码码距
            device: 加载设备
        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._readout_dim = readout_dim

        # TODO: 在此加载真实 AlphaQubit 模型
        # self.model = AlphaQubitModel.load(checkpoint_path)
        # self.model.to(device)
        # self.model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        raise NotImplementedError(
            "AlphaQubitAdapter 是模板类，需组员填写实现。\n"
            "请参考类文档完成以下步骤：\n"
            "  1. 在 __init__ 中加载 AlphaQubit 模型\n"
            "  2. 实现 forward_with_intermediates() 方法\n"
            "  3. 确认 hidden_dim 和 readout_dim 与模型匹配"
        )

    @torch.no_grad()
    def forward_with_intermediates(self, inputs):
        """
        将 AlphaQubit 的输出映射到蒸馏管线的标准 intermediates 格式。

        需要映射的键（参考 TeacherAdapter 文档）：
            'cnn_features':     [batch, rounds, n_stab, hidden_dim]
            'decoder_states':   [batch, rounds, n_stab, hidden_dim]
            'readout_features': [batch, readout_dim]
            'readout_logits':   [batch, 1]

        如果 AlphaQubit 没有某些中间层，可以：
        - 用零张量填充（对应的 gamma 权重设为 0 即可跳过该信号）
        - 或者用最接近的中间表征替代
        """
        # TODO: 实现前向传播和中间表征提取
        # 示例骨架：
        # output = self.model(inputs)
        # logits = output['logits']  # [batch, 1]
        # intermediates = {
        #     'cnn_features': output['spatial_features'],      # [B, R, N, H]
        #     'decoder_states': output['temporal_states'],      # [B, R, N, H]
        #     'readout_features': output['readout_features'],   # [B, readout_dim]
        #     'readout_logits': logits,                         # [B, 1]
        # }
        # return logits, intermediates
        raise NotImplementedError("需实现 forward_with_intermediates()")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def readout_dim(self) -> int:
        return self._readout_dim


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


def load_teacher(teacher_config: dict, distance: int, device="cpu") -> TeacherAdapter:
    """
    根据配置统一加载 Teacher 模型。

    根据 teacher_config["type"] 分派不同的加载逻辑：
    - "mock"（默认）：加载 Mock Teacher（大号 StudentDecoder）
    - "alphaqubit"：加载 AlphaQubit 适配器

    Args:
        teacher_config: 配置字典（YAML 的 teacher 部分），支持的键：
            type: str = "mock"          Teacher 类型
            checkpoint: str             模型 checkpoint 路径
            probe_heads: str (可选)     Probe heads checkpoint 路径（仅 mock）
            hidden_dim: int (可选)      外部 teacher 隐藏维度（仅 alphaqubit）
            readout_dim: int (可选)     外部 teacher readout 维度（仅 alphaqubit）
        distance: 码距
        device: 加载设备

    Returns:
        实现了 TeacherAdapter 接口的 Teacher 实例
    """
    teacher_type = teacher_config.get("type", "mock")

    if teacher_type == "mock":
        checkpoint_path = teacher_config["checkpoint"]
        probe_heads_path = teacher_config.get("probe_heads", None)
        if probe_heads_path:
            return load_mock_teacher_with_probes(
                checkpoint_path=checkpoint_path,
                probe_heads_path=probe_heads_path,
                distance=distance,
                device=device,
            )
        else:
            return load_mock_teacher(
                checkpoint_path=checkpoint_path,
                distance=distance,
                device=device,
            )

    elif teacher_type == "alphaqubit":
        return AlphaQubitAdapter(
            checkpoint_path=teacher_config["checkpoint"],
            hidden_dim=teacher_config["hidden_dim"],
            readout_dim=teacher_config["readout_dim"],
            distance=distance,
            device=device,
        )

    else:
        raise ValueError(
            f"未知的 teacher 类型: '{teacher_type}'\n"
            f"支持的类型: 'mock', 'alphaqubit'"
        )
