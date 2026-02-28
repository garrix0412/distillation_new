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
from .Transformer_0225_fixed import FullMapper, AlphaQubitV2
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
    对于真实 AlphaQubit Teacher，应替换为对实际 AlphaQubit 模型的封装（待 Teacher 模型就绪后实现）。

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
    AlphaQubitV2 模型的蒸馏适配器。

    将组员复现的 AlphaQubitV2（X+Z 联合处理，RNN+TF 交替架构）
    接入蒸馏管线。处理输入格式桥接和中间特征提取。

    关键差异：
    - 输入：AlphaQubitV2 接收 raw flat detectors [B, num_detectors]，
      而 Student 接收 reshape+permuted 的 [B, rounds, n_stab]
    - 空间：Teacher 使用全部 stabilizer（num_z + num_x），Student 使用 n_stab
    - 时间：Teacher 有 num_t 时间步，Student 有 rounds 步，两者可能不同
    - 中间特征形状不同，CNN/RNN 特征 KD 不兼容（gamma_cnn=0, gamma_rnn=0）
    - Response KD 和 readout feature KD 完全兼容

    配置格式（在 YAML 的 teacher 部分）：
        teacher:
          type: "alphaqubit"
          checkpoint: "path/to/alphaqubit_checkpoint.pt"
          hidden_dim: 256     # d_model
          readout_dim: 256    # d_model（readout 输出也是 d_model 维）
          # n_heads: 8        # 可选，默认 8
    """

    def __init__(self, checkpoint_path: str, hidden_dim: int, readout_dim: int,
                 distance: int, rounds: int, device: str = "cpu",
                 n_heads: int = None):
        """
        初始化 AlphaQubitV2 适配器。

        Args:
            checkpoint_path: AlphaQubitV2 模型 checkpoint 路径
            hidden_dim: 模型 d_model 维度
            readout_dim: 模型 readout 维度（AlphaQubitV2 中等于 d_model）
            distance: 表面码码距
            rounds: 纠错轮数（需与 checkpoint 训练时一致）
            device: 加载设备
            n_heads: 注意力头数（可选，默认 8）
        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._readout_dim = readout_dim

        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 提取 state_dict（支持多种 checkpoint 格式）
        if "model_state" in checkpoint:
            raw_state = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            raw_state = checkpoint["model_state_dict"]
        else:
            # 假设 checkpoint 本身就是 state_dict
            raw_state = checkpoint

        # 清理 key 前缀（DDP / torch.compile 产生的前缀）
        clean_state = {}
        for k, v in raw_state.items():
            new_k = k.replace("_orig_mod.", "").replace("module.", "")
            clean_state[new_k] = v

        d_model = hidden_dim
        if n_heads is None:
            n_heads = 8

        # 重建 FullMapper（用当前 distance 和 rounds 重新构建拓扑映射）
        mapper = FullMapper(distance, rounds)

        # 构建模型（V2 架构固定，无 n_layers 参数）
        self.model = AlphaQubitV2(
            mapper, d_model=d_model, n_heads=n_heads
        )

        # 加载权重，排除 mapper 相关 buffer（由 FullMapper 重新构建）
        exclude_prefixes = [
            'gather_z', 'valid_z', 'z_neighbors', 'z_hint_neighbors',
            'gather_x', 'valid_x', 'x_neighbors', 'x_hint_neighbors',
            'spatial_coords',
        ]
        filtered_state = {
            k: v for k, v in clean_state.items()
            if not any(k.startswith(prefix) or k.endswith(prefix) for prefix in exclude_prefixes)
        }
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.to(device)
        self.model.eval()

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward_with_intermediates(self, inputs):
        """
        从 inputs['raw_detectors'] 取原始平展检测事件，调用 AlphaQubitV2。

        Args:
            inputs: 字典，必须包含 'raw_detectors': [batch, num_detectors]

        Returns:
            logits: [batch, 1]
            intermediates: dict
                'cnn_features': [B, num_t, num_stab, d_model] — TF 空间混合输出
                'decoder_states': [B, num_t, num_stab, d_model] — RNN 隐状态
                'readout_features': [B, d_model] — cross-attention 后的 query
                'readout_logits': [B, 1] — 最终 logits
        """
        self.model.eval()
        raw_det = inputs["raw_detectors"]  # [B, num_detectors]
        logits, intermediates = self.model.forward_with_intermediates(raw_det)

        # 分离所有中间表征，确保梯度不会流向 teacher
        intermediates = {k: v.detach() for k, v in intermediates.items()}
        return logits.detach(), intermediates

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


def load_teacher(teacher_config: dict, distance: int, device="cpu",
                 rounds: int = None) -> TeacherAdapter:
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
            n_heads: int (可选)         注意力头数（仅 alphaqubit）
        distance: 码距
        device: 加载设备
        rounds: 纠错轮数（alphaqubit 类型必需，用于重建 FullMapper）

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
        if rounds is None:
            raise ValueError(
                "alphaqubit teacher 需要 rounds 参数以重建 FullMapper。"
                "请确保在 load_teacher() 调用时传入 rounds。"
            )
        return AlphaQubitAdapter(
            checkpoint_path=teacher_config["checkpoint"],
            hidden_dim=teacher_config["hidden_dim"],
            readout_dim=teacher_config.get("readout_dim", teacher_config["hidden_dim"]),
            distance=distance,
            rounds=rounds,
            device=device,
            n_heads=teacher_config.get("n_heads", None),
        )

    else:
        raise ValueError(
            f"未知的 teacher 类型: '{teacher_type}'\n"
            f"支持的类型: 'mock', 'alphaqubit'"
        )
