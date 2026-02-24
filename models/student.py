"""
知识蒸馏用 Student 解码器模型。

面向 surface code 的轻量级循环神经网络解码器，
设计目标是从 AlphaQubit（Teacher）蒸馏后部署到 FPGA。

架构：StabilizerEmbedder → CNN 空间混合 → GRU/LSTM 时序 → Readout

两种变体：
  B: CNN + LSTM
  D: CNN + GRU（默认，更轻量）
"""

import torch
import torch.nn as nn

from .modules.embedder import StabilizerEmbedder
from .modules.cnn_block import CNNBlock
from .modules.recurrent import DecoderRNN
from .modules.readout import ReadoutNetwork


class StudentDecoder(nn.Module):
    """
    Surface code 纠错用 Student 神经解码器。

    逐轮处理 syndrome 数据：
    1. 每轮：嵌入新的 stabilizer 测量值
    2. 与上一轮解码器状态结合（RNN 更新）
    3. CNN 空间混合
    4. 最后一轮结束后：readout 网络预测逻辑错误

    支持 `return_intermediates` 用于知识蒸馏，
    在预定义的 hook 点暴露内部表征。
    """

    def __init__(
        self,
        distance: int,
        hidden_dim: int = 32,
        conv_dim: int = 16,
        readout_dim: int = 16,
        n_cnn_layers: int = 2,
        n_cnn_blocks: int = 1,
        n_readout_layers: int = 2,
        rnn_type: str = "gru",
        use_soft: bool = True,
    ):
        """
        Args:
            distance: 码距 d，决定 n_stabilizers = d^2 - 1。
            hidden_dim: 每个 stabilizer 的隐藏维度。
            conv_dim: CNN 卷积内部维度。
            readout_dim: readout 网络维度。
            n_cnn_layers: 每个 CNN block 的卷积层数。
            n_cnn_blocks: CNN block 数量（AlphaQubit 使用 3 层 transformer）。
            n_readout_layers: readout 中残差层的数量。
            rnn_type: 'gru'（变体 D）或 'lstm'（变体 B）。
            use_soft: 是否提供 soft readout 输入。
        """
        super().__init__()
        self.distance = distance
        self.n_stabilizers = distance * distance - 1
        self.hidden_dim = hidden_dim
        self.readout_dim = readout_dim
        self.use_soft = use_soft

        # 1. Stabilizer 嵌入器
        self.embedder = StabilizerEmbedder(
            n_stabilizers=self.n_stabilizers,
            hidden_dim=hidden_dim,
            use_soft=use_soft,
        )

        # 2. CNN 空间混合模块
        self.cnn_blocks = nn.ModuleList()
        for _ in range(n_cnn_blocks):
            self.cnn_blocks.append(
                CNNBlock(
                    hidden_dim=hidden_dim,
                    conv_dim=conv_dim,
                    n_conv_layers=n_cnn_layers,
                    distance=distance,
                )
            )

        # 3. 循环时序模块
        self.rnn = DecoderRNN(
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
        )

        # 4. Readout 网络
        self.readout = ReadoutNetwork(
            hidden_dim=hidden_dim,
            readout_dim=readout_dim,
            n_layers=n_readout_layers,
            distance=distance,
        )

    def forward(self, inputs, return_intermediates=False):
        """
        处理完整的纠错轮次序列。

        Args:
            inputs: 字典，包含以下键：
                'detection_events': [batch, rounds, n_stab] 二值 float32
                'soft_events': [batch, rounds, n_stab] float32（可选）
            return_intermediates: 若为 True，返回用于知识蒸馏的中间表征。

        Returns:
            logits: [batch, 1] P(逻辑错误) 的 logit
            intermediates: 字典（仅当 return_intermediates=True 时）
                'cnn_features': [batch, rounds, n_stab, hidden_dim]
                    每轮 CNN 输出（Hook A：空间特征）
                'decoder_states': [batch, rounds, n_stab, hidden_dim]
                    每轮解码器状态（Hook B：时序特征）
                'readout_features': [batch, readout_dim]
                    Readout 中间特征
                'readout_logits': [batch, 1]
                    与 logits 相同（Hook C/D：决策特征）
        """
        detection_events = inputs["detection_events"]
        soft_events = inputs.get("soft_events", None)

        batch_size, n_rounds, n_stab = detection_events.shape
        device = detection_events.device

        # 初始化解码器状态
        state, cell = self.rnn.init_state(batch_size, n_stab, device)

        # 按需收集中间表征
        all_cnn_features = []
        all_decoder_states = []

        # 逐轮处理
        for r in range(n_rounds):
            # 提取当前轮的输入
            events_r = detection_events[:, r, :]  # [batch, n_stab]
            soft_r = soft_events[:, r, :] if soft_events is not None else None

            # 1. 嵌入当前轮的 stabilizer 输入
            embedding = self.embedder(events_r, soft_r)  # [batch, n_stab, hidden_dim]

            # 2. RNN 更新：与上一状态结合
            state, cell = self.rnn.step(embedding, state, cell)

            # 保存解码器状态（Hook B：RNN 之后、CNN 之前的时序特征）
            if return_intermediates:
                all_decoder_states.append(state)

            # 3. CNN 空间混合
            cnn_out = state
            for cnn_block in self.cnn_blocks:
                cnn_out = cnn_block(cnn_out)

            # 用 CNN 输出更新状态
            state = cnn_out

            # 保存 CNN 特征（Hook A：CNN 之后的空间特征）
            if return_intermediates:
                all_cnn_features.append(cnn_out)

        # 4. 从最终状态进行 readout
        logits, readout_features = self.readout(state, return_features=True)

        if return_intermediates:
            intermediates = {
                "cnn_features": torch.stack(all_cnn_features, dim=1),
                "decoder_states": torch.stack(all_decoder_states, dim=1),
                "readout_features": readout_features,
                "readout_logits": logits,
            }
            return logits, intermediates

        return logits

    def count_parameters(self):
        """统计可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_student(
    distance: int,
    size: str = "small",
    rnn_type: str = "gru",
    use_soft: bool = True,
) -> StudentDecoder:
    """
    工厂函数，创建不同大小的 Student 模型。

    Args:
        distance: 码距。
        size: 'tiny'、'small'、'medium' 或 'large'（mock teacher）。
        rnn_type: 'gru' 或 'lstm'。
        use_soft: 是否使用 soft readout 输入。

    Returns:
        StudentDecoder 模型。
    """
    configs = {
        "tiny": dict(hidden_dim=16, conv_dim=8, readout_dim=8,
                      n_cnn_layers=1, n_cnn_blocks=1, n_readout_layers=1),
        "small": dict(hidden_dim=32, conv_dim=16, readout_dim=16,
                       n_cnn_layers=2, n_cnn_blocks=1, n_readout_layers=2),
        "medium": dict(hidden_dim=64, conv_dim=32, readout_dim=32,
                        n_cnn_layers=2, n_cnn_blocks=2, n_readout_layers=3),
        "large": dict(hidden_dim=128, conv_dim=64, readout_dim=64,
                       n_cnn_layers=3, n_cnn_blocks=2, n_readout_layers=4),
    }
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(configs.keys())}")

    cfg = configs[size]
    return StudentDecoder(
        distance=distance,
        rnn_type=rnn_type,
        use_soft=use_soft,
        **cfg,
    )
