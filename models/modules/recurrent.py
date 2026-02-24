"""
纠错轮次间时序处理的循环模块。

RNN 核心逐轮处理 syndrome 数据，维护每个 stabilizer 的
解码器状态，随时间累积信息。
参考 AlphaQubit 的循环架构，但使用 GRU/LSTM
替代完整的基于 Transformer 的循环结构。
"""

import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    循环解码器状态更新模块。

    在每个纠错轮次：
    1. 新的 stabilizer 嵌入与先前解码器状态结合
    2. 空间混合（CNN block）处理组合后的表征
    3. 输出成为新的解码器状态

    对应 AlphaQubit 的循环结构（Fig. 2a, Extended Data Fig. 4d），
    但用 CNN 替代了 Syndrome Transformer。

    支持 GRU 和 LSTM 两种变体。
    """

    def __init__(
        self,
        hidden_dim: int,
        rnn_type: str = "gru",
        scale_factor: float = 0.707,
    ):
        """
        Args:
            hidden_dim: 每个 stabilizer 隐藏状态的维度。
            rnn_type: 'gru' 或 'lstm'。
            scale_factor: 状态 + 嵌入求和的缩放因子（AlphaQubit 中为 1/sqrt(2)）。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.scale_factor = scale_factor

        # Per-stabilizer RNN：处理每个 stabilizer 的时序序列
        # 输入：hidden_dim（组合状态），输出：hidden_dim
        if rnn_type == "gru":
            self.rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)
        elif rnn_type == "lstm":
            self.rnn_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

    def init_state(self, batch_size: int, n_stabilizers: int, device: torch.device):
        """
        将解码器状态初始化为零（沿用 AlphaQubit 做法）。

        Returns:
            state: [batch, n_stab, hidden_dim] 零初始化的解码器状态
            cell: [batch, n_stab, hidden_dim] 零 cell 状态（仅 LSTM，GRU 为 None）
        """
        h = torch.zeros(batch_size, n_stabilizers, self.hidden_dim, device=device)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
            return h, c
        return h, None

    def step(self, embedding, state, cell=None):
        """
        循环更新的一步。

        沿用 AlphaQubit 做法：
            combined = (state + embedding) * scale_factor
            new_state = RNN(combined, state)

        Args:
            embedding: [batch, n_stab, hidden_dim] 新的 stabilizer 嵌入 S_n
            state: [batch, n_stab, hidden_dim] 上一步解码器状态
            cell: [batch, n_stab, hidden_dim] 上一步 cell 状态（仅 LSTM）

        Returns:
            new_state: [batch, n_stab, hidden_dim]
            new_cell: [batch, n_stab, hidden_dim] 或 None
        """
        batch_size, n_stab, dim = embedding.shape

        # 组合：状态 + 新嵌入，然后缩放
        combined = (state + embedding) * self.scale_factor

        # 将 stabilizer 维度展平到 batch 维度，供 RNN cell 使用
        # [batch * n_stab, hidden_dim]
        combined_flat = combined.reshape(-1, dim)
        state_flat = state.reshape(-1, dim)

        if self.rnn_type == "gru":
            new_state_flat = self.rnn_cell(combined_flat, state_flat)
            new_state = new_state_flat.reshape(batch_size, n_stab, dim)
            return new_state, None
        else:
            cell_flat = cell.reshape(-1, dim)
            new_state_flat, new_cell_flat = self.rnn_cell(
                combined_flat, (state_flat, cell_flat)
            )
            new_state = new_state_flat.reshape(batch_size, n_stab, dim)
            new_cell = new_cell_flat.reshape(batch_size, n_stab, dim)
            return new_state, new_cell
