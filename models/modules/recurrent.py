"""
Recurrent modules for temporal processing across error-correction rounds.

The RNN core processes syndrome data round-by-round, maintaining a
per-stabilizer decoder state that accumulates information over time.
This follows AlphaQubit's recurrent architecture but uses GRU/LSTM
instead of the full Transformer-based recurrence.
"""

import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    Recurrent decoder state update module.

    At each error-correction round:
    1. New stabilizer embeddings are combined with the previous decoder state
    2. Spatial mixing (CNN block) processes the combined representation
    3. The output becomes the new decoder state

    This mirrors AlphaQubit's recurrent structure (Fig. 2a, Extended Data Fig. 4d)
    but with CNN replacing the Syndrome Transformer.

    Supports both GRU and LSTM variants.
    """

    def __init__(
        self,
        hidden_dim: int,
        rnn_type: str = "gru",
        scale_factor: float = 0.707,
    ):
        """
        Args:
            hidden_dim: Dimension of per-stabilizer hidden state.
            rnn_type: 'gru' or 'lstm'.
            scale_factor: Scaling factor for state + embedding sum (1/sqrt(2) in AlphaQubit).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.scale_factor = scale_factor

        # Per-stabilizer RNN: processes each stabilizer's temporal sequence
        # Input: hidden_dim (combined state), Output: hidden_dim
        if rnn_type == "gru":
            self.rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)
        elif rnn_type == "lstm":
            self.rnn_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

    def init_state(self, batch_size: int, n_stabilizers: int, device: torch.device):
        """
        Initialize the decoder state to zeros (following AlphaQubit).

        Returns:
            state: [batch, n_stab, hidden_dim] zero-initialized decoder state
            cell: [batch, n_stab, hidden_dim] zero cell state (LSTM only, None for GRU)
        """
        h = torch.zeros(batch_size, n_stabilizers, self.hidden_dim, device=device)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)
            return h, c
        return h, None

    def step(self, embedding, state, cell=None):
        """
        One step of the recurrent update.

        Following AlphaQubit:
            combined = (state + embedding) * scale_factor
            new_state = RNN(combined, state)

        Args:
            embedding: [batch, n_stab, hidden_dim] new stabilizer embedding S_n
            state: [batch, n_stab, hidden_dim] previous decoder state
            cell: [batch, n_stab, hidden_dim] previous cell state (LSTM only)

        Returns:
            new_state: [batch, n_stab, hidden_dim]
            new_cell: [batch, n_stab, hidden_dim] or None
        """
        batch_size, n_stab, dim = embedding.shape

        # Combine: state + new embedding, then scale
        combined = (state + embedding) * self.scale_factor

        # Flatten stabilizer dimension into batch for RNN cell
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
