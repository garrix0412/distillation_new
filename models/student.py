"""
Student Decoder Model for knowledge distillation.

A lightweight recurrent neural network decoder for the surface code,
designed to be distilled from AlphaQubit (Teacher) and deployed on FPGA.

Architecture: StabilizerEmbedder → CNN spatial mixing → GRU/LSTM temporal → Readout

Two variants:
  B: CNN + LSTM
  D: CNN + GRU (default, lighter)
"""

import torch
import torch.nn as nn

from .modules.embedder import StabilizerEmbedder
from .modules.cnn_block import CNNBlock
from .modules.recurrent import DecoderRNN
from .modules.readout import ReadoutNetwork


class StudentDecoder(nn.Module):
    """
    Student neural decoder for surface code error correction.

    Processes syndrome data round-by-round:
    1. Each round: embed new stabilizer measurements
    2. Combine with previous decoder state (RNN update)
    3. Apply CNN spatial mixing
    4. After final round: readout network predicts logical error

    Supports `return_intermediates` for knowledge distillation,
    exposing internal representations at defined hook points.
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
            distance: Code distance d. Determines n_stabilizers = d^2 - 1.
            hidden_dim: Per-stabilizer hidden dimension.
            conv_dim: Internal dimension for CNN convolutions.
            readout_dim: Dimension for readout network.
            n_cnn_layers: Number of conv layers per CNN block.
            n_cnn_blocks: Number of CNN blocks (AlphaQubit uses 3 transformer layers).
            n_readout_layers: Number of residual layers in readout.
            rnn_type: 'gru' (variant D) or 'lstm' (variant B).
            use_soft: Whether soft readout inputs are provided.
        """
        super().__init__()
        self.distance = distance
        self.n_stabilizers = distance * distance - 1
        self.hidden_dim = hidden_dim
        self.use_soft = use_soft

        # 1. Stabilizer Embedder
        self.embedder = StabilizerEmbedder(
            n_stabilizers=self.n_stabilizers,
            hidden_dim=hidden_dim,
            use_soft=use_soft,
        )

        # 2. CNN spatial mixing blocks
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

        # 3. Recurrent temporal module
        self.rnn = DecoderRNN(
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
        )

        # 4. Readout network
        self.readout = ReadoutNetwork(
            hidden_dim=hidden_dim,
            readout_dim=readout_dim,
            n_layers=n_readout_layers,
            distance=distance,
        )

    def forward(self, inputs, return_intermediates=False):
        """
        Process a full sequence of error-correction rounds.

        Args:
            inputs: dict with keys:
                'detection_events': [batch, rounds, n_stab] binary float32
                'soft_events': [batch, rounds, n_stab] float32 (optional)
            return_intermediates: If True, return intermediate representations
                for knowledge distillation.

        Returns:
            logits: [batch, 1] logit for P(logical error)
            intermediates: dict (only if return_intermediates=True)
                'cnn_features': [batch, rounds, n_stab, hidden_dim]
                    CNN output per round (Hook A: spatial features)
                'decoder_states': [batch, rounds, n_stab, hidden_dim]
                    Decoder state per round (Hook B: temporal features)
                'readout_features': [batch, readout_dim]
                    Readout intermediate features
                'readout_logits': [batch, 1]
                    Same as logits (Hook C/D: decision features)
        """
        detection_events = inputs["detection_events"]
        soft_events = inputs.get("soft_events", None)

        batch_size, n_rounds, n_stab = detection_events.shape
        device = detection_events.device

        # Initialize decoder state
        state, cell = self.rnn.init_state(batch_size, n_stab, device)

        # Collect intermediates if requested
        all_cnn_features = []
        all_decoder_states = []

        # Process round by round
        for r in range(n_rounds):
            # Extract this round's inputs
            events_r = detection_events[:, r, :]  # [batch, n_stab]
            soft_r = soft_events[:, r, :] if soft_events is not None else None

            # 1. Embed stabilizer inputs for this round
            embedding = self.embedder(events_r, soft_r)  # [batch, n_stab, hidden_dim]

            # 2. RNN update: combine with previous state
            state, cell = self.rnn.step(embedding, state, cell)

            # 3. CNN spatial mixing
            cnn_out = state
            for cnn_block in self.cnn_blocks:
                cnn_out = cnn_block(cnn_out)

            # Update state with CNN output
            state = cnn_out

            if return_intermediates:
                all_cnn_features.append(cnn_out)
                all_decoder_states.append(state)

        # 4. Readout from final state
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
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_student(
    distance: int,
    size: str = "small",
    rnn_type: str = "gru",
    use_soft: bool = True,
) -> StudentDecoder:
    """
    Factory function to create Student models of different sizes.

    Args:
        distance: Code distance.
        size: 'tiny', 'small', 'medium', or 'large' (mock teacher).
        rnn_type: 'gru' or 'lstm'.
        use_soft: Whether to use soft readout inputs.

    Returns:
        StudentDecoder model.
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
