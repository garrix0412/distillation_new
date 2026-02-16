"""
Readout Network module.

Takes the final decoder state and produces a prediction of whether
a logical error occurred. Simplified version of AlphaQubit's readout.
"""

import torch
import torch.nn as nn


class ReadoutNetwork(nn.Module):
    """
    Readout network that maps decoder state to logical error prediction.

    Simplified version of AlphaQubit's readout:
    1. Linear projection (dimensionality reduction)
    2. Mean pooling over stabilizers
    3. Small residual network
    4. Final logit output

    AlphaQubit uses scatter → 2x2 conv → pool along rows/columns → ResNet.
    Our simplified version pools all stabilizers directly, which is more
    FPGA-friendly and sufficient for small code distances.
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
            hidden_dim: Input dimension per stabilizer.
            readout_dim: Internal dimension for readout processing.
            n_layers: Number of residual layers.
            distance: Code distance (for per-line prediction).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_dim = readout_dim
        self.distance = distance

        # Project from hidden_dim to readout_dim
        self.proj = nn.Linear(hidden_dim, readout_dim)

        # Residual blocks
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

        # Final classification head
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, 1),
        )

    def forward(self, decoder_state, return_features=False):
        """
        Args:
            decoder_state: [batch, n_stab, hidden_dim] final decoder state
            return_features: If True, also return intermediate features

        Returns:
            logits: [batch, 1] logit for P(logical error)
            features: [batch, readout_dim] (only if return_features=True)
        """
        # Project to readout dimension
        h = self.proj(decoder_state)  # [batch, n_stab, readout_dim]

        # Mean pool over stabilizers
        h = h.mean(dim=1)  # [batch, readout_dim]

        # Residual blocks
        for block in self.res_blocks:
            h = h + block(h)

        features = h

        # Final logit
        logits = self.head(h)  # [batch, 1]

        if return_features:
            return logits, features
        return logits
