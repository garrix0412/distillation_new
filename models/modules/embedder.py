"""
Stabilizer Embedder module.

Maps per-stabilizer input features (detection events, soft readout)
to a hidden representation vector for each stabilizer.
Follows the AlphaQubit StabilizerEmbedder design but simplified.
"""

import torch
import torch.nn as nn


class StabilizerEmbedder(nn.Module):
    """
    Embed per-stabilizer inputs into hidden representations.

    For each stabilizer at each round, takes input features
    (detection event, soft readout probability) and produces
    a hidden vector of dimension `hidden_dim`.

    Architecture:
    - Separate linear projections for each input feature
    - Sum the projections
    - Add a learned stabilizer index embedding
    - Pass through a residual block
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
            n_stabilizers: Number of stabilizers (d^2 - 1).
            hidden_dim: Dimension of the output embedding per stabilizer.
            n_input_features: Number of input features per stabilizer.
                2 = (detection_event, soft_event) when use_soft=True
                1 = (detection_event) when use_soft=False
            use_soft: Whether soft readout is provided as input.
        """
        super().__init__()
        self.n_stabilizers = n_stabilizers
        self.hidden_dim = hidden_dim
        self.use_soft = use_soft
        self.n_input_features = 2 if use_soft else 1

        # Separate linear projection for each input feature
        # Each maps a scalar to hidden_dim
        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(self.n_input_features)]
        )

        # Learned stabilizer index embedding
        self.stabilizer_embedding = nn.Embedding(n_stabilizers, hidden_dim)

        # Residual block for mixing
        self.res_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, detection_events, soft_events=None):
        """
        Args:
            detection_events: [batch, n_stabilizers] binary detection events for one round
            soft_events: [batch, n_stabilizers] soft probabilities for one round (optional)

        Returns:
            embeddings: [batch, n_stabilizers, hidden_dim]
        """
        batch_size = detection_events.shape[0]

        # Project each feature: [batch, n_stab] -> [batch, n_stab, 1] -> [batch, n_stab, hidden_dim]
        features = [self.feature_projections[0](detection_events.unsqueeze(-1))]

        if self.use_soft and soft_events is not None:
            features.append(self.feature_projections[1](soft_events.unsqueeze(-1)))

        # Sum projections: [batch, n_stab, hidden_dim]
        h = torch.stack(features, dim=0).sum(dim=0)

        # Add stabilizer index embedding
        stab_indices = torch.arange(self.n_stabilizers, device=h.device)
        stab_embed = self.stabilizer_embedding(stab_indices)  # [n_stab, hidden_dim]
        h = h + stab_embed.unsqueeze(0)  # broadcast over batch

        # Residual block
        h = h + self.res_block(h)

        return h
