"""
Auxiliary probe heads for fused logits distillation.

Lightweight readout heads that map intermediate Teacher representations
(CNN features, decoder states) to logit space. These are trained on
frozen Teacher features and used to construct fused soft targets.

Each probe head: mean_pool over stabilizers → Linear → logit.
"""

import torch
import torch.nn as nn


class AuxiliaryProbeHead(nn.Module):
    """
    Lightweight probe that maps per-stabilizer features to a single logit.

    Takes the last round's intermediate representation and produces
    a logit predicting P(logical error), similar to the main readout
    but much simpler (single linear layer after pooling).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        """
        Args:
            features: [batch, n_stab, hidden_dim] last-round features

        Returns:
            logits: [batch, 1]
        """
        pooled = features.mean(dim=1)  # [batch, hidden_dim]
        return self.head(pooled)  # [batch, 1]


class ProbeHeadSet(nn.Module):
    """
    Collection of probe heads for CNN and RNN intermediate features.

    Produces three logit streams from teacher intermediates:
    - z_cnn: from CNN features (last round)
    - z_rnn: from decoder states (last round)
    - z_final: from teacher's readout (passed through, not computed here)

    Fused logit: z_fuse = (z_cnn + z_rnn + z_final) / 3
    """

    def __init__(self, teacher_dim: int):
        super().__init__()
        self.cnn_probe = AuxiliaryProbeHead(teacher_dim)
        self.rnn_probe = AuxiliaryProbeHead(teacher_dim)

    def forward(self, teacher_intermediates):
        """
        Args:
            teacher_intermediates: dict from TeacherWrapper.forward_with_intermediates()
                Must contain 'cnn_features', 'decoder_states', 'readout_logits'.

        Returns:
            dict with 'cnn_logits', 'rnn_logits', 'fused_logits'
        """
        # Last round features: [batch, n_stab, hidden_dim]
        cnn_last = teacher_intermediates["cnn_features"][:, -1, :, :]
        rnn_last = teacher_intermediates["decoder_states"][:, -1, :, :]
        z_final = teacher_intermediates["readout_logits"]

        z_cnn = self.cnn_probe(cnn_last)
        z_rnn = self.rnn_probe(rnn_last)
        z_fused = (z_cnn + z_rnn + z_final) / 3.0

        return {
            "cnn_logits": z_cnn,
            "rnn_logits": z_rnn,
            "fused_logits": z_fused,
        }
