"""
Teacher model interface for knowledge distillation.

This module provides a unified interface for loading and using a Teacher
model (either Mock Teacher or the real AlphaQubit reproduction) in the
distillation pipeline.

The Teacher must expose intermediate representations at defined hook points:
  - Hook A: CNN/Transformer features (spatial, per round)
  - Hook B: Decoder states (temporal, per round)
  - Hook C: Readout logits (decision)
  - Hook D: Final logits (output)
"""

import torch
import torch.nn as nn

from .student import StudentDecoder, create_student
from distillation.probe_heads import ProbeHeadSet


class TeacherWrapper(nn.Module):
    """
    Wrapper that loads a trained model and provides the Teacher interface.

    For the Mock Teacher, this wraps a larger StudentDecoder.
    For the real AlphaQubit Teacher, this should be replaced with
    a wrapper around the actual AlphaQubit model (to be done when
    the Teacher model is ready).

    The key contract is `forward_with_intermediates()` which returns
    both predictions and internal representations for KD.
    """

    def __init__(self, model: nn.Module, probe_heads: ProbeHeadSet = None):
        super().__init__()
        self.model = model
        self.probe_heads = probe_heads
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Freeze probe heads too (they are pre-trained)
        if self.probe_heads is not None:
            for param in self.probe_heads.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, inputs):
        """Standard forward pass (inference only)."""
        self.model.eval()
        return self.model(inputs)

    @torch.no_grad()
    def forward_with_intermediates(self, inputs):
        """
        Forward pass returning intermediate representations for KD.

        Args:
            inputs: dict with 'detection_events' and optional 'soft_events'

        Returns:
            logits: [batch, 1]
            intermediates: dict
                'cnn_features': [batch, rounds, n_stab, hidden_dim]
                    Per-round CNN output (Hook A: spatial features)
                'decoder_states': [batch, rounds, n_stab, hidden_dim]
                    Per-round decoder state (Hook B: temporal features)
                'readout_features': [batch, readout_dim]
                    Readout internal features
                'readout_logits': [batch, 1]
                    Output logits (Hook C/D)
        """
        self.model.eval()
        logits, intermediates = self.model(inputs, return_intermediates=True)
        # Detach all intermediates to ensure no gradient flow to teacher
        intermediates = {
            k: v.detach() for k, v in intermediates.items()
        }
        # Compute fused logits if probe heads are available
        if self.probe_heads is not None:
            self.probe_heads.eval()
            fused_outputs = self.probe_heads(intermediates)
            for k, v in fused_outputs.items():
                intermediates[k] = v.detach()
        return logits.detach(), intermediates

    @property
    def hidden_dim(self):
        return self.model.hidden_dim


def load_mock_teacher(
    checkpoint_path: str,
    distance: int,
    device: str = "cpu",
) -> TeacherWrapper:
    """
    Load a trained Mock Teacher from checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint.
        distance: Code distance.
        device: Device to load the model on.

    Returns:
        TeacherWrapper with frozen parameters.
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
    Load a trained Mock Teacher with pre-trained probe heads for fused logits.

    Args:
        checkpoint_path: Path to the saved teacher model checkpoint.
        probe_heads_path: Path to the saved probe heads checkpoint.
        distance: Code distance.
        device: Device to load the model on.

    Returns:
        TeacherWrapper with frozen parameters and probe heads.
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
