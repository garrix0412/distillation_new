"""
Knowledge Distillation loss functions.

Implements various KD losses for the multi-signal distillation pipeline:
- Response-based KD: soft logit matching (KL divergence / MSE)
- Feature-based KD: intermediate representation alignment (MSE / cosine)
- Combined losses with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseKDLoss(nn.Module):
    """
    Response-based Knowledge Distillation loss.

    Matches Student's output distribution to Teacher's soft output.
    For binary classification (logical error prediction), we use
    KL divergence between the two Bernoulli distributions parameterized
    by the logits.

    L_response = T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Distillation temperature. Higher T produces
                softer probability distributions.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: [batch, 1] student output logits
            teacher_logits: [batch, 1] teacher output logits

        Returns:
            Scalar KD loss.
        """
        s = student_logits.squeeze(-1) / self.temperature
        t = teacher_logits.squeeze(-1) / self.temperature

        # For binary classification: KL between two Bernoulli distributions
        # KL(Ber(p_t) || Ber(p_s)) using logits
        # = p_t * log(p_t/p_s) + (1-p_t) * log((1-p_t)/(1-p_s))
        # Equivalent to: BCE(sigmoid(t), sigmoid(s)) when using the right formulation
        # We use the standard KD formulation with sigmoid + KL
        p_t = torch.sigmoid(t).detach()
        loss = F.binary_cross_entropy_with_logits(s, p_t, reduction="mean")

        return loss * (self.temperature ** 2)


class FeatureKDLoss(nn.Module):
    """
    Feature-based Knowledge Distillation loss.

    Aligns intermediate representations between Student and Teacher.
    Supports optional projection head when dimensions differ.

    L_feature = MSE(proj(f_student), f_teacher)  or
    L_feature = 1 - cosine_sim(proj(f_student), f_teacher)
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        loss_type: str = "mse",
    ):
        """
        Args:
            student_dim: Student feature dimension.
            teacher_dim: Teacher feature dimension.
            loss_type: 'mse' or 'cosine'.
        """
        super().__init__()
        self.loss_type = loss_type

        # Projection head for dimension alignment (only during training)
        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [..., student_dim] student intermediate features
            teacher_features: [..., teacher_dim] teacher intermediate features

        Returns:
            Scalar feature alignment loss.
        """
        # Project student features to teacher dimension
        projected = self.projector(student_features)

        if self.loss_type == "mse":
            return F.mse_loss(projected, teacher_features)
        elif self.loss_type == "cosine":
            # Flatten to 2D for cosine similarity
            p_flat = projected.reshape(-1, projected.shape[-1])
            t_flat = teacher_features.reshape(-1, teacher_features.shape[-1])
            cos_sim = F.cosine_similarity(p_flat, t_flat, dim=-1)
            return (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for the full KD pipeline.

    Total loss = alpha * L_task + beta * L_response
              + gamma_cnn * L_feature_cnn + gamma_rnn * L_feature_rnn
              + gamma_fused * L_fused

    Where:
    - L_task: Cross-entropy with ground truth labels
    - L_response: Soft logit matching with teacher
    - L_feature_cnn: CNN feature alignment (spatial, Hook A)
    - L_feature_rnn: Decoder state alignment (temporal, Hook B)
    - L_fused: Fused logit KD (matching student to fused teacher logits)
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma_cnn: float = 0.0,
        gamma_rnn: float = 0.0,
        gamma_fused: float = 0.0,
        temperature: float = 1.0,
        feature_loss_type: str = "mse",
    ):
        """
        Args:
            student_dim: Student hidden dimension.
            teacher_dim: Teacher hidden dimension.
            alpha: Weight for task loss (ground truth CE).
            beta: Weight for response KD loss (soft logit matching).
            gamma_cnn: Weight for CNN feature KD loss (spatial).
            gamma_rnn: Weight for RNN/decoder state KD loss (temporal).
            gamma_fused: Weight for fused logit KD loss.
            temperature: Distillation temperature.
            feature_loss_type: 'mse' or 'cosine' for feature alignment.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_cnn = gamma_cnn
        self.gamma_rnn = gamma_rnn
        self.gamma_fused = gamma_fused

        # Task loss (ground truth supervision)
        self.task_loss = nn.BCEWithLogitsLoss()

        # Response KD loss
        self.response_kd = ResponseKDLoss(temperature=temperature)

        # Fused logit KD loss (reuses ResponseKDLoss with same temperature)
        if gamma_fused > 0:
            self.fused_kd = ResponseKDLoss(temperature=temperature)

        # Feature KD losses (only if weights > 0)
        if gamma_cnn > 0:
            self.cnn_feature_kd = FeatureKDLoss(
                student_dim, teacher_dim, feature_loss_type
            )
        if gamma_rnn > 0:
            self.rnn_feature_kd = FeatureKDLoss(
                student_dim, teacher_dim, feature_loss_type
            )

    def forward(
        self,
        student_logits,
        labels,
        teacher_logits=None,
        student_intermediates=None,
        teacher_intermediates=None,
    ):
        """
        Compute combined distillation loss.

        Args:
            student_logits: [batch, 1] student predictions
            labels: [batch] ground truth labels
            teacher_logits: [batch, 1] teacher predictions (optional)
            student_intermediates: dict of student intermediate representations
            teacher_intermediates: dict of teacher intermediate representations

        Returns:
            total_loss: Scalar combined loss
            loss_dict: dict of individual loss components for logging
        """
        loss_dict = {}

        # Task loss (always present)
        l_task = self.task_loss(student_logits.squeeze(-1), labels)
        loss_dict["task"] = l_task.item()
        total = self.alpha * l_task

        # Response KD loss
        if teacher_logits is not None and self.beta > 0:
            l_response = self.response_kd(student_logits, teacher_logits)
            loss_dict["response_kd"] = l_response.item()
            total = total + self.beta * l_response

        # CNN Feature KD loss (spatial, Hook A)
        if (
            self.gamma_cnn > 0
            and student_intermediates is not None
            and teacher_intermediates is not None
        ):
            l_cnn = self.cnn_feature_kd(
                student_intermediates["cnn_features"],
                teacher_intermediates["cnn_features"],
            )
            loss_dict["cnn_feature_kd"] = l_cnn.item()
            total = total + self.gamma_cnn * l_cnn

        # RNN Feature KD loss (temporal, Hook B)
        if (
            self.gamma_rnn > 0
            and student_intermediates is not None
            and teacher_intermediates is not None
        ):
            l_rnn = self.rnn_feature_kd(
                student_intermediates["decoder_states"],
                teacher_intermediates["decoder_states"],
            )
            loss_dict["rnn_feature_kd"] = l_rnn.item()
            total = total + self.gamma_rnn * l_rnn

        # Fused logit KD loss
        if (
            self.gamma_fused > 0
            and teacher_intermediates is not None
            and "fused_logits" in teacher_intermediates
        ):
            l_fused = self.fused_kd(
                student_logits, teacher_intermediates["fused_logits"]
            )
            loss_dict["fused_kd"] = l_fused.item()
            total = total + self.gamma_fused * l_fused

        loss_dict["total"] = total.item()
        return total, loss_dict
