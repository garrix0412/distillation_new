"""
知识蒸馏损失函数。

实现多信号蒸馏管线的各类 KD 损失：
- 基于响应的 KD：soft logit 匹配（KL 散度 / MSE）
- 基于特征的 KD：中间表征对齐（MSE / cosine）
- 可配置权重的组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseKDLoss(nn.Module):
    """
    基于响应的知识蒸馏损失。

    将 Student 的输出分布与 Teacher 的 soft 输出匹配。
    对于二分类（逻辑错误预测），使用
    两个由 logits 参数化的 Bernoulli 分布之间的 KL 散度。

    L_response = T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: 蒸馏温度。较高的 T 产生更平滑的概率分布。
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: [batch, 1] student 输出 logits
            teacher_logits: [batch, 1] teacher 输出 logits

        Returns:
            标量 KD 损失。
        """
        s = student_logits.squeeze(-1) / self.temperature
        t = teacher_logits.squeeze(-1) / self.temperature

        # 二分类：两个 Bernoulli 分布之间的 KL 散度
        # KL(Ber(p_t) || Ber(p_s))，使用 logits
        # = p_t * log(p_t/p_s) + (1-p_t) * log((1-p_t)/(1-p_s))
        # 等价于使用正确公式的 BCE(sigmoid(t), sigmoid(s))
        # 我们使用标准 KD 公式：sigmoid + KL
        p_t = torch.sigmoid(t).detach()
        loss = F.binary_cross_entropy_with_logits(s, p_t, reduction="mean")

        return loss * (self.temperature ** 2)


class FeatureKDLoss(nn.Module):
    """
    基于特征的知识蒸馏损失。

    对齐 Student 和 Teacher 的中间表征。
    维度不同时支持可选的投影头。

    L_feature = MSE(proj(f_student), f_teacher)  或
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
            student_dim: Student 特征维度。
            teacher_dim: Teacher 特征维度。
            loss_type: 'mse' 或 'cosine'。
        """
        super().__init__()
        self.loss_type = loss_type

        # 维度对齐用投影头（仅训练时使用）
        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [..., student_dim] student 中间特征
            teacher_features: [..., teacher_dim] teacher 中间特征

        Returns:
            标量特征对齐损失。
        """
        # 将 student 特征投影到 teacher 维度
        projected = self.projector(student_features)

        if self.loss_type == "mse":
            return F.mse_loss(projected, teacher_features)
        elif self.loss_type == "cosine":
            # 展平为 2D 以计算余弦相似度
            p_flat = projected.reshape(-1, projected.shape[-1])
            t_flat = teacher_features.reshape(-1, teacher_features.shape[-1])
            cos_sim = F.cosine_similarity(p_flat, t_flat, dim=-1)
            return (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class DistillationLoss(nn.Module):
    """
    完整 KD 管线的组合蒸馏损失。

    总损失 = alpha * L_task + beta * L_response
              + gamma_cnn * L_feature_cnn + gamma_rnn * L_feature_rnn
              + gamma_readout * L_readout_feature + gamma_fused * L_fused

    其中：
    - L_task: 与真实标签的交叉熵
    - L_response: 与 teacher 的 soft logit 匹配
    - L_feature_cnn: CNN 特征对齐（空间，Hook A）
    - L_feature_rnn: 解码器状态对齐（时序，Hook B）
    - L_readout_feature: Readout 特征对齐
    - L_fused: Fused logit KD（student 匹配 teacher 的融合 logits）
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma_cnn: float = 0.0,
        gamma_rnn: float = 0.0,
        gamma_readout: float = 0.0,
        gamma_fused: float = 0.0,
        temperature: float = 1.0,
        feature_loss_type: str = "mse",
        student_readout_dim: int = None,
        teacher_readout_dim: int = None,
    ):
        """
        Args:
            student_dim: Student 隐藏维度。
            teacher_dim: Teacher 隐藏维度。
            alpha: 任务损失权重（真实标签 CE）。
            beta: 响应 KD 损失权重（soft logit 匹配）。
            gamma_cnn: CNN 特征 KD 损失权重（空间）。
            gamma_rnn: RNN/解码器状态 KD 损失权重（时序）。
            gamma_readout: Readout 特征 KD 损失权重。
            gamma_fused: Fused logit KD 损失权重。
            temperature: 蒸馏温度。
            feature_loss_type: 特征对齐用 'mse' 或 'cosine'。
            student_readout_dim: Student readout 维度（默认为 student_dim）。
            teacher_readout_dim: Teacher readout 维度（默认为 teacher_dim）。
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_cnn = gamma_cnn
        self.gamma_rnn = gamma_rnn
        self.gamma_readout = gamma_readout
        self.gamma_fused = gamma_fused

        # 任务损失（真实标签监督）
        self.task_loss = nn.BCEWithLogitsLoss()

        # 响应 KD 损失
        self.response_kd = ResponseKDLoss(temperature=temperature)

        # Fused logit KD 损失（复用 ResponseKDLoss，相同温度）
        if gamma_fused > 0:
            self.fused_kd = ResponseKDLoss(temperature=temperature)

        # 特征 KD 损失（仅在权重 > 0 时创建）
        if gamma_cnn > 0:
            self.cnn_feature_kd = FeatureKDLoss(
                student_dim, teacher_dim, feature_loss_type
            )
        if gamma_rnn > 0:
            self.rnn_feature_kd = FeatureKDLoss(
                student_dim, teacher_dim, feature_loss_type
            )
        if gamma_readout > 0:
            s_rdim = student_readout_dim if student_readout_dim is not None else student_dim
            t_rdim = teacher_readout_dim if teacher_readout_dim is not None else teacher_dim
            self.readout_feature_kd = FeatureKDLoss(
                s_rdim, t_rdim, feature_loss_type
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
        计算组合蒸馏损失。

        Args:
            student_logits: [batch, 1] student 预测
            labels: [batch] 真实标签
            teacher_logits: [batch, 1] teacher 预测（可选）
            student_intermediates: student 中间表征字典
            teacher_intermediates: teacher 中间表征字典

        Returns:
            total_loss: 标量组合损失
            loss_dict: 各损失分量字典，用于日志记录
        """
        loss_dict = {}

        # 任务损失（始终存在）
        l_task = self.task_loss(student_logits.squeeze(-1), labels)
        loss_dict["task"] = l_task.item()
        total = self.alpha * l_task

        # 响应 KD 损失
        if teacher_logits is not None and self.beta > 0:
            l_response = self.response_kd(student_logits, teacher_logits)
            loss_dict["response_kd"] = l_response.item()
            total = total + self.beta * l_response

        # CNN 特征 KD 损失（空间，Hook A）
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

        # RNN 特征 KD 损失（时序，Hook B）
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

        # Readout 特征 KD 损失
        if (
            self.gamma_readout > 0
            and student_intermediates is not None
            and teacher_intermediates is not None
        ):
            l_readout = self.readout_feature_kd(
                student_intermediates["readout_features"],
                teacher_intermediates["readout_features"],
            )
            loss_dict["readout_feature_kd"] = l_readout.item()
            total = total + self.gamma_readout * l_readout

        # Fused logit KD 损失
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
