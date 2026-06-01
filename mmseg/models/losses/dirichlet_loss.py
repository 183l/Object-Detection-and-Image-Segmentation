import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.logging import MessageHub


def _standard_dirichlet_kl(alpha, y, num_classes):
    """计算标准狄利克雷 KL 散度：KL(Dir(alpha_tilde) || Dir(1))。

    标准 EDL 正则化项，用于约束非目标类别的证据量，增强模型对
    分布外样本（OOD）的不确定性区分能力。

    Args:
        alpha (Tensor): 狄利克雷浓度参数，shape (N, C)。
        y (Tensor): 真实标签的单热编码，shape (N, C)，dtype float32。
        num_classes (int): 类别数量 C。

    Returns:
        Tensor: 每个像素的 KL 散度值，shape (N,)。

    数学公式：
        alpha_tilde_k = y_k + (1 - y_k) * alpha_k
        KL = lgamma(sum(alpha_tilde)) - lgamma(K)
             - sum(lgamma(alpha_tilde)) + sum(lgamma(1))
             + sum((alpha_tilde - 1) * (digamma(alpha_tilde) - digamma(sum(alpha_tilde))))
    """
    # 构造目标分布的浓度参数（目标类保持为 1，非目标类保留原始 alpha）
    alpha_tilde = y + (1.0 - y) * alpha                          # shape (N, C)
    sum_alpha_tilde = alpha_tilde.sum(dim=1, keepdim=True)       # shape (N, 1)

    # 先验均匀分布参数（所有类别 alpha = 1）
    beta = torch.ones_like(alpha_tilde)                          # shape (N, C)
    sum_beta = torch.full_like(sum_alpha_tilde, num_classes)     # shape (N, 1)

    # 标准 KL 散度精确计算
    kl = (torch.lgamma(sum_alpha_tilde) - torch.lgamma(sum_beta)
          - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
          + torch.lgamma(beta).sum(dim=1, keepdim=True)
          + ((alpha_tilde - beta)
             * (torch.digamma(alpha_tilde)
                - torch.digamma(sum_alpha_tilde))).sum(dim=1, keepdim=True))

    return kl.squeeze(1)  # shape (N,)


@MODELS.register_module()
class DirichletLoss(nn.Module):
    """狄利克雷损失函数（Dirichlet Evidence Loss）。

    基于证据深度学习（EDL）框架，将分类损失建模为狄利克雷分布的
    负对数似然，并附加 KL 散度正则化项以约束非目标类别的证据量。

    损失组成：
        L = L_fit + anneal_coef * L_kl

    其中：
        L_fit = sum_k y_k * (digamma(S) - digamma(alpha_k))
        L_kl  = KL(Dir(alpha_tilde) || Dir(1))

    Args:
        loss_weight (float): 损失权重，默认 1.0。
        class_weight (list[float], optional): 各类别的像素级权重，
            用于缓解类别不平衡问题，默认 None（等权重）。
        ignore_index (int): 忽略的标签值，默认 255（Cityscapes 无效区域）。
        loss_name (str): 损失名称，用于日志记录，默认 'loss_dirichlet'。
        anneal_iters (int): KL 权重退火的迭代步数。在 [0, anneal_iters]
            内 KL 权重从 0 线性增长到 kl_weight，之后保持不变，默认 3000。
        kl_weight (float): KL 散度正则化项的最终权重，默认 0.001。
        use_standard_kl (bool): 是否使用标准狄利克雷 KL 散度。
            True 使用精确的 lgamma/digamma 实现（OOD 区分度更强）；
            False 使用简化版（速度更快，向后兼容），默认 False。
    """

    def __init__(self,
                 loss_weight=1.0,
                 class_weight=None,
                 ignore_index=255,
                 loss_name='loss_dirichlet',
                 anneal_iters=3000,
                 kl_weight=0.001,
                 use_standard_kl=False,
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.anneal_iters = anneal_iters
        self.kl_weight = kl_weight
        self.use_standard_kl = use_standard_kl

    def forward(self, cls_score, label, weight=None, **kwargs):
        """计算狄利克雷损失。

        Args:
            cls_score (Tensor): 模型输出的原始 logits，shape (N, C, H, W)。
            label (Tensor): 真实标签，shape (N, H, W)，dtype int64。
            weight (Tensor, optional): 像素级权重，默认 None。

        Returns:
            Tensor: 标量损失值。
        """
        # 从 MMEngine 官方 Runner 获取当前全局迭代步数，用于 KL 退火调度
        message_hub = MessageHub.get_current_instance()
        if message_hub is not None:
            current_iter = message_hub.get_info('iter', default=0)
        else:
            current_iter = 0

        # 有效像素掩码过滤（剥离 ignore_index 无效区域）
        valid_mask = (label != self.ignore_index)
        valid_logits = cls_score.permute(0, 2, 3, 1)[valid_mask]  # (N_valid, C)
        valid_labels = label[valid_mask]                           # (N_valid,)

        # 兜底保护：整个 batch 全为无效像素时返回零梯度
        if valid_logits.numel() == 0:
            return cls_score.sum() * 0.0

        num_classes = valid_logits.shape[1]
        y = F.one_hot(valid_labels, num_classes=num_classes).float()  # (N_valid, C)

        # 动态类别权重映射
        if self.class_weight is not None:
            class_weight_tensor = torch.tensor(
                self.class_weight, device=valid_logits.device, dtype=torch.float32)
            pixel_weights = class_weight_tensor[valid_labels]
        else:
            pixel_weights = torch.ones_like(valid_labels, dtype=torch.float32)

        # 证据空间转换（Softplus 激活确保非负）
        evidence = F.softplus(valid_logits)      # e_k = Softplus(z_k)
        alpha = evidence + 1.0                   # alpha_k = e_k + 1
        S = torch.sum(alpha, dim=1, keepdim=True)  # S = sum(alpha_k)

        # 拟合项：基于双伽马函数的期望对数似然
        # L_fit = sum_k y_k * (digamma(S) - digamma(alpha_k))
        loss_fit_per_pixel = torch.sum(
            y * (torch.digamma(S) - torch.digamma(alpha)), dim=1)  # (N_valid,)

        # KL 正则化项（可选标准版或简化版）
        if self.use_standard_kl:
            # 标准狄利克雷 KL 散度（精确实现，OOD 区分度更强）
            kl_penalty_per_pixel = _standard_dirichlet_kl(alpha, y, num_classes)
        else:
            # 简化版 KL 惩罚（仅惩罚非目标类别的证据总量，速度更快）
            kl_penalty_per_pixel = torch.sum((1 - y) * evidence, dim=1)

        # 基于全局迭代步数的 KL 权重线性退火调度
        anneal_coef = min(1.0, current_iter / max(self.anneal_iters, 1)) * self.kl_weight

        # 联合应用像素级类别权重
        weighted_fit = loss_fit_per_pixel * pixel_weights
        weighted_kl = kl_penalty_per_pixel * pixel_weights

        # 最终融合损失
        total_loss = weighted_fit.mean() + anneal_coef * weighted_kl.mean()
        return self.loss_weight * total_loss

    @property
    def loss_name(self):
        """返回损失名称，用于 MMEngine 日志记录。"""
        return self._loss_name
