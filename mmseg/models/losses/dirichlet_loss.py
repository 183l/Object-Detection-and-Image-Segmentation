import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.logging import MessageHub  # <--- 【核心重构】：引入官方全局唯一信息枢纽

@MODELS.register_module()
class DirichletLoss(nn.Module):
    def __init__(self, 
                 loss_weight=1.0, 
                 class_weight=None, 
                 ignore_index=255, 
                 loss_name='loss_dirichlet', 
                 anneal_iters=3000, 
                 kl_weight=0.001, 
                 **kwargs):
        """
        完全体 DirichletLoss (Digamma 引擎版)
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.class_weight = class_weight  
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        self.anneal_iters = anneal_iters
        self.kl_weight = kl_weight

    def forward(self, cls_score, label, weight=None, **kwargs):
        # ====================================================================
        # 【全网终极修复】：彻底废除 self.current_iter += 1 的自杀式多分支嗑药时钟
        # 直接白嫖 MMEngine 官方 Runner 维护的绝对大盘迭代步数，任凭多尺度分支如何调用，时间永远精准对齐！
        # ====================================================================
        message_hub = MessageHub.get_current_instance()
        if message_hub is not None:
            current_iter = message_hub.get_info('iter')  # 刚性获取大盘真实绝对迭代步数 
        else:
            current_iter = 0

        # 有效像素掩码过滤，剥离 Cityscapes 的 255 无效背景
        valid_mask = (label != self.ignore_index)
        valid_logits = cls_score.permute(0, 2, 3, 1)[valid_mask]
        valid_labels = label[valid_mask]
         
        # 兜底保护：如果整个 batch 全是无效像素，直接返回零梯度，防止显卡报 NaN 崩溃
        if valid_logits.numel() == 0:
            return cls_score.sum() * 0.0
         
        num_classes = valid_logits.shape[1]
        y = F.one_hot(valid_labels, num_classes=num_classes).float()
         
        # ====================================================================
        # 动态应用类别权重映射
        # ====================================================================
        if self.class_weight is not None:
            class_weight_tensor = torch.tensor(self.class_weight, device=valid_logits.device, dtype=torch.float32)
            pixel_weights = class_weight_tensor[valid_labels]
        else:
            pixel_weights = torch.ones_like(valid_labels, dtype=torch.float32)

        # 证据空间转换映射 (Softplus 激活确保非负) [cite: 49, 51, 52]
        evidence = F.softplus(valid_logits)
        alpha = evidence + 1.0                              # 浓度参数 alpha [cite: 51, 53]
        S = torch.sum(alpha, dim=1, keepdim=True)           # 总特殊度/总证据量 S [cite: 54]

        # ====================================================================
        # ====================================================================
        loss_fit_per_pixel = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1) # [cite: 116]
        
        # 认知不确定性强约束项：清道夫机制，无情铲平非目标类的零星灌水证据 [cite: 63, 65]
        kl_penalty_per_pixel = torch.sum((1 - y) * evidence, dim=1) # [cite: 64]
         
        # ====================================================================
        # 基于绝对无偏大盘时间的“梯度均衡动态退火调度” [cite: 16, 83]
        # ====================================================================
        anneal_coef = min(1.0, current_iter / self.anneal_iters) * self.kl_weight # [cite: 84, 89, 92]
        
        # 联合应用像素级截断权重
        weighted_fit = loss_fit_per_pixel * pixel_weights
        weighted_kl = kl_penalty_per_pixel * pixel_weights
        
        # 最终完全体融合损失
        total_loss = weighted_fit.mean() + anneal_coef * weighted_kl.mean() # [cite: 98]
        return self.loss_weight * total_loss # [cite: 99]

    @property
    def loss_name(self):
        return self._loss_name