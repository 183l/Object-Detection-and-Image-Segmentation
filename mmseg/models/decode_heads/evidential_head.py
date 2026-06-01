import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from .segformer_head import SegformerHead


@MODELS.register_module()
class EvidentialHead(SegformerHead):
    """证据解码头（Evidential Decode Head）。

    基于证据深度学习（EDL）框架，将网络输出的原始 logits 转换为
    狄利克雷分布（Dirichlet Distribution）的期望概率，从而同时实现
    语义分割预测与认知不确定性估计。

    推理流程：
        1. 通过 Softplus 激活将 logits 映射为非负证据量 e_k = Softplus(z_k)
        2. 计算狄利克雷浓度参数 alpha_k = e_k + 1
        3. 计算总证据量 S = sum(alpha_k)
        4. 输出类别期望概率 p_k = alpha_k / S

    注意：
        本模块重写了 predict_by_feat，使 SegDataSample.seg_logits 字段
        存储的是 Dirichlet 期望概率（范围 0~1），而非原始 logits。
        eval_ece_unified.py 中的 edl 分支依赖此语义约定。
    """

    def predict_by_feat(self, seg_logits, batch_img_metas):
        """将原始 logits 转换为 Dirichlet 期望概率后交付父类进行后处理。

        Args:
            seg_logits (Tensor): 解码头前向输出的原始 logits，
                shape 为 (N, C, H, W)。
            batch_img_metas (list[dict]): 每张图像的元信息列表。

        Returns:
            Tensor: 经过 Resize 后处理的 Dirichlet 期望概率图，
                shape 为 (N, C, H_orig, W_orig)，数值范围 [0, 1]。
        """
        # =======================================================
        # RoadCalib 核心拦截：Logits -> Dirichlet Probabilities
        # =======================================================
        evidence = F.softplus(seg_logits)           # 非负证据量 e_k
        alpha = evidence + 1.0                       # 狄利克雷浓度参数 alpha_k
        S = torch.sum(alpha, dim=1, keepdim=True)   # 总证据量 S
        prob = alpha / S                             # 类别期望概率 p_k = alpha_k / S

        # 将期望概率传给父类，由父类完成 Resize 等标准后处理
        return super().predict_by_feat(prob, batch_img_metas)
