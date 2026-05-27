import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from .segformer_head import SegformerHead
@MODELS.register_module()
class EvidentialHead(SegformerHead):
    def predict_by_feat(self, seg_logits, batch_img_metas):
    # =======================================================
    # RoadCalib 核心拦截：Logits -> Dirichlet Probabilities
    # =======================================================
        evidence = F.softplus(seg_logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S # 这就是 RoadCalib 预测的概率
        
        # 把算好的概率传给父类，让它去完成缩放 (Resize) 等标准后处理逻辑
        return super().predict_by_feat(prob, batch_img_metas)