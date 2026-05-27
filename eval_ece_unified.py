import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

# Cityscapes 19 类别名称，方便论文制表
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Unified ECE Evaluation Script for Baseline and EDL')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--out', default=None, help='Output JSON path')
    parser.add_argument('--n-bins', type=int, default=15, help='Number of bins for ECE calculation')
    # ==========================================
    # 控制使用哪种概率计算法则
    # ==========================================
    parser.add_argument('--method', type=str, default='edl', choices=['baseline', 'edl'], 
                        help='Choose probability calculation method: "baseline" (Softmax) or "edl" (Dirichlet Evidence)')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = cfg.get('work_dir', './work_dirs/eval_tmp')
    
    runner = Runner.from_cfg(cfg)
    model = runner.model
    
    # === 【已修复】：外层改用单引号，彻底解决内层大括号双引号冲突 ===
    print(f'\n[{"="*10} {args.method.upper()} EVAL MODE {"="*10}]')
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    dataloader = runner.val_dataloader
    n_bins = args.n_bins
    ignore_index = 255
    num_classes = model.decode_head.num_classes 

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # --- 整体统计量 ---
    conf_acc_counts = torch.zeros(n_bins)
    conf_sum = torch.zeros(n_bins)
    acc_sum = torch.zeros(n_bins)
    total_nll, total_brier, total_pixels = 0.0, 0.0, 0
    total_intersect = torch.zeros(num_classes)
    total_union = torch.zeros(num_classes)
    
    # --- 类别级统计量 (Class-wise) ---
    conf_acc_counts_cls = torch.zeros((num_classes, n_bins))
    conf_sum_cls = torch.zeros((num_classes, n_bins))
    acc_sum_cls = torch.zeros((num_classes, n_bins))
    total_pixels_cls = torch.zeros(num_classes)

    print(f'Evaluating on {len(dataloader.dataset)} images with {args.method.upper()} formula...')
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc='Eval')):
            out = model.test_step(data)
            
            # 【核心机制对齐】：抓取模型 test_step 吐出的逐像素结果
            logits = out[0].seg_logits.data
            target = out[0].gt_sem_seg.data
            
            if target.dim() == 3:
                target = target.squeeze(0)

            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits.unsqueeze(0), size=target.shape[-2:], 
                                       mode='bilinear', align_corners=False).squeeze(0)

            logits = logits.permute(1, 2, 0).reshape(-1, num_classes)
            target = target.reshape(-1)
            
            mask = target != ignore_index
            if mask.sum() == 0: 
                continue
            
            logits_valid = logits[mask]
            target_valid = target[mask]
            
            if args.method == 'baseline':
                # 基线模型：传统的原始未激活得分，依然需要过一次 Softmax 映射为相对概率
                probs = F.softmax(logits_valid, dim=1)
                
            elif args.method == 'edl':
                # RoadCalib 模型：
                # 【破案关键】：因为 model.test_step 内部已经执行了自定义的 EvidentialHead.predict_by_feat，
                # 并在内部将 logits 转换成了期望概率 prob = alpha / S 交付给大盘。
                # 所以此处的 logits_valid 在物理含义上早已经是 0.0 ~ 1.0 之间合法的正统期望概率了！
                # 我们在此处直接放行，拒绝任何二次归一化毒害，还原模型最真实、自信的置信度大盘！
                probs = logits_valid  
            # ====================================================================
            
            conf, pred = probs.max(dim=1)
            acc = pred.eq(target_valid).float()
            
            # --- 整体指标累加 ---
            c_cpu, a_cpu, pred_cpu = conf.cpu(), acc.cpu(), pred.cpu()
            for b in range(n_bins):
                m = (c_cpu > bin_lowers[b]) & (c_cpu <= bin_uppers[b])
                if m.sum() > 0:
                    conf_acc_counts[b] += m.sum()
                    conf_sum[b] += c_cpu[m].sum().item()
                    acc_sum[b] += a_cpu[m].sum().item()

            # 评估 NLL 与 Brier 分数
            total_nll += F.cross_entropy(logits_valid.clamp(min=1e-7).log(), target_valid, reduction='sum').item()
            total_brier += ((probs - F.one_hot(target_valid, num_classes)).pow(2)).sum().sum().item()
            total_pixels += target_valid.size(0)
            
            # --- Class-wise 指标累加 ---
            target_cpu = target_valid.cpu()
            for k in range(num_classes):
                total_intersect[k] += ((pred_cpu == k) & (target_cpu == k)).sum().item()
                total_union[k] += ((pred_cpu == k) | (target_cpu == k)).sum().item()
                
                mask_k = (pred_cpu == k)
                if mask_k.sum() == 0:
                    continue
                
                total_pixels_cls[k] += mask_k.sum().item()
                conf_k = c_cpu[mask_k]
                acc_k = a_cpu[mask_k]
                
                for b in range(n_bins):
                    m_b = (conf_k > bin_lowers[b]) & (conf_k <= bin_uppers[b])
                    if m_b.sum() > 0:
                        conf_acc_counts_cls[k, b] += m_b.sum()
                        conf_sum_cls[k, b] += conf_k[m_b].sum().item()
                        acc_sum_cls[k, b] += acc_k[m_b].sum().item()

    # --- 计算最终整体指标 ---
    ece = torch.abs((conf_sum / conf_acc_counts.clamp(min=1)) - 
                    (acc_sum / conf_acc_counts.clamp(min=1)))
    ece = (ece * conf_acc_counts / total_pixels).sum().item()
    
    ious = total_intersect / total_union.clamp(min=1)
    miou = ious.mean().item()

    # --- 计算 Class-wise ECE ---
    class_ece_list = []
    for k in range(num_classes):
        if total_pixels_cls[k] == 0:
            class_ece_list.append(0.0)
            continue
        ece_k = torch.abs((conf_sum_cls[k] / conf_acc_counts_cls[k].clamp(min=1)) - 
                          (acc_sum_cls[k] / conf_acc_counts_cls[k].clamp(min=1)))
        ece_k = (ece_k * conf_acc_counts_cls[k] / total_pixels_cls[k]).sum().item()
        class_ece_list.append(round(ece_k, 6))

    results = {
        'method': args.method.upper(),
        'miou': round(miou, 6),
        'ece': round(ece, 6),
        'nll': round(total_nll / total_pixels, 6),
        'brier': round(total_brier / total_pixels, 6),
        'per_class_iou': [round(v.item(), 4) for v in ious],
        'per_class_ece': class_ece_list
    }

    # === 【已修复】：打印格式区域全部对齐单双引号，杜绝 SyntaxError ===
    print(f'\n{"="*60}')
    print(f'Model Type   : {results["method"]} ({"Dirichlet EDL" if args.method == "edl" else "Softmax"})')
    print(f'Overall mIoU : {results["miou"]:.4f}')
    print(f'Overall ECE  : {results["ece"]:.6f}')
    print(f'{"="*60}')
    print(f'{"Class Name":<15} | {"IoU":<10} | {"ECE":<10}')
    print(f'{"-"*60}')
    for k in range(num_classes):
        c_name = CITYSCAPES_CLASSES[k] if k < len(CITYSCAPES_CLASSES) else f'Class {k}'
        iou_val = results["per_class_iou"][k]
        ece_val = results["per_class_ece"][k]
        print(f'{c_name:<15} | {iou_val:<10.4f} | {ece_val:<10.6f}')
    print(f'{"="*60}')

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved JSON results to {args.out}')

if __name__ == '__main__':
    main()