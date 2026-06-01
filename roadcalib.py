"""早期校准分析脚本（Road Calibration Analysis Script）。

仅支持 Baseline（Softmax）模式的 ECE、mIoU、NLL 和 Brier Score 评估。
如需同时支持 EDL 模式，请使用 eval_ece_unified.py。

修复记录：
    - Issue #6: 修复 Brier Score 中 F.one_hot 返回 int64 与 float32 不匹配
      的类型隐患，显式添加 .float() 转换。

用法：
    python roadcalib.py <config> <checkpoint> [--out results.json] [--n-bins 15]
"""
import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

# Cityscapes 19 类别名称
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Road Calibration Analysis Script (Baseline Softmax Mode)')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--out', default=None, help='Output JSON path')
    parser.add_argument('--n-bins', type=int, default=15,
                        help='Number of bins for ECE calculation')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = cfg.get('work_dir', './work_dirs/eval_tmp')

    runner = Runner.from_cfg(cfg)
    model = runner.model

    print(f'Loading checkpoint from {args.checkpoint}...')
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    dataloader = runner.val_dataloader
    n_bins = args.n_bins
    ignore_index = 255
    num_classes = model.decode_head.num_classes

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # 整体统计量
    conf_acc_counts = torch.zeros(n_bins)
    conf_sum = torch.zeros(n_bins)
    acc_sum = torch.zeros(n_bins)
    total_nll, total_brier, total_pixels = 0.0, 0.0, 0
    total_intersect = torch.zeros(num_classes)
    total_union = torch.zeros(num_classes)

    # 类别级统计量（Class-wise）
    conf_acc_counts_cls = torch.zeros((num_classes, n_bins))
    conf_sum_cls = torch.zeros((num_classes, n_bins))
    acc_sum_cls = torch.zeros((num_classes, n_bins))
    total_pixels_cls = torch.zeros(num_classes)

    print(f'Evaluating on {len(dataloader.dataset)} images (Baseline Softmax Mode)...')

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc='Eval')):
            out = model.test_step(data)

            logits = out[0].seg_logits.data
            target = out[0].gt_sem_seg.data

            if target.dim() == 3:
                target = target.squeeze(0)

            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits.unsqueeze(0), size=target.shape[-2:],
                    mode='bilinear', align_corners=False).squeeze(0)

            logits = logits.permute(1, 2, 0).reshape(-1, num_classes)
            target = target.reshape(-1)

            mask = target != ignore_index
            if mask.sum() == 0:
                continue

            logits_valid = logits[mask]
            target_valid = target[mask]

            probs = F.softmax(logits_valid, dim=1)
            conf, pred = probs.max(dim=1)
            acc = pred.eq(target_valid).float()

            # NLL：直接使用原始 logits 计算交叉熵
            total_nll += F.cross_entropy(
                logits_valid, target_valid, reduction='sum').item()

            # 修复 Issue #6：显式 .float() 转换，解决 int64/float32 类型不匹配
            total_brier += (
                (probs - F.one_hot(target_valid, num_classes).float()).pow(2)
            ).sum().item()
            total_pixels += target_valid.size(0)

            c_cpu = conf.cpu()
            a_cpu = acc.cpu()
            pred_cpu = pred.cpu()
            target_cpu = target_valid.cpu()

            # 整体 ECE 累加
            for b in range(n_bins):
                # 修复 Issue #3：第一个 bin 使用闭区间 [0, upper]
                if b == 0:
                    m = (c_cpu >= bin_lowers[b]) & (c_cpu <= bin_uppers[b])
                else:
                    m = (c_cpu > bin_lowers[b]) & (c_cpu <= bin_uppers[b])

                if m.sum() > 0:
                    conf_acc_counts[b] += m.sum()
                    conf_sum[b] += c_cpu[m].sum().item()
                    acc_sum[b] += a_cpu[m].sum().item()

            # 类别级指标累加
            for k in range(num_classes):
                total_intersect[k] += (
                    (pred_cpu == k) & (target_cpu == k)).sum().item()
                total_union[k] += (
                    (pred_cpu == k) | (target_cpu == k)).sum().item()

                # 修复 Issue #4：基于真实标签划分像素集合
                mask_k = (target_cpu == k)
                if mask_k.sum() == 0:
                    continue

                total_pixels_cls[k] += mask_k.sum().item()
                conf_k = c_cpu[mask_k]
                acc_k = a_cpu[mask_k]

                for b in range(n_bins):
                    if b == 0:
                        m_b = (conf_k >= bin_lowers[b]) & (conf_k <= bin_uppers[b])
                    else:
                        m_b = (conf_k > bin_lowers[b]) & (conf_k <= bin_uppers[b])

                    if m_b.sum() > 0:
                        conf_acc_counts_cls[k, b] += m_b.sum()
                        conf_sum_cls[k, b] += conf_k[m_b].sum().item()
                        acc_sum_cls[k, b] += acc_k[m_b].sum().item()

    # 计算最终整体指标
    ece = torch.abs(
        (conf_sum / conf_acc_counts.clamp(min=1)) -
        (acc_sum / conf_acc_counts.clamp(min=1)))
    ece = (ece * conf_acc_counts / total_pixels).sum().item()

    ious = total_intersect / total_union.clamp(min=1)
    miou = ious.mean().item()

    # 计算 Class-wise ECE
    class_ece_list = []
    for k in range(num_classes):
        if total_pixels_cls[k] == 0:
            class_ece_list.append(0.0)
            continue
        ece_k = torch.abs(
            (conf_sum_cls[k] / conf_acc_counts_cls[k].clamp(min=1)) -
            (acc_sum_cls[k] / conf_acc_counts_cls[k].clamp(min=1)))
        ece_k = (ece_k * conf_acc_counts_cls[k] / total_pixels_cls[k]).sum().item()
        class_ece_list.append(round(ece_k, 6))

    results = {
        'miou': round(miou, 6),
        'ece': round(ece, 6),
        'nll': round(total_nll / total_pixels, 6),
        'brier': round(total_brier / total_pixels, 6),
        'per_class_iou': [round(v.item(), 4) for v in ious],
        'per_class_ece': class_ece_list
    }

    print(f'\n{"="*60}')
    print(f'Overall mIoU : {results["miou"]:.4f}')
    print(f'Overall ECE  : {results["ece"]:.6f}')
    print(f'Overall NLL  : {results["nll"]:.6f}')
    print(f'Brier Score  : {results["brier"]:.6f}')
    print(f'{"="*60}')
    print(f'{"Class Name":<15} | {"IoU":<10} | {"ECE":<10}')
    print(f'{"-"*60}')
    for k in range(num_classes):
        c_name = (CITYSCAPES_CLASSES[k]
                  if k < len(CITYSCAPES_CLASSES) else f'Class {k}')
        iou_val = results['per_class_iou'][k]
        ece_val = results['per_class_ece'][k]
        print(f'{c_name:<15} | {iou_val:<10.4f} | {ece_val:<10.6f}')
    print(f'{"="*60}')

    if args.out:
        os.makedirs(
            os.path.dirname(args.out) if os.path.dirname(args.out) else '.',
            exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
