"""统一校准指标评估脚本（Unified ECE Evaluation Script）。

支持对 Baseline（Softmax）和 EDL（Dirichlet Evidence）两种模型进行
期望校准误差（ECE）、mIoU、负对数似然（NLL）和 Brier Score 的统一评估。

修复记录：
    - Issue #2: 修复 baseline 模式下 NLL 计算错误（原代码对 logits 执行
      .clamp().log() 后送入 cross_entropy，导致数值完全失真）。
    - Issue #3: 修复 ECE 第一个 bin 边界条件，改为 >= 防止置信度为 0 的
      像素丢失。
    - Issue #4: 修复 Class-wise ECE 统计分母，改为基于真实标签（target_cpu == k）
      而非预测值（pred_cpu == k）。
    - Issue #6: 修复 Brier Score 中 F.one_hot 返回 int64 与 float32 不匹配的
      类型隐患，显式添加 .float() 转换。

用法：
    python eval_ece_unified.py <config> <checkpoint> [--method baseline|edl]
                               [--out results.json] [--n-bins 15]
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
        description='Unified ECE Evaluation Script for Baseline and EDL')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--out', default=None, help='Output JSON path')
    parser.add_argument('--n-bins', type=int, default=15,
                        help='Number of bins for ECE calculation')
    parser.add_argument(
        '--method', type=str, default='edl',
        choices=['baseline', 'edl'],
        help='Choose probability calculation method: '
             '"baseline" (Softmax) or "edl" (Dirichlet Evidence)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = cfg.get('work_dir', './work_dirs/eval_tmp')

    runner = Runner.from_cfg(cfg)
    model = runner.model

    print(f'\n[{"="*10} {args.method.upper()} EVAL MODE {"="*10}]')
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

    print(f'Evaluating on {len(dataloader.dataset)} images '
          f'with {args.method.upper()} formula...')

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc='Eval')):
            out = model.test_step(data)

            # 获取模型输出的逐像素结果
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

            # ====================================================================
            # 修复 Issue #2：按模式分支计算概率和 NLL，避免对 logits 执行错误变换
            # ====================================================================
            if args.method == 'baseline':
                # Baseline 模式：输入是原始未归一化 logits
                probs = F.softmax(logits_valid, dim=1)
                # NLL 直接使用原始 logits 计算交叉熵（F.cross_entropy 内部做 log_softmax）
                nll_val = F.cross_entropy(
                    logits_valid, target_valid, reduction='sum').item()

            elif args.method == 'edl':
                # EDL 模式：EvidentialHead.predict_by_feat 已将 logits 转换为
                # 狄利克雷期望概率 p = alpha/S，数值范围 [0, 1]
                probs = logits_valid
                # NLL 使用 nll_loss 作用于 log(p)，拒绝二次 softmax 变换
                nll_val = F.nll_loss(
                    probs.clamp(min=1e-7).log(),
                    target_valid, reduction='sum').item()
            # ====================================================================

            total_nll += nll_val

            # 修复 Issue #6：显式 .float() 转换，解决 int64/float32 类型不匹配
            total_brier += (
                (probs - F.one_hot(target_valid, num_classes).float()).pow(2)
            ).sum().item()
            total_pixels += target_valid.size(0)

            conf, pred = probs.max(dim=1)
            acc = pred.eq(target_valid).float()

            c_cpu = conf.cpu()
            a_cpu = acc.cpu()
            pred_cpu = pred.cpu()
            target_cpu = target_valid.cpu()

            # 整体 ECE 累加
            for b in range(n_bins):
                # 修复 Issue #3：第一个 bin 使用闭区间 [0, upper]，防止 conf=0 丢失
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
                # mIoU 统计：intersect = TP，union = TP + FP + FN
                total_intersect[k] += (
                    (pred_cpu == k) & (target_cpu == k)).sum().item()
                total_union[k] += (
                    (pred_cpu == k) | (target_cpu == k)).sum().item()

                # 修复 Issue #4：Class-wise ECE 基于真实标签（target_cpu == k）
                # 而非预测值，反映真实类别分布上的置信度校准表现
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
        'method': args.method.upper(),
        'miou': round(miou, 6),
        'ece': round(ece, 6),
        'nll': round(total_nll / total_pixels, 6),
        'brier': round(total_brier / total_pixels, 6),
        'per_class_iou': [round(v.item(), 4) for v in ious],
        'per_class_ece': class_ece_list
    }

    print(f'\n{"="*60}')
    print(f'Model Type   : {results["method"]} '
          f'({"Dirichlet EDL" if args.method == "edl" else "Softmax"})')
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
        print(f'\nSaved JSON results to {args.out}')


if __name__ == '__main__':
    main()
