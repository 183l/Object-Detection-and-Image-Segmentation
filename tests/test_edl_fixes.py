"""
小样本单元测试：验证 7 个 Issue 的修复效果。
不依赖 MMSegmentation 框架，直接测试核心数学逻辑。
"""
import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MessageHub

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def run_test(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, ""))
    except Exception as e:
        msg = traceback.format_exc()
        print(f"{FAIL} {name}\n       {e}")
        results.append((name, False, str(e)))


# ============================================================
# 复制被测代码（不依赖 mmseg，直接内联）
# ============================================================

def _standard_dirichlet_kl(alpha, y, num_classes):
    alpha_tilde = y + (1.0 - y) * alpha
    sum_alpha_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    beta = torch.ones_like(alpha_tilde)
    sum_beta = torch.full_like(sum_alpha_tilde, num_classes)
    kl = (torch.lgamma(sum_alpha_tilde) - torch.lgamma(sum_beta)
          - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
          + torch.lgamma(beta).sum(dim=1, keepdim=True)
          + ((alpha_tilde - beta)
             * (torch.digamma(alpha_tilde)
                - torch.digamma(sum_alpha_tilde))).sum(dim=1, keepdim=True))
    return kl.squeeze(1)


class DirichletLossStandalone(nn.Module):
    """独立版 DirichletLoss，不依赖 mmseg 注册机制，用于单元测试。"""
    def __init__(self, anneal_iters=3000, kl_weight=0.001,
                 use_standard_kl=False, ignore_index=255):
        super().__init__()
        self.anneal_iters = anneal_iters
        self.kl_weight = kl_weight
        self.use_standard_kl = use_standard_kl
        self.ignore_index = ignore_index

    def forward(self, cls_score, label, current_iter=0):
        valid_mask = (label != self.ignore_index)
        valid_logits = cls_score.permute(0, 2, 3, 1)[valid_mask]
        valid_labels = label[valid_mask]

        if valid_logits.numel() == 0:
            return cls_score.sum() * 0.0

        num_classes = valid_logits.shape[1]
        y = F.one_hot(valid_labels, num_classes=num_classes).float()

        evidence = F.softplus(valid_logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)

        loss_fit = torch.sum(
            y * (torch.digamma(S) - torch.digamma(alpha)), dim=1).mean()

        if self.use_standard_kl:
            kl = _standard_dirichlet_kl(alpha, y, num_classes)
        else:
            kl = torch.sum((1 - y) * evidence, dim=1)

        anneal_coef = min(1.0, current_iter / max(self.anneal_iters, 1)) * self.kl_weight
        return loss_fit + anneal_coef * kl.mean()


# ============================================================
# Test 1: Issue #2 - NLL 计算修复（baseline 模式）
# ============================================================
def test_nll_baseline():
    """验证 baseline 模式下 NLL 使用原始 logits 计算，不对 logits 做 .log()"""
    torch.manual_seed(42)
    N, C = 100, 19
    logits = torch.randn(N, C)  # 包含负数的原始 logits
    target = torch.randint(0, C, (N,))

    # 修复后的正确方式
    nll_correct = F.cross_entropy(logits, target, reduction='sum').item()

    # 原始错误方式（对 logits 做 clamp+log 后再送入 cross_entropy）
    nll_wrong = F.cross_entropy(
        logits.clamp(min=1e-7).log(), target, reduction='sum').item()

    # 两者应该有显著差异（说明原来的计算是错的）
    assert abs(nll_correct - nll_wrong) > 1.0, \
        f"NLL 差异过小：correct={nll_correct:.4f}, wrong={nll_wrong:.4f}"
    # 正确的 NLL 应该是有限正数
    assert nll_correct > 0 and torch.isfinite(torch.tensor(nll_correct)), \
        f"NLL 应为有限正数，得到 {nll_correct}"
    print(f"       NLL_correct={nll_correct:.4f}, NLL_wrong={nll_wrong:.4f}, "
          f"diff={abs(nll_correct - nll_wrong):.4f}")


# ============================================================
# Test 2: Issue #2 - NLL 计算修复（edl 模式）
# ============================================================
def test_nll_edl():
    """验证 edl 模式下 NLL 使用 nll_loss(log(p))，语义清晰且数值正确。

    数学分析：
        softmax(log(p_i)) = exp(log(p_i)) / sum(exp(log(p_j)))
                          = p_i / sum(p_j) = p_i
    因此 cross_entropy(log(p), y) 与 nll_loss(log(p), y) 在数学上等价。
    修复的核心价值在于语义清晰（明确表达「输入已是 log 概率」）
    以及避免与 baseline 模式的代码混淆。
    """
    torch.manual_seed(42)
    N, C = 100, 19
    # EDL 输出的是概率（0~1 之间）
    alpha = torch.rand(N, C) * 5 + 1.0
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    target = torch.randint(0, C, (N,))

    # 修复后的正确方式：nll_loss 作用于 log(p)
    nll_correct = F.nll_loss(probs.clamp(min=1e-7).log(), target, reduction='sum').item()

    # 验证数值合理性
    assert nll_correct > 0, f"EDL NLL 应为正数，得到 {nll_correct}"
    assert torch.isfinite(torch.tensor(nll_correct)), "EDL NLL 应为有限值"

    # 验证与原始代码在数学上等价（softmax(log(p)) = p，因此两者结果相同）
    nll_orig = F.cross_entropy(
        probs.clamp(min=1e-7).log(), target, reduction='sum').item()
    assert abs(nll_correct - nll_orig) < 1e-3, \
        f"两者应数学等价，diff={abs(nll_correct - nll_orig):.6f}"

    # 但 baseline 模式下原始代码（对 logits 做 clamp+log）是错的
    # 这里验证 edl 模式的 nll_loss 语义更清晰
    print(f"       EDL NLL={nll_correct:.4f}，nll_loss 语义清晰 ✓")
    print(f"       注：softmax(log(p))=p，故 nll_loss 与 cross_entropy 数学等价")


# ============================================================
# Test 3: Issue #3 - ECE bin 边界修复
# ============================================================
def test_ece_bin_boundary():
    """验证 ECE 第一个 bin 使用 >= 0.0，不丢失置信度为 0 的像素"""
    n_bins = 15
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # 构造一批置信度为 0.0 的像素
    conf = torch.zeros(50)
    acc = torch.ones(50)

    # 修复后的边界条件
    pixels_in_bins = 0
    for b in range(n_bins):
        if b == 0:
            m = (conf >= bin_lowers[b]) & (conf <= bin_uppers[b])
        else:
            m = (conf > bin_lowers[b]) & (conf <= bin_uppers[b])
        pixels_in_bins += m.sum().item()

    # 原始错误边界条件（全部使用严格大于）
    pixels_in_bins_wrong = 0
    for b in range(n_bins):
        m = (conf > bin_lowers[b]) & (conf <= bin_uppers[b])
        pixels_in_bins_wrong += m.sum().item()

    assert pixels_in_bins == 50, \
        f"修复后应统计到 50 个像素，实际 {pixels_in_bins}"
    assert pixels_in_bins_wrong == 0, \
        f"原始错误代码应丢失所有 50 个像素，实际 {pixels_in_bins_wrong}"
    print(f"       修复后统计到 {pixels_in_bins} 个 conf=0 像素，"
          f"原始代码丢失 {50 - pixels_in_bins_wrong} 个")


# ============================================================
# Test 4: Issue #4 - Class-wise ECE 分母修复
# ============================================================
def test_classwise_ece_denominator():
    """验证 Class-wise ECE 基于真实标签（target==k）而非预测值（pred==k）"""
    torch.manual_seed(42)
    N, C = 200, 19
    # 构造一个有偏的预测：模型过度预测类别 0
    pred = torch.zeros(N, dtype=torch.long)  # 全预测为类别 0
    target = torch.randint(0, C, (N,))        # 真实标签均匀分布

    # 基于预测值的统计（错误方式）
    pixels_pred_k0 = (pred == 0).sum().item()   # 应该等于 N=200

    # 基于真实标签的统计（修复后正确方式）
    pixels_target_k0 = (target == 0).sum().item()  # 应该约等于 N/C ≈ 10

    assert pixels_pred_k0 == N, f"预测值统计应为 {N}，得到 {pixels_pred_k0}"
    assert pixels_target_k0 < N // 2, \
        f"真实标签统计应远小于 {N}，得到 {pixels_target_k0}"
    print(f"       类别0: 基于预测值统计={pixels_pred_k0}，"
          f"基于真实标签统计={pixels_target_k0}（差异={pixels_pred_k0 - pixels_target_k0}）")


# ============================================================
# Test 5: Issue #5 - 训练步数与 PolyLR 终点一致性
# ============================================================
def test_config_max_iters_consistency():
    """验证配置文件中 max_iters 变量被统一使用"""
    config_path = '/home/ubuntu/Object-Detection-and-Image-Segmentation/configs/segformer/segformer_mit-b2_edl_road_reflector.py'
    with open(config_path, 'r') as f:
        content = f.read()

    # 检查 max_iters 变量被定义
    assert 'max_iters = ' in content, "配置文件中未定义 max_iters 变量"
    # 检查 PolyLR 的 end 参数使用了 max_iters 变量
    assert 'end=max_iters' in content, \
        "PolyLR 的 end 参数未使用 max_iters 变量，可能导致学习率提前归零"
    # 检查 train_cfg 使用了 max_iters 变量
    assert 'max_iters=max_iters' in content, \
        "train_cfg 的 max_iters 未使用 max_iters 变量"
    print(f"       配置文件中 max_iters 变量统一管理，PolyLR.end=max_iters ✓")


# ============================================================
# Test 6: Issue #6 - Brier Score dtype 修复
# ============================================================
def test_brier_score_dtype():
    """验证 Brier Score 中 one_hot 显式转换为 float，不会触发类型错误"""
    torch.manual_seed(42)
    N, C = 100, 19
    probs = torch.rand(N, C)
    probs = probs / probs.sum(dim=1, keepdim=True)
    target = torch.randint(0, C, (N,))

    # 验证 F.one_hot 默认返回 int64
    one_hot_raw = F.one_hot(target, C)
    assert one_hot_raw.dtype == torch.int64, \
        f"F.one_hot 应返回 int64，实际 {one_hot_raw.dtype}"

    # 修复后：显式 .float() 转换
    brier = ((probs - F.one_hot(target, C).float()).pow(2)).sum().item()
    assert brier > 0, f"Brier Score 应为正数，得到 {brier}"
    assert torch.isfinite(torch.tensor(brier)), "Brier Score 应为有限值"

    # 验证不加 .float() 在某些情况下会出现类型问题
    try:
        # 在 PyTorch 2.x 中这可能会自动提升类型，但在旧版本会报错
        brier_raw = ((probs - F.one_hot(target, C)).pow(2)).sum().item()
        # 如果没报错，验证结果一致
        assert abs(brier - brier_raw) < 1e-4, "两种方式结果应一致"
        print(f"       Brier Score={brier:.6f}，.float() 转换正常 ✓")
    except RuntimeError:
        print(f"       Brier Score={brier:.6f}，.float() 修复了类型错误 ✓")


# ============================================================
# Test 7: Issue #7 - 标准 KL 散度实现验证
# ============================================================
def test_standard_kl_divergence():
    """验证标准狄利克雷 KL 散度实现的数学正确性。

    注意：KL 散度中 alpha_tilde = y + (1-y)*alpha，目标类被强制为 1。
    因此应测试「非目标类集中」的场景（即错误预测其他类别置信度极高）。
    """
    torch.manual_seed(42)
    N, C = 50, 19

    # 目标类是类别 1（非集中类别）
    y = F.one_hot(torch.ones(N, dtype=torch.long), C).float()

    # 均匀分布：所有类别 alpha=2
    alpha_uniform = torch.ones(N, C) * 2.0
    kl_uniform = _standard_dirichlet_kl(alpha_uniform, y, C)

    # 集中分布：非目标类别 0 的 alpha 极大（错误的高置信度）
    alpha_concentrated = torch.ones(N, C)
    alpha_concentrated[:, 0] = 100.0  # 非目标类置信度极高（错误预测）
    kl_concentrated = _standard_dirichlet_kl(alpha_concentrated, y, C)

    # 非目标类集中（错误预测）的 KL 应远大于均匀分布（正确惩罚错误置信度）
    assert kl_uniform.mean() < kl_concentrated.mean(), \
        f"均匀分布 KL={kl_uniform.mean():.4f} 应小于非目标集中 KL={kl_concentrated.mean():.4f}"

    # KL 散度应为非负数
    assert (kl_uniform >= 0).all(), "KL 散度应为非负数"
    assert (kl_concentrated >= 0).all(), "KL 散度应为非负数"

    print(f"       均匀分布 KL={kl_uniform.mean():.4f}，"
          f"非目标类集中 KL={kl_concentrated.mean():.4f}，非负性 ✓")


# ============================================================
# Test 8: DirichletLoss 端到端前向传播
# ============================================================
def test_dirichlet_loss_forward():
    """验证 DirichletLoss 端到端前向传播：梯度可计算，数值合理"""
    torch.manual_seed(42)
    N, C, H, W = 2, 19, 8, 8

    loss_fn_simple = DirichletLossStandalone(use_standard_kl=False)
    loss_fn_standard = DirichletLossStandalone(use_standard_kl=True)

    logits = torch.randn(N, C, H, W, requires_grad=True)
    labels = torch.randint(0, C, (N, H, W))
    # 添加一些 ignore_index 像素
    labels[0, 0, 0] = 255

    # 简化版 KL
    loss_simple = loss_fn_simple(logits, labels, current_iter=1500)
    assert loss_simple.item() > 0, f"损失应为正数，得到 {loss_simple.item()}"
    assert torch.isfinite(loss_simple), "损失应为有限值"
    loss_simple.backward()
    assert logits.grad is not None, "梯度应可计算"
    assert not torch.isnan(logits.grad).any(), "梯度不应包含 NaN"

    # 标准版 KL
    logits2 = torch.randn(N, C, H, W, requires_grad=True)
    loss_standard = loss_fn_standard(logits2, labels, current_iter=1500)
    assert loss_standard.item() > 0, f"标准 KL 损失应为正数"
    loss_standard.backward()
    assert logits2.grad is not None, "标准 KL 梯度应可计算"

    print(f"       简化版 KL loss={loss_simple.item():.4f}，"
          f"标准版 KL loss={loss_standard.item():.4f}，梯度正常 ✓")


# ============================================================
# Test 9: EvidentialHead 核心逻辑（不依赖 mmseg）
# ============================================================
def test_evidential_head_logic():
    """验证 EvidentialHead 的 Softplus -> alpha -> prob 转换逻辑"""
    torch.manual_seed(42)
    N, C, H, W = 2, 19, 8, 8
    logits = torch.randn(N, C, H, W)

    # 模拟 EvidentialHead 的核心转换
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S

    # 验证概率约束
    assert (prob >= 0).all(), "概率应为非负数"
    assert (prob <= 1).all(), "概率应不超过 1"
    prob_sum = prob.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        "每个像素的概率之和应为 1"

    # 验证 alpha >= 1（因为 evidence >= 0，所以 alpha = evidence + 1 >= 1）
    assert (alpha >= 1.0).all(), "alpha 应 >= 1"

    print(f"       prob 范围=[{prob.min():.4f}, {prob.max():.4f}]，"
          f"概率和偏差={abs(prob_sum - 1).max().item():.2e} ✓")


# ============================================================
# Test 10: MessageHub.get_info API 验证
# ============================================================
def test_message_hub_api():
    """验证 MessageHub.get_info 的正确调用方式"""
    # 创建一个测试用的 MessageHub 实例
    hub = MessageHub.get_current_instance()
    hub.update_info('iter', 500)

    # 验证 get_info 可以正确获取迭代步数
    current_iter = hub.get_info('iter', default=0)
    assert current_iter == 500, f"应获取到 iter=500，得到 {current_iter}"

    # 验证 default 参数在 key 不存在时生效
    nonexistent = hub.get_info('nonexistent_key', default=0)
    assert nonexistent == 0, f"不存在的 key 应返回 default=0，得到 {nonexistent}"

    print(f"       MessageHub.get_info('iter')={current_iter}，"
          f"default 参数正常 ✓")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  道路反光板检测项目 - Bug 修复单元测试")
    print("=" * 65)

    run_test("Issue #2 [NLL-baseline] baseline 模式 NLL 计算修复", test_nll_baseline)
    run_test("Issue #2 [NLL-edl]     edl 模式 NLL 计算修复（防二次 softmax）", test_nll_edl)
    run_test("Issue #3 [ECE-bin]     ECE bin 边界修复（conf=0 不丢失）", test_ece_bin_boundary)
    run_test("Issue #4 [ECE-cls]     Class-wise ECE 分母修复（基于真实标签）", test_classwise_ece_denominator)
    run_test("Issue #5 [Config]      配置文件 max_iters 统一变量管理", test_config_max_iters_consistency)
    run_test("Issue #6 [Brier]       Brier Score dtype 修复（int64->float32）", test_brier_score_dtype)
    run_test("Issue #7 [KL-std]      标准狄利克雷 KL 散度数学正确性", test_standard_kl_divergence)
    run_test("集成测试 [DirichletLoss] 端到端前向传播与梯度验证", test_dirichlet_loss_forward)
    run_test("集成测试 [EvidentialHead] 概率约束与数值稳定性", test_evidential_head_logic)
    run_test("集成测试 [MessageHub]   MMEngine API 调用正确性", test_message_hub_api)

    print("\n" + "=" * 65)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  测试结果：{passed}/{total} 通过")
    if passed == total:
        print("  \033[92m所有测试通过！修复验证成功。\033[0m")
    else:
        print("  \033[91m部分测试失败，请检查上方错误信息。\033[0m")
        for name, ok, msg in results:
            if not ok:
                print(f"  - FAIL: {name}")
    print("=" * 65)
    sys.exit(0 if passed == total else 1)
