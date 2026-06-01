# Issues 问题追踪记录

本文件记录代码审查过程中发现的 7 个关键缺陷，对应 commit `fix/edl-bugs-and-integration`。

---

## Issue #1 [Bug] 核心集成缺失：EvidentialHead 和 DirichletLoss 未在配置文件中被启用

**标签**：`bug` `critical`

### 问题描述

自定义的 `EvidentialHead` 和 `DirichletLoss` 模块已在 `mmseg/models/` 中实现并注册，但当前唯一的训练配置文件 `configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py` 仅继承了基础配置，未对 `decode_head.type` 和 `loss_decode.type` 进行任何覆写。

### 影响

直接运行训练脚本将完全退化为标准的 `SegformerHead + CrossEntropyLoss` 训练，EDL 代码处于完全闲置状态。

### 涉及文件

- `configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py`
- `configs/_base_/models/segformer_mit-b0.py`

### 修复方案

新增专用 EDL 实验配置文件 `configs/segformer/segformer_mit-b2_edl_road_reflector.py`，显式指定 `EvidentialHead` 和 `DirichletLoss`。

**状态**：已修复 ✅

---

## Issue #2 [Bug] eval_ece_unified.py 中 NLL 计算公式在 baseline 模式下严重错误

**标签**：`bug` `critical` `evaluation`

### 问题描述

`eval_ece_unified.py` 第 122 行对两种模式统一使用了如下 NLL 计算：

```python
total_nll += F.cross_entropy(logits_valid.clamp(min=1e-7).log(), target_valid, reduction='sum').item()
```

**在 `baseline` 模式下**：`logits_valid` 是原始未归一化得分（含大量负数），对其执行 `.clamp(min=1e-7).log()` 会将所有负数强行截断为 `1e-7`，随后 `log(1e-7) ≈ -16.1`，彻底破坏了原始得分的数值分布。

**在 `edl` 模式下**：`logits_valid` 已经是概率 $p$，取 `log(p)` 后送入 `F.cross_entropy`，该函数内部会再执行一次 `LogSoftmax`，造成二次变换，结果不精确。

### 影响

评估报告中的 NLL 指标完全失真，无法用于学术论文制表与对比。

### 修复方案

按模式分支计算：
```python
if args.method == 'baseline':
    nll_val = F.cross_entropy(logits_valid, target_valid, reduction='sum').item()
elif args.method == 'edl':
    nll_val = F.nll_loss(probs.clamp(min=1e-7).log(), target_valid, reduction='sum').item()
```

**状态**：已修复 ✅

---

## Issue #3 [Bug] ECE 区间划分导致置信度为 0 的像素丢失

**标签**：`bug` `evaluation`

### 问题描述

ECE 计算中，第一个置信度区间的边界条件使用了严格大于号：

```python
m = (c_cpu > bin_lowers[b]) & (c_cpu <= bin_uppers[b])
```

`bin_lowers[0] = 0.0`，因此置信度恰好等于 0.0 的像素被排除在所有区间之外，无法参与 ECE 统计，导致统计像素总数对不上。

### 修复方案

第一个 bin 改用闭区间 `>=`：

```python
if b == 0:
    m = (c_cpu >= bin_lowers[b]) & (c_cpu <= bin_uppers[b])
else:
    m = (c_cpu > bin_lowers[b]) & (c_cpu <= bin_uppers[b])
```

**状态**：已修复 ✅

---

## Issue #4 [Bug] Class-wise ECE 统计分母语义错位（基于预测值而非真实标签）

**标签**：`bug` `evaluation`

### 问题描述

类别级 ECE 统计中，`total_pixels_cls[k]` 累加的是模型**预测为类别 k** 的像素数（`pred_cpu == k`），而非**真实标签为类别 k** 的像素数（`target_cpu == k`）。这导致计算出的 Class-wise ECE 反映的是"模型预测集合上的置信度校准"，而非"真实类别分布上的置信度校准"。

### 修复方案

将 `mask_k` 的依据从预测值改为真实标签：

```python
mask_k = (target_cpu == k)  # 修复：基于真实标签
```

**状态**：已修复 ✅

---

## Issue #5 [Bug] 训练配置文件中迭代步数与注释严重不一致

**标签**：`bug` `config`

### 问题描述

`configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py` 中多处注释写明"20K 快速实验"，但实际代码中 `max_iters=10000`，`PolyLR` 的 `end=10000`，注释中还标注 `# <--- 终点设为 20K`，形成严重矛盾。

若后续将训练步数修改为 20K 但遗漏修改 `PolyLR` 的 `end`，学习率将在 10K 步时提前降为 0，导致模型后半段无法收敛。

### 修复方案

统一使用变量 `max_iters` 管理训练步数，确保 `train_cfg` 和 `param_scheduler` 使用同一变量：

```python
max_iters = 10000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1000)
param_scheduler = [
    ...,
    dict(type='PolyLR', ..., end=max_iters, ...)
]
```

**状态**：已修复 ✅

---

## Issue #6 [Bug] Brier Score 计算中存在 int64/float32 数据类型不匹配隐患

**标签**：`bug` `evaluation`

### 问题描述

`eval_ece_unified.py` 和 `roadcalib.py` 中 Brier Score 计算：

```python
total_brier += ((probs - F.one_hot(target_valid, num_classes)).pow(2)).sum().sum().item()
```

`F.one_hot` 返回 `torch.int64` 类型张量，而 `probs` 为 `torch.float32`。在部分 PyTorch 版本或硬件环境下，两者直接做减法会触发类型不匹配错误或隐式类型提升，增加显存开销。

### 修复方案

显式添加 `.float()` 类型转换：

```python
total_brier += ((probs - F.one_hot(target_valid, num_classes).float()).pow(2)).sum().sum().item()
```

**状态**：已修复 ✅

---

## Issue #7 [Enhancement] DirichletLoss 中 KL 散度正则项使用了非标准简化实现

**标签**：`enhancement` `math`

### 问题描述

`mmseg/models/losses/dirichlet_loss.py` 第 70 行的 KL 惩罚项：

```python
kl_penalty_per_pixel = torch.sum((1 - y) * evidence, dim=1)
```

这是标准狄利克雷 KL 散度 $\text{KL}(\text{Dir}(\tilde{\alpha}) \| \text{Dir}(\mathbf{1}))$ 的极大简化版，仅惩罚非目标类别的证据总量，缺少 `lgamma` 和 `digamma` 项，约束强度显著弱于标准 KL 散度，可能导致 OOD 不确定性区分度不足。

### 修复方案

提供标准 KL 散度实现作为可选项，并通过构造函数参数 `use_standard_kl` 控制：

```python
def standard_dirichlet_kl(alpha, y, num_classes):
    alpha_tilde = y + (1.0 - y) * alpha
    sum_alpha_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    beta = torch.ones_like(alpha_tilde)
    sum_beta = torch.tensor(num_classes, device=alpha.device, dtype=torch.float32)
    kl = (torch.lgamma(sum_alpha_tilde) - torch.lgamma(sum_beta)
          - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
          + torch.lgamma(beta).sum(dim=1, keepdim=True)
          + ((alpha_tilde - beta) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde))).sum(dim=1, keepdim=True))
    return kl.squeeze(1)
```

**状态**：已修复（新增 `use_standard_kl` 参数，默认 `False` 保持向后兼容）✅
