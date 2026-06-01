# Object Detection and Image Segmentation

面向道路巡检场景的视觉研究仓库，当前聚焦于"路边反光板缺失检测"任务。

这个仓库基于 [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 二次开发，现阶段公开的内容以 `SegFormer` 语义分割 baseline 为主，同时引入了**证据深度学习（Evidential Deep Learning, EDL）** 模块，用于对模型预测的不确定性进行估计与校准。

---

## 1. 项目目标

目标是从道路巡检视频中自动识别路边反光板是否存在缺失、损坏或异常区域，并逐步支撑两类落地方向：

- **实时巡检**：面向车载或边缘设备的视频流分析
- **离线分析**：面向批量巡检视频的自动检测与告警

---

## 2. 当前公开版本包含什么

当前公开版本更接近"研究型 baseline 仓库"，主要包含：

- 基于 `MMSegmentation` 的 `SegFormer` 训练与评估环境
- 若干 `SegFormer` 配置文件与实验代码
- **自定义 EDL 模块**：`EvidentialHead`（证据解码头）和 `DirichletLoss`（狄利克雷损失函数）
- **统一校准评估脚本**：支持 Baseline（Softmax）和 EDL（Dirichlet）两种模式的 ECE / NLL / Brier Score 评估
- 项目文档、环境说明和数据组织建议
- **单元测试套件**：`tests/test_edl_fixes.py`，覆盖所有核心修复点（10/10 通过）

当前不包含：

- 原始巡检视频
- 业务数据集与标注文件
- 训练权重 `*.pth`
- `work_dirs/` 下的实验输出

---

## 3. 技术路线

当前规划的完整方案分三步推进：

1. **`SegFormer` 语义分割 baseline**
   先完成可复现的像素级分割流程，验证"正常反光板 / 缺失区域 / 背景"的基本表达能力。

2. **`YOLO` 候选框检测**
   先在整帧视频中定位疑似异常区域，再送入分割模型做局部精细判断。

3. **`SAM 2/3` 标注辅助**
   用于视频样本的交互式追踪和快速标注，缩短前期数据制作周期。

---

## 4. 自定义 EDL 模块说明

### 4.1 证据解码头 `EvidentialHead`

`mmseg/models/decode_heads/evidential_head.py`

基于证据深度学习（EDL）框架，将 SegFormer 解码头输出的原始 logits 转换为
**狄利克雷分布（Dirichlet Distribution）的期望概率**，同时实现语义分割预测与认知不确定性估计。

推理流程：

```
logits z_k  →  Softplus  →  evidence e_k = Softplus(z_k)
            →  alpha_k = e_k + 1
            →  S = sum(alpha_k)
            →  prob p_k = alpha_k / S  ∈ [0, 1]
```

### 4.2 狄利克雷损失函数 `DirichletLoss`

`mmseg/models/losses/dirichlet_loss.py`

损失由**拟合项**和**KL 散度正则化项**组成，并支持基于全局迭代步数的线性退火调度：

```
L = L_fit + anneal_coef * L_kl

L_fit = sum_k y_k * (digamma(S) - digamma(alpha_k))
L_kl  = KL(Dir(alpha_tilde) || Dir(1))
```

支持两种 KL 实现：
- `use_standard_kl=False`（默认）：简化版，速度快，向后兼容
- `use_standard_kl=True`：标准狄利克雷 KL 散度（精确实现，OOD 区分度更强）

### 4.3 EDL 训练配置

`configs/segformer/segformer_mit-b2_edl_road_reflector.py`

专用于 EDL 实验的训练配置文件，正确集成了 `EvidentialHead` 和 `DirichletLoss`。

```bash
# 使用 EDL 配置训练
python tools/train.py configs/segformer/segformer_mit-b2_edl_road_reflector.py \
    --work-dir work_dirs/segformer_mit-b2_edl
```

---

## 5. 当前仓库里最相关的文件

| 文件/目录 | 说明 |
|:---|:---|
| `configs/segformer/` | SegFormer 训练配置（含 EDL 专用配置） |
| `mmseg/models/decode_heads/evidential_head.py` | 证据解码头（EDL 核心模块） |
| `mmseg/models/losses/dirichlet_loss.py` | 狄利克雷损失函数（EDL 核心模块） |
| `eval_ece_unified.py` | 统一校准指标评估脚本（支持 Baseline 和 EDL 双模式） |
| `roadcalib.py` | 早期校准分析脚本（Baseline 模式） |
| `tools/calc_weights.py` | 类别权重计算脚本 |
| `tests/test_edl_fixes.py` | 单元测试套件（10 项测试，覆盖所有核心修复点） |
| `docs/issues/ISSUES.md` | 代码审查问题追踪记录（7 个已修复缺陷） |

---

## 6. 环境说明

### 6.1 本地 Windows 开发环境示例

```bash
python -m venv openmmlab20
openmmlab20\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmengine==0.10.7
git clone https://github.com/183l/Object-Detection-and-Image-Segmentation.git
cd Object-Detection-and-Image-Segmentation
pip install -v -e .
python -c "import mmcv, mmengine, mmseg; print(mmcv.__version__); print(mmengine.__version__); print(mmseg.__version__)"
```

对应的期望版本：

| 组件 | 版本 |
|:---|:---|
| PyTorch | `2.0.1` |
| CUDA | `11.8` |
| MMCV | `2.1.0` |
| MMEngine | `0.10.7` |
| MMSegmentation | `1.2.2` |

### 6.2 当前服务器实验环境

| 组件 | 版本 |
|:---|:---|
| Python | `3.8.20` |
| PyTorch | `1.10.2+cu102` |
| TorchVision | `0.11.3+cu102` |
| MMCV | `2.1.0` |
| MMEngine | `0.7.1` |
| MMSegmentation | `1.2.2` |
| GPU | Tesla V100 32GB |

> 说明：cuda 版本与 torch 版本需根据实际情况调整，但不影响正常代码的运行。

---

## 7. 快速开始

### 7.1 克隆仓库

```bash
git clone https://github.com/183l/Object-Detection-and-Image-Segmentation.git
cd Object-Detection-and-Image-Segmentation
```

### 7.2 安装依赖

```bash
conda create -n segformer python=3.8 -y
conda activate segformer
pip install -U pip setuptools wheel
pip install mmengine==0.10.7
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -v -e .
```

### 7.3 训练

**Baseline（标准 CrossEntropy）：**

```bash
python tools/train.py configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py \
    --work-dir work_dirs/segformer_mit-b2_baseline
```

**EDL（证据深度学习）：**

```bash
python tools/train.py configs/segformer/segformer_mit-b2_edl_road_reflector.py \
    --work-dir work_dirs/segformer_mit-b2_edl
```

### 7.4 评估

```bash
# 标准测试（mIoU）
python tools/test.py <config> <checkpoint>

# 统一校准指标评估（ECE / NLL / Brier Score）
# Baseline 模式
python eval_ece_unified.py <config> <checkpoint> --method baseline --out results_baseline.json

# EDL 模式
python eval_ece_unified.py <config> <checkpoint> --method edl --out results_edl.json
```

### 7.5 运行单元测试

```bash
# 安装测试依赖
pip install torch mmengine

# 运行所有修复验证测试（不依赖完整 MMSegmentation 环境）
python tests/test_edl_fixes.py
```

预期输出：

```
=================================================================
  道路反光板检测项目 - Bug 修复单元测试
=================================================================
[PASS] Issue #2 [NLL-baseline] baseline 模式 NLL 计算修复
[PASS] Issue #2 [NLL-edl]     edl 模式 NLL 计算修复（防二次 softmax）
[PASS] Issue #3 [ECE-bin]     ECE bin 边界修复（conf=0 不丢失）
[PASS] Issue #4 [ECE-cls]     Class-wise ECE 分母修复（基于真实标签）
[PASS] Issue #5 [Config]      配置文件 max_iters 统一变量管理
[PASS] Issue #6 [Brier]       Brier Score dtype 修复（int64->float32）
[PASS] Issue #7 [KL-std]      标准狄利克雷 KL 散度数学正确性
[PASS] 集成测试 [DirichletLoss] 端到端前向传播与梯度验证
[PASS] 集成测试 [EvidentialHead] 概率约束与数值稳定性
[PASS] 集成测试 [MessageHub]   MMEngine API 调用正确性
=================================================================
  测试结果：10/10 通过
=================================================================
```

---

## 8. 官方脚本与自定义脚本

| 类型 | 脚本 | 说明 |
|:---|:---|:---|
| 官方通用入口 | `tools/train.py` | 训练主入口 |
| 官方通用入口 | `tools/test.py` | 测试主入口 |
| 自定义评估 | `eval_ece_unified.py` | 统一校准指标评估（Baseline + EDL 双模式） |
| 自定义评估 | `roadcalib.py` | 早期校准分析（Baseline 模式） |
| 自定义工具 | `tools/calc_weights.py` | 类别权重计算 |
| 测试套件 | `tests/test_edl_fixes.py` | 核心修复验证单元测试 |

---

## 9. 仓库结构

```text
.
├─ configs/
│   ├─ _base_/                          # 基础配置（模型、数据集、调度器）
│   └─ segformer/
│       ├─ segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py  # Baseline 配置
│       └─ segformer_mit-b2_edl_road_reflector.py              # EDL 专用配置 ✨
├─ mmseg/
│   └─ models/
│       ├─ decode_heads/
│       │   └─ evidential_head.py       # 证据解码头 ✨
│       └─ losses/
│           └─ dirichlet_loss.py        # 狄利克雷损失函数 ✨
├─ tools/
│   ├─ train.py                         # 官方训练入口
│   ├─ test.py                          # 官方测试入口
│   └─ calc_weights.py                  # 类别权重计算工具
├─ tests/
│   └─ test_edl_fixes.py               # 单元测试套件 ✨
├─ docs/
│   └─ issues/
│       └─ ISSUES.md                    # 问题追踪记录（7 个已修复缺陷）✨
├─ eval_ece_unified.py                  # 统一校准评估脚本 ✨
├─ roadcalib.py                         # 早期校准分析脚本
└─ README.md
```

> ✨ 标注为本次新增或修复的文件。

---

## 10. 代码修复记录（v0.2.0）

本次提交（`fix/edl-bugs-and-integration`）修复了代码审查中发现的 7 个关键缺陷，详见 `docs/issues/ISSUES.md`：

| Issue | 类型 | 涉及文件 | 状态 |
|:---:|:---|:---|:---:|
| #1 | 核心集成缺失：EDL 模块未在配置文件中启用 | `configs/segformer/` | ✅ 已修复 |
| #2 | NLL 计算错误：baseline 模式对 logits 做 `.log()` 导致数值失真 | `eval_ece_unified.py` | ✅ 已修复 |
| #3 | ECE bin 边界：置信度为 0 的像素丢失 | `eval_ece_unified.py`, `roadcalib.py` | ✅ 已修复 |
| #4 | Class-wise ECE 分母语义错位（基于预测值而非真实标签） | `eval_ece_unified.py`, `roadcalib.py` | ✅ 已修复 |
| #5 | 配置文件训练步数与注释不一致 | `configs/segformer/` | ✅ 已修复 |
| #6 | Brier Score 中 `int64`/`float32` 类型不匹配 | `eval_ece_unified.py`, `roadcalib.py` | ✅ 已修复 |
| #7 | KL 散度正则项使用非标准简化实现 | `mmseg/models/losses/dirichlet_loss.py` | ✅ 已修复 |

---

## 11. 当前阶段的待办事项

1. 收集并整理包含"缺失反光板"的巡检视频片段
2. 建立视频抽帧与数据清洗流程
3. 跑通 `SegFormer-B0 ~ B3` 的基线实验与速度评估
4. 定义反光板任务专用数据集与标注规范
5. 逐步接入 `YOLO + SegFormer` 二阶段方案
6. 在真实数据集上对比 Baseline 和 EDL 的 ECE / NLL / Brier Score 指标
