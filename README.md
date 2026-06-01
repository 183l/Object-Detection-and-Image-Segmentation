# 道路巡检反光板检测系统

> 基于 **证据深度学习（Evidential Deep Learning, EDL）** 的语义分割框架，在 MMSegmentation 基础上深度重构，实现道路反光板的高精度分割与可靠的认知不确定性估计。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![MMSeg](https://img.shields.io/badge/MMSegmentation-1.x-green)](https://github.com/open-mmlab/mmsegmentation)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

---

## 项目简介

本项目以 [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 为基础框架，针对道路巡检场景中反光板（Road Reflector）缺失与损坏检测任务，引入了**证据深度学习（Evidential Deep Learning, EDL）**机制进行深度二次开发。

目标是从道路巡检视频中自动识别路边反光板是否存在缺失、损坏或异常区域，并逐步支撑两类落地方向：
- **实时巡检**：面向车载或边缘设备的视频流分析
- **离线分析**：面向批量巡检视频的自动检测与告警

传统的 Softmax 语义分割网络在面对小样本、高噪声或分布外（Out-of-Distribution, OOD）样本时，存在严重的"过度自信"（Overconfidence）缺陷。本项目通过将网络输出建模为**狄利克雷分布（Dirichlet Distribution）**的参数，使模型在输出分割预测的同时，提供可靠的**认知不确定性**（Cognitive Uncertainty）指标，为安全关键的道路巡检决策提供风险量化依据。

---

## 核心算法：证据深度学习（EDL）

### 数学原理

对于每个像素，网络输出的 logits $z$ 经过 Softplus 激活后得到非负证据量 $e_k$，进而映射为狄利克雷分布参数 $\alpha_k$：

$$e_k = \ln(1 + \exp(z_k)) \ge 0, \quad \alpha_k = e_k + 1$$

狄利克雷分布的精度和 $S = \sum_k \alpha_k$ 决定了预测的期望概率与认知不确定性：

$$\hat{p}_k = \frac{\alpha_k}{S}, \quad u = \frac{K}{S}$$

其中 $u \in (0, 1]$ 越大，表示模型对该像素的预测越不确定（证据越少）。

### 推理流程

```
logits z_k  →  Softplus  →  evidence e_k = Softplus(z_k)
            →  alpha_k = e_k + 1
            →  S = sum(alpha_k)
            →  prob p_k = alpha_k / S  ∈ [0, 1]
            →  uncertainty u = K / S  ∈ (0, 1]
```

### 损失函数

总损失由**期望交叉熵拟合项**与**标准狄利克雷 KL 散度正则化项**组成，并通过线性退火系数 $\lambda_t$ 控制正则化强度：

$$L = L_{\text{fit}} + \lambda_t \cdot L_{\text{kl}}, \quad \lambda_t = \min\!\left(1.0,\, \frac{t}{T_{\text{anneal}}}\right)$$

支持两种 KL 实现：
- `use_standard_kl=False`（默认）：简化版，速度快，向后兼容
- `use_standard_kl=True`：标准狄利克雷 KL 散度（精确实现，OOD 区分度更强）

---

## 实验结果对比（v0.2.0）

在 3 类高噪声小样本数据集（300 训练样本 / 500 测试样本，噪声方差 σ=1.2）上，重构后的 EDL 算法相比传统 Softmax 基线取得了全面优势：

| 测评指标 | 传统 Softmax 基线 | 重构 EDL 算法 | 改善幅度 |
|:---|:---:|:---:|:---:|
| **平均交并比 mIoU ↑** | 0.5634 | **0.5973** | **+3.39%** |
| **负对数似然 NLL ↓** | 0.7009 | **0.5998** | **−14.4%** |
| **Brier 分数 ↓** | 0.3624 | **0.3327** | **−8.2%** |
| **期望校准误差 ECE ↓** | 0.0954 | **0.0464** | **−51.4%** |
| **OOD 样本置信度（越低越好）** | 0.9013（过度自信） | 合理低置信度 | 风险可控 |

详细实验结果与可视化图表见 [`experiments/`](experiments/) 目录，完整分析报告见 [`experiments/Experiment_Report.md`](experiments/Experiment_Report.md)。

---

## 一键安装

### 环境要求

- Python >= 3.8
- CUDA >= 11.7（CPU 模式亦可运行实验脚本）
- Conda（推荐）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/183l/Object-Detection-and-Image-Segmentation.git
cd Object-Detection-and-Image-Segmentation

# 运行一键安装脚本（自动检测 CUDA 版本）
chmod +x install.sh
./install.sh

# 指定参数（可选）
./install.sh --env my-env --cuda 11.8
```

安装脚本将自动完成以下步骤：
1. 系统环境检测（Python 版本、CUDA 版本、Git）
2. 创建 Conda 虚拟环境（`road-reflector`）
3. 安装 PyTorch（根据 CUDA 版本自动选择）
4. 安装 MMSegmentation 生态（mmengine、mmcv、mmdet）
5. 以开发模式安装本项目
6. 安装评估与实验依赖
7. 自动验证 EDL 自定义模块注册
8. 运行单元测试（10/10 全部通过）

### 手动安装（Windows / 服务器）

**本地 Windows 开发环境（CUDA 11.8）：**

```bash
python -m venv openmmlab20
openmmlab20\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmengine==0.10.7
pip install -v -e .
python -c "import mmcv, mmengine, mmseg; print(mmcv.__version__); print(mmengine.__version__); print(mmseg.__version__)"
```

**推荐版本组合：**

| 组件 | 版本 |
|:---|:---|
| PyTorch | `2.0.1` |
| CUDA | `11.8` |
| MMCV | `2.1.0` |
| MMEngine | `0.10.7` |
| MMSegmentation | `1.2.2` |

**当前服务器实验环境：**

| 组件 | 版本 |
|:---|:---|
| Python | `3.8.20` |
| PyTorch | `1.10.2+cu102` |
| MMCV | `2.1.0` |
| MMEngine | `0.7.1` |
| MMSegmentation | `1.2.2` |
| GPU | Tesla V100 32GB |

---

## 快速开始

### 运行小样本对比实验

```bash
conda activate road-reflector
python experiments/comparison_experiment.py
```

实验将输出 Baseline vs EDL 的完整指标对比，并生成 6 幅可视化图表（保存至 `experiment_results/`）。

### 训练

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

### 评估

```bash
# 标准测试（mIoU）
python tools/test.py <config> <checkpoint>

# 统一校准指标评估（ECE / NLL / Brier Score）
# Baseline 模式
python eval_ece_unified.py <config> <checkpoint> --method baseline --out results_baseline.json

# EDL 模式
python eval_ece_unified.py <config> <checkpoint> --method edl --out results_edl.json

# 道路校准评估
python roadcalib.py --checkpoint <checkpoint>
```

### 运行单元测试

```bash
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

## 仓库结构

```text
.
├─ install.sh                           # 一键安装脚本 ✨
├─ eval_ece_unified.py                  # 统一校准评估脚本（已修复）✨
├─ roadcalib.py                         # 道路校准分析脚本（已修复）✨
│
├─ configs/
│   ├─ _base_/                          # 基础配置（模型、数据集、调度器）
│   └─ segformer/
│       ├─ segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py  # Baseline 配置
│       └─ segformer_mit-b2_edl_road_reflector.py              # EDL 专用配置 ✨
│
├─ mmseg/
│   └─ models/
│       ├─ decode_heads/
│       │   └─ evidential_head.py       # 证据解码头（已重构）✨
│       └─ losses/
│           └─ dirichlet_loss.py        # 狄利克雷损失函数（已重构）✨
│
├─ tools/
│   ├─ train.py                         # 官方训练入口
│   ├─ test.py                          # 官方测试入口
│   └─ calc_weights.py                  # 类别权重计算工具
│
├─ tests/
│   └─ test_edl_fixes.py               # 单元测试套件（10/10 通过）✨
│
├─ experiments/
│   ├─ comparison_experiment.py         # Baseline vs EDL 对比实验脚本 ✨
│   ├─ comparison_results.json          # 实验数值结果 ✨
│   ├─ comparison_results.png           # 实验可视化图表 ✨
│   └─ Experiment_Report.md            # 实验分析报告 ✨
│
├─ docs/
│   ├─ issues/
│   │   └─ ISSUES.md                   # 7 个已修复问题的规范记录 ✨
│   ├─ Secondary_Development_Report.md  # 二次开发报告（含大数据集方案）✨
│   ├─ Secondary_Development_Report.docx
│   ├─ Algorithm_Theory_Innovation_Report.md  # 算法理论创新报告 ✨
│   └─ Algorithm_Theory_Innovation_Report.docx
│
└─ Code_Analysis_Report.md             # 初始代码分析报告（7 个问题诊断）✨
```

> ✨ 标注为本次新增或修复的文件。

---

## 文档导航

| 文档 | 说明 |
|:---|:---|
| [二次开发报告](docs/Secondary_Development_Report.md) | 系统重构细节、大数据集处理方案、工程化建议 |
| [算法理论创新报告](docs/Algorithm_Theory_Innovation_Report.md) | EDL 数学推导、算法特点分析、创新成果总结 |
| [代码分析报告](Code_Analysis_Report.md) | 初始代码审查与 7 个问题的详细诊断 |
| [实验报告](experiments/Experiment_Report.md) | Baseline vs EDL 完整对比实验分析 |
| [问题记录](docs/issues/ISSUES.md) | 7 个已修复问题的规范化 Issue 文档 |

---

## 代码修复记录（v0.2.0）

本次提交修复了代码审查中发现的 7 个关键缺陷，详见 [`docs/issues/ISSUES.md`](docs/issues/ISSUES.md)：

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

## 技术路线

当前规划的完整方案分三步推进：

1. **`SegFormer` 语义分割 baseline**
   先完成可复现的像素级分割流程，验证"正常反光板 / 缺失区域 / 背景"的基本表达能力。

2. **`YOLO` 候选框检测**
   先在整帧视频中定位疑似异常区域，再送入分割模型做局部精细判断。

3. **`SAM 2/3` 标注辅助**
   用于视频样本的交互式追踪和快速标注，缩短前期数据制作周期。

---

## 当前阶段的待办事项

1. 收集并整理包含"缺失反光板"的巡检视频片段
2. 建立视频抽帧与数据清洗流程
3. 跑通 `SegFormer-B0 ~ B3` 的基线实验与速度评估
4. 定义反光板任务专用数据集与标注规范
5. 逐步接入 `YOLO + SegFormer` 二阶段方案
6. 在真实数据集上对比 Baseline 和 EDL 的 ECE / NLL / Brier Score 指标
7. 探索不确定性引导的主动学习（Active Learning）数据采集策略

---

## 开源协议

本项目基于 [Apache License 2.0](LICENSE) 开源，在 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 框架基础上进行二次开发。
