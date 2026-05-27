# Object Detection and Image Segmentation

面向道路巡检场景的视觉研究仓库，当前聚焦于“路边反光板缺失检测”任务。

这个仓库基于 [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 二次开发，现阶段公开的内容以 `SegFormer` 语义分割 baseline 为主，同时为后续接入 `YOLO + SegFormer` 复合检测方案和 `SAM 2/3` 辅助标注流程做准备。

## 1. 项目目标

目标是从道路巡检视频中自动识别路边反光板是否存在缺失、损坏或异常区域，并逐步支撑两类落地方向：

- 实时巡检：面向车载或边缘设备的视频流分析
- 离线分析：面向批量巡检视频的自动检测与告警

## 2. 当前公开版本包含什么

当前公开版本更接近“研究型 baseline 仓库”，主要包含：

- 基于 `MMSegmentation` 的 `SegFormer` 训练与评估环境
- 若干 `SegFormer` 配置文件与实验代码
- 不确定性 / 校准相关实验脚本
- 项目文档、环境说明和数据组织建议

当前不包含：

- 原始巡检视频
- 业务数据集与标注文件
- 训练权重 `*.pth`
- `work_dirs/` 下的实验输出

## 3. 技术路线

当前规划的完整方案分三步推进：

1. `SegFormer` 语义分割 baseline
   先完成可复现的像素级分割流程，验证“正常反光板 / 缺失区域 / 背景”的基本表达能力。

2. `YOLO` 候选框检测
   先在整帧视频中定位疑似异常区域，再送入分割模型做局部精细判断。

3. `SAM 2/3` 标注辅助
   用于视频样本的交互式追踪和快速标注，缩短前期数据制作周期。

## 4. 当前仓库里最相关的文件

- `configs/segformer/`
  SegFormer 相关训练配置
- `eval_ece_unified.py`
  统一校准指标评估脚本
- `roadcalib.py`
  早期校准分析脚本
- `tools/calc_weights.py`
  类别权重计算脚本

说明：
当前公开版本仍以分割 baseline 为主，目标检测部分还处在规划与后续接入阶段。

## 5. 环境说明

### 5.1 本地 Windows 开发环境示例

这是我之前本地整理过的一套可用安装流程，适合作为 Windows 侧开发参考：

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

- PyTorch `2.0.1`
- CUDA `11.8`
- MMCV `2.1.0`
- MMEngine `0.10.7`
- MMSegmentation `1.2.2`

### 5.2 当前服务器实验环境

当前项目实际整理与实验时使用过的服务器环境如下：

- Python `3.8.20`
- PyTorch `1.10.2+cu102`
- TorchVision `0.11.3+cu102`
- MMCV `2.1.0`
- MMEngine `0.7.1`
- MMSegmentation `1.2.2`
- GPU `Tesla V100 32GB`

说明：
cuda版本与torch版本需根据电脑真实情况做调整，但不影响正常代码的运行

## 6. 快速开始

### 6.1 克隆仓库

```bash
git clone https://github.com/183l/Object-Detection-and-Image-Segmentation.git
cd Object-Detection-and-Image-Segmentation
```

### 6.2 安装依赖

如果你使用 `conda`，推荐按下面的顺序安装：

```bash
conda create -n segformer python=3.8 -y
conda activate segformer
pip install -U pip setuptools wheel
pip install mmengine==0.10.7
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -v -e .
```

说明：

- `pip install -v -e .` 会安装仓库本身和 `requirements/runtime.txt` 中的运行时依赖
- `mmcv` 和 `mmengine` 需要你提前装好，当前仓库不会仅靠 `pip install -v -e .` 自动补齐这两项
- 上面这组命令对应的是“Windows 本地开发环境示例”，也就是 `PyTorch 2.0.1 + CUDA 11.8 + MMCV 2.1.0 + MMEngine 0.10.7`
- 如果你使用的是服务器环境，就要改成与你服务器一致的 `torch / cuda / mmengine / mmcv` 版本组合

如果你使用 Windows `venv`，可直接参考上面的“本地 Windows 开发环境示例”。

### 6.3 训练 baseline

官方训练入口保持不变，但下面这条命令只是当前仓库里现成可跑的 `Cityscapes + SegFormer-B2` 基线示例：

```bash
python tools/train.py configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py --work-dir work_dirs/segformer_mit-b2_cityscapes
```

### 6.4 测试与评估

```bash
python tools/test.py <config> <checkpoint>
python eval_ece_unified.py <config> <checkpoint> --out results.json
```

## 7. 官方脚本与自定义脚本

当前仓库使用方式建议这样理解：

- 官方通用入口：
  `tools/train.py`、`tools/test.py`
- 当前自定义实验脚本：
  `eval_ece_unified.py`、`roadcalib.py`

也就是说，训练和测试主流程仍然沿用官方入口，自定义部分主要集中在评估分析和实验配置上。


## 8. 仓库结构建议理解

当前可以先按下面的方式理解这个仓库：

```text
.
├─ configs/segformer/          # SegFormer 训练配置
├─ mmseg/                      # MMSegmentation 主体代码
├─ tools/                      # 官方训练脚本与部分工具脚本
├─ eval_ece_unified.py         # 自定义评估脚本
├─ roadcalib.py                # 自定义分析脚本
└─ README.md
```

## 9. 当前阶段的待办事项

1. 收集并整理包含“缺失反光板”的巡检视频片段
2. 建立视频抽帧与数据清洗流程
3. 跑通 `SegFormer-B0 ~ B3` 的基线实验与速度评估
4. 定义反光板任务专用数据集与标注规范
5. 逐步接入 `YOLO + SegFormer` 二阶段方案
