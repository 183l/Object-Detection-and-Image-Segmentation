# 路边反光板缺失检测

基于 `MMSegmentation` 和 `SegFormer` 的道路视频路边反光板缺失检测研究仓库。

当前目标是先搭建一个可复现实验 baseline，跑通视频抽帧、语义分割训练、评估分析与后续工程扩展路径，并为后面的 `YOLO + SegFormer` 复合方案以及 `SAM 2/3` 辅助标注流程做准备。

## 当前状态

- 已完成基础环境搭建与 `SegFormer` 分割实验框架
- 已加入部分不确定性/校准相关实验代码
- 当前仓库 **不包含** 业务数据、视频样本、训练权重和 `work_dirs/` 输出
- 反光板专用数据集、类别体系和完整 pipeline 仍在持续整理中

这意味着：
现在公开的内容更接近“研究型 baseline 仓库”，而不是已经完成工程交付的产品仓库。

## 技术路线

计划中的完整方案分为三个层次：

1. `SegFormer` 语义分割
   作为基础模型，用于识别正常区域、疑似缺失区域和背景区域。

2. `YOLO` 候选区域检测
   在视频帧上先定位疑似异常区域，再调用分割模型做局部精细判断。

3. `SAM 2/3` 标注辅助
   用于视频样本的交互式追踪与快速标注，降低数据制作成本。

## 仓库里目前最相关的文件

- `configs/segformer/`
  SegFormer 相关训练配置
- `mmseg/models/decode_heads/evidential_head.py`
  证据式分割头实验
- `mmseg/models/losses/dirichlet_loss.py`
  Dirichlet 损失实验
- `eval_ece_unified.py`
  统一校准评估脚本
- `roadcalib.py`
  早期校准评估脚本

## 环境信息

当前已验证环境如下：

- Python `3.8.20`
- PyTorch `1.10.2+cu102`
- TorchVision `0.11.3+cu102`
- MMCV `2.1.0`
- MMEngine `0.7.1`
- MMSegmentation `1.2.2`
- GPU `Tesla V100 32GB`

如果你准备在其他机器上复现，建议优先参考 `MMSegmentation` 官方安装说明，再根据本仓库的实验配置微调版本。

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd mmsegmentation
```

### 2. 创建环境

```bash
conda create -n segformer python=3.8 -y
conda activate segformer
pip install -U pip
pip install -v -e .
```

如果需要严格对齐当前实验环境，请额外核对 `torch`、`torchvision`、`mmcv` 和 CUDA 版本。

### 3. 训练 baseline

```bash
python tools/train.py configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py
```

说明：
当前仓库里现成的配置仍然主要来自 `SegFormer` / `Cityscapes` 实验流程。后续切换到反光板任务时，建议新增独立的数据集配置文件，而不是继续直接改通用配置。

### 4. 评估

```bash
python tools/test.py <config> <checkpoint>
python eval_ece_unified.py <config> <checkpoint> --out results.json
```

## 建议的数据组织方式

仓库不直接包含数据，但建议按下面的结构整理，方便后续训练脚本统一接入：

```text
data/
  road_reflector/
    videos/
      raw/
      clips/
    frames/
      train/
      val/
      test/
    annotations/
      train/
      val/
      test/
    splits/
      train.txt
      val.txt
      test.txt
```

## 接下来准备做的事

1. 收集并整理包含“缺失反光板”的巡检视频片段
2. 建立视频抽帧与数据清洗流程
3. 跑通 `SegFormer-B0 ~ B3` 的基线实验与速度评估
4. 定义反光板任务专用数据集与类别标注规范
5. 逐步接入 `YOLO + SegFormer` 的二阶段方案

## GitHub 公开说明

建议公开上传：

- 代码
- 配置
- 训练/评估脚本
- 项目说明文档

建议不要上传：

- 数据集
- 视频原文件
- 权重文件 `*.pth`
- `work_dirs/`
- 临时输出、缓存和服务器私有信息

更详细的项目说明见 [docs/road_reflector_project.md](docs/road_reflector_project.md)。

## 致谢

本项目基于 [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 二次开发，在此感谢原始项目提供的分割框架与开源生态支持。
