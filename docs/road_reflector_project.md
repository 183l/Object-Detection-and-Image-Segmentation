# 路边反光板缺失检测项目说明

## 1. 项目目标

本项目面向道路巡检视频场景，目标是自动识别视频画面中的路边反光板是否存在缺失、损坏或异常区域，并为后续两类工程落地方案提供基础：

- 实时巡检：面向车载或边缘设备的视频流检测
- 离线分析：面向批量巡检视频的自动分析与告警

## 2. 当前技术路线

当前规划采用分阶段复合方案：

1. `SegFormer` 作为基础语义分割模型
   用于学习“正常反光板区域 / 缺失区域 / 背景”的像素级表达，先建立稳定 baseline。

2. `YOLO` 作为候选框检测模块
   先从整帧视频中定位疑似破损或缺失区域，再把局部区域送入 SegFormer 做精细化分割。

3. `SAM 2/3` 作为数据标注辅助工具
   用于视频样本的快速交互式追踪和粗标，缩短前期数据集制作周期。

## 3. 当前仓库实际公开内容

这个仓库目前公开的是“基础训练与评估框架”，重点包括：

- 基于 `MMSegmentation` 的 `SegFormer` 训练环境
- 若干 `SegFormer` 配置与实验脚本
- 不确定性/校准相关实验代码
- 基线训练、测试、评估所需的代码改动

当前 **不包含**：

- 原始巡检视频
- 抽帧后的业务数据集
- 标注文件
- 训练权重
- `work_dirs/` 下的实验输出

原因主要是数据隐私、文件体积和实验过程仍在持续迭代。

## 4. 目前和项目最相关的代码位置

- `configs/segformer/`
  SegFormer 训练配置
- `mmseg/models/decode_heads/evidential_head.py`
  证据式分割头实验
- `mmseg/models/losses/dirichlet_loss.py`
  Dirichlet 损失实验
- `eval_ece_unified.py`
  统一校准指标评估脚本
- `roadcalib.py`
  早期校准评估脚本

说明：
这些代码更偏“SegFormer baseline + calibration 实验”阶段，尚未完全切换到“反光板专用数据集与任务定义”。

## 5. 建议的数据目录约定

公开仓库里不放数据，但建议保留一致的数据组织方式，方便后续复现：

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

如果后续同时维护检测和分割两套标注，也可以扩展为：

```text
data/
  road_reflector/
    detection/
    segmentation/
```

## 6. 推荐的基线推进顺序

1. 视频抽帧
   先从巡检视频中抽取关键帧，建立最小可训练样本集。

2. 初始标注
   先做二分类或三分类分割：
   `background / reflector / missing_or_damaged`

3. 跑通 SegFormer baseline
   从 `SegFormer-B0` 开始，优先验证训练是否稳定、显存是否可控、评估流程是否完整。

4. 增加候选区域检测
   在整帧图像上加入 YOLO，减少分割模型的无效计算区域。

5. 引入时间维度
   后续再考虑追踪、时序平滑、多帧一致性判断和告警逻辑。

## 7. 公开仓库建议保留的内容

建议上传到 GitHub 的内容：

- 代码本体
- 自定义模型改动
- 训练与评估脚本
- 配置文件
- `README.md`
- 任务说明文档

建议不要上传的内容：

- 业务数据
- 视频原文件
- 权重文件 `*.pth`
- `work_dirs/`
- 临时输出图和 notebook 检查点
- 服务器路径、账号信息、临时日志

## 8. 后续建议

如果下一步要把仓库进一步做成“更像项目而不是实验目录”，建议优先补这几项：

1. 新建反光板专用数据集配置
2. 新增视频抽帧脚本
3. 新增数据集格式说明与标注规范
4. 明确类别定义与评估指标
5. 为 `SegFormer-B0 ~ B3` 补一组统一 benchmark

## 9. 致谢

本项目基于 [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 进行二次开发，在此对原始项目和社区工作表示感谢。
