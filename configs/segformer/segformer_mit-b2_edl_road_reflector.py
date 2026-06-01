# SegFormer-B2 + EvidentialHead + DirichletLoss 专用配置文件
#
# 修复 Issue #1：将 EvidentialHead 和 DirichletLoss 正式集成到训练配置中。
# 修复 Issue #5：统一使用 max_iters 变量管理训练步数，确保 PolyLR 终点与
#               训练步数严格一致，避免学习率提前归零。
#
# 用法：
#   python tools/train.py configs/segformer/segformer_mit-b2_edl_road_reflector.py

_base_ = ['./segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

# SegFormer-B2 预训练权重
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

# 由 tools/calc_weights.py 计算得到的归一化类别权重（Cityscapes 19 类）
# 请根据实际数据集重新计算后替换此处的数值
class_weight = [
    0.8524, 1.1201, 0.9654, 1.0521, 1.0112, 0.9845,
    1.1524, 1.0854, 0.9214, 0.9745, 0.8954, 1.0214,
    1.1124, 0.8854, 1.2145, 1.1854, 1.3214, 1.2541, 1.0124
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(
        # 修复 Issue #1：启用自定义证据解码头
        type='EvidentialHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            # 修复 Issue #1：启用自定义狄利克雷损失
            type='DirichletLoss',
            loss_weight=1.0,
            class_weight=class_weight,
            ignore_index=255,
            anneal_iters=3000,   # KL 退火迭代步数（建议为 max_iters 的 30%）
            kl_weight=0.001,     # KL 正则化最终权重
            use_standard_kl=False  # 设为 True 可启用精确 KL 散度（OOD 效果更好）
        )
    )
)

# 修复 Issue #5：统一使用 max_iters 变量，确保 PolyLR 终点与训练步数一致
max_iters = 10000

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=1000)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=max_iters,  # 与 max_iters 严格对齐，防止学习率提前归零
        by_epoch=False)
]

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
