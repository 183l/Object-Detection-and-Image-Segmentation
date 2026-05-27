_base_ = ['./segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512])
)

# ====================================================================
# 以下是为了跑 20K 快速实验新增的配置
# ====================================================================

# 1. 拦截并修改训练循环：总里程设为 20000，每 2000 步验证一次 (测那500张图)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)

# 2. 拦截并修改学习率调度器 (极其重要！)
# 默认的 PolyLR 是按 160K 的进度慢慢降温的。
# 既然我们现在把终点改成了 20K，就必须让学习率在 20K 的时候顺利降到 0，否则模型收敛不了。
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=10000,  # <--- 终点设为 20K
        by_epoch=False,
    )
]