# 1. 修改 _base_：删除了找不到的 model 文件引用
_base_ = [
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='deit-small',   # 架构：Small
        img_size=224,
        patch_size=16,
        
        # --- 修改部分开始 ---
        init_cfg=dict(
            type='Pretrained',
            # 这是一个有效的官方 DeiT-Small 权重链接
            checkpoint='https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth',
            prefix='backbone',
        ),
        # --- 修改部分结束 ---

        frozen_stages=12     # 保持你的冻结设置
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=27,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

# 3. 优化器配置 (保持不变)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0)
)

# 4. 学习率策略 (保持不变)
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100, eta_min=1e-6
)

# 5. 训练配置 (保持不变)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

# 6. Hooks (保持不变)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)