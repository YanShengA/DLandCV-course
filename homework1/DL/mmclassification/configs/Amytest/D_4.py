_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(num_classes=27, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
)

# 保持和 B-1 一致使用 AdamW
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05)
)

# 组合策略：Linear Warmup + Cosine Annealing
param_scheduler = [
    # 1. 预热阶段：前 5 个 epoch 从很小的学习率线性增加到 0.001
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    # 2. 余弦退火阶段：从第 5 epoch 到第 100 epoch
    dict(
        type='CosineAnnealingLR',
        T_max=95, # 100 - 5
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6
    )
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)