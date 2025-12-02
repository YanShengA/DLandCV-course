_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        # 1. 加载预训练权重
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        # 2. 冻结 ResNet 的全部 4 个 stage
        frozen_stages=4
    ),
    head=dict(
        num_classes=27,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05)
)

param_scheduler = dict(
    type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100, eta_min=1e-6
)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)