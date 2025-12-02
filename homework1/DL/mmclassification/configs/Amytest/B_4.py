_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth',
            # prefix='backbone'
        )
    ),
    head=dict(
        num_classes=27,
        in_channels=768,
        # 不重写 loss 部分，直接继承 _base_ 中的 LabelSmoothLoss
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