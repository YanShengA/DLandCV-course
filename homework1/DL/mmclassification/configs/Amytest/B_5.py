_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_large_imagenet.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth',
            prefix='backbone', 
        )
    ),
    head=dict(
        # 这里不需要改 type，默认继承 StackedLinearClsHead
        num_classes=27,
        # MobileNetV3 默认 head 含有 mid_channels=[1280]，这里会自动继承，无需修改
        # in_channels=960 也会自动继承
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
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