_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(num_classes=27, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
)

# 1. 优化器: RMSprop
# alpha=0.99, eps=1e-8 是标准参数
optim_wrapper = dict(
    optimizer=dict(type='RMSprop', lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0.0001, momentum=0)
)

# 2. 策略: Cosine
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