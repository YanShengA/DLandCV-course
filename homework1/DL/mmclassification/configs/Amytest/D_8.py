_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(num_classes=27, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
)

# 1. 优化器: Adam
# 学习率通常设为 0.001 (比 SGD 小)
# Weight Decay 通常设小一点，如 0.0001
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001)
)

# 2. 策略: 配合自适应优化器，通常使用 Cosine 策略效果更好
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