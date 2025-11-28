_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(num_classes=27, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
)

# 1. 优化器: SGD, LR=0.002
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
)

# 2. 学习率策略: 恒定 (Constant)
# 使用 ConstantLR 确保全程不变
param_scheduler = dict(type='ConstantLR', by_epoch=True)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)