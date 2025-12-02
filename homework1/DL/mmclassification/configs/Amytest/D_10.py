_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(num_classes=27, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
)

# 1. 优化器: SGD, 关键修改 momentum=0
# 保持 lr=0.02 与 A-3 一致，以控制变量
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.0, weight_decay=0.0001)
)

# 2. 策略: 保持与 A-3 一致使用 Step Decay
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1
)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)