_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(num_classes=27, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
)

# 策略同 A-3: SGD, LR=0.02
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# 策略同 A-3: Step Decay (三段式)
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1
)

# 修改 Batch Size
# 目标总 BS=16, 双卡 -> 单卡设为 8
train_dataloader = dict(batch_size=8)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False) # 禁用自动缩放，严格控制变量

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)