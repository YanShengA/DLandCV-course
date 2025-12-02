_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    head=dict(num_classes=27, in_channels=2048, loss=dict(type='CrossEntropyLoss', loss_weight=1.0))
)

# --- 修改部分：定义一个没有数据增强的 Pipeline ---
train_pipeline_no_aug = [
    dict(type='LoadImageFromFile'),
    # 将 RandomResizedCrop 替换为 Resize (强制缩放到 224x224)
    dict(type='Resize', scale=(224, 224)),
    # 删除了 RandomFlip
    dict(type='PackInputs'),
]

# 覆盖 train_dataloader 中的 pipeline
train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline_no_aug)
)

# 保持 A-3 的训练策略 (SGD + Step Decay)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)
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