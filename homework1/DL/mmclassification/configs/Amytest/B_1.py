_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

# 1. 模型: 加载预训练权重
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='torchvision://resnet50',
        )
    ),
    head=dict(
        num_classes=27,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

# 2. 优化器: 修改为 AdamW, lr=0.001
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05)
)

# 3. 学习率策略: 余弦退火 (Cosine Annealing)
param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=100, # 训练 100 epoch
    by_epoch=True,
    begin=0,
    end=100,
    eta_min=1e-6 # 最小学习率
)

# 4. 训练时长
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

# 只保存最优权重
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)