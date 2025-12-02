# 1. 基础配置引用 (去掉了找不到的 model 文件)
_base_ = [
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

# 2. 模型定义 (参照 vit-base 修改为 vit-small)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        # 核心修改：ViT-Small 的架构参数在 mmcls 中对应 'deit-small'
        # 参数细节：embed_dims=384, num_layers=12, num_heads=6
        arch='deit-small',
        img_size=224,
        patch_size=16,
        drop_rate=0.1, # 保持参考配置中的 dropout
        # 如果你想从头训练(不加载预训练权重)，设为 None
        # 如果想用随机初始化，也可以参考你发的配置保留 Kaiming 初始化(针对PatchEmbed层)
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=27,    # 你的数据集类别数
        in_channels=384,   # ViT-Small/DeiT-Small 的输出通道是 384 (Base是768)
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

# 3. 优化器 (沿用你之前的设置)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0) # Transformer 必备的梯度裁剪
)

# 4. 学习率策略
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100, eta_min=1e-6
)

# 5. 训练与运行时配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)