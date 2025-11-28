# 1. 修改 _base_：移除不存在的 model 路径
_base_ = [
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

# 2. 完整定义模型 (ViT-Small)
model = dict(
    type='ImageClassifier', # 必须添加
    backbone=dict(
        type='VisionTransformer',
        # MMCls 中 'deit-small' 的架构参数 = ViT-Small (384通道, 12层, 6头)
        arch='deit-small', 
        img_size=224,
        patch_size=16,
        
        # --- 关键修改：使用有效的权重链接 ---
        init_cfg=dict(
            type='Pretrained',
            # 这是官方有效的 DeiT-Small (ImageNet-1k) 权重，不会报 404
            checkpoint='https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth',
            prefix='backbone',
        )
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead', # 必须指定 Head 类型
        num_classes=27,
        in_channels=384, # 对应 Small 的输出
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

# 3. 优化器配置 (保持你原有的设置)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0) 
)

# 4. 学习率策略 (保持你原有的设置)
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100, eta_min=1e-6
)

# 5. 运行配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto')
)