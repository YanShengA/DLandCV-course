_base_ = [
    # 修改点：将原来的 swin_tiny_224.py 改为 tiny_224.py
    '../_base_/models/swin_transformer/tiny_224.py', 
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth',
            prefix='backbone'
        )
    ),
    head=dict(
        num_classes=27,   # 只保留你需要修改的类别数
        in_channels=768,
        # 【重点】这里直接把 loss=dict(...) 这一整行删掉
        # MMPretrain 会自动使用 base 中的 LabelSmoothLoss
    )
)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    clip_grad=dict(max_norm=5.0) # Swin 推荐的梯度裁剪
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