_base_ = [
    '../_base_/models/resnet18.py',  # 1. 修改模型基座文件 (或者 resnet34.py)
    '../_base_/datasets/Amy_dataset_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet18', # 2. 修改预训练权重链接 (或者 resnet34)
        )
    ),
    head=dict(
        num_classes=27,
        in_channels=512,     # 3. !!! 注意：这里必须改为 512 !!!
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

# --- 2. 优化器设置 (双卡适配) ---
# 单卡BS=32 -> 双卡BS=64
# 推荐学习率 lr = 0.02
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# --- 3. 学习率策略 ---
# 在第 30, 60, 90 epoch 衰减学习率
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# --- 4. 训练循环设置 ---
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# 禁用自动缩放，因为我们已经手动设置了 lr
auto_scale_lr = dict(enable=False)

# --- 5. 钩子设置 (Hooks) - 只保存最优权重 ---
default_hooks = dict(
    # 日志打印：每 10 个 iter 打印一次 (因为你数据少，iter 少)
    logger=dict(type='LoggerHook', interval=10),
    
    # 权重保存：
    # interval=1: 每个 epoch 结束后都进行评估判断
    # save_best='auto': 自动根据评估指标(准确率)保存最好的模型
    # max_keep_ckpts=1: 只保留 1 个文件，旧的会被删除
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        max_keep_ckpts=1, 
        save_best='auto'
    )
)