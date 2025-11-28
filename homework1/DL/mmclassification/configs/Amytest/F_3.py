# 继承基础配置 (含原有的 BatchSize=32 和 LR=0.02)
_base_ = ['./resnet50.py']

# =======================================================
# 👇 只需要修改这唯一的一个数字 👇
# =======================================================
experiment_size = 320   # 例如改这里为 384, 448, 512, 640...
# =======================================================


# --- 以下代码会自动运行，无需修改 ---

# 1. 自动计算 ResizeEdge 的比例
# 按照 ImageNet 惯例，Resize 稍微比 Crop 大一点 (系数约 1.14)
# 这样能保证 CenterCrop 时边缘信息更丰富
resize_short_edge = int(experiment_size * (256 / 224))

# 2. 动态构建训练 Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # 这里的 scale 自动引用上面的 experiment_size
    dict(type='RandomResizedCrop', scale=experiment_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

# 3. 动态构建测试 Pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # 这里的 scale 自动引用计算好的 resize_short_edge
    dict(type='ResizeEdge', scale=resize_short_edge, edge='short'),
    # 这里的 crop_size 自动引用 experiment_size
    dict(type='CenterCrop', crop_size=experiment_size),
    dict(type='PackInputs'),
]

# 4. 覆盖数据加载器中的 Pipeline
# 注意：这里只覆盖了 dataset.pipeline，
# batch_size 和 num_workers 会直接继承 _base_ 里的设置 (即 BS=32)
train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline)
)

val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline)
)

test_dataloader = val_dataloader