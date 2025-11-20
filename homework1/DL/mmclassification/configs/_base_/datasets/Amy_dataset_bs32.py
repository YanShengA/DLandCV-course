# configs/_base_/datasets/my_dataset_bs32.py

# 1. 数据集设置 (Dataset settings)
# CustomDataset 适用于你自己的、遵循 'class/image.jpg' 目录结构的数据集
dataset_type = 'CustomDataset'
# 你的数据集根目录
data_root = '/media/HDD0/wzl/mmcls/dataset_1/'

# 2. 数据预处理器配置 (Data preprocessor)
# 这是新版 mmcls/MMEngine 的特性，用于在模型前处理数据，如归一化
data_preprocessor = dict(
    num_classes=27,  # <-- 关键修改：将类别数改为 27
    # RGB 格式的归一化参数
    # 建议保持 ImageNet 的均值和标准差，因为你使用的 ResNet 预训练模型是在 ImageNet 上训练的
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # 将图像从 BGR 格式转换为 RGB 格式
    to_rgb=True,
)

# 3. 训练数据处理流水线 (Training pipeline)
train_pipeline = [
    # 从文件加载图像
    dict(type='LoadImageFromFile'),
    # 随机裁剪图像到 224x224
    dict(type='RandomResizedCrop', scale=224),
    # 50% 的概率水平翻转
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 打包数据，这是新版框架必须的步骤
    dict(type='PackInputs'),
]

# 4. 测试数据处理流水线 (Test pipeline)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # 将图像短边缩放到 256，保持长宽比
    dict(type='ResizeEdge', scale=256, edge='short'),
    # 从中心裁剪出 224x224 的区域
    dict(type='CenterCrop', crop_size=224),
    # 打包数据
    dict(type='PackInputs'),
]

# 5. 训练数据加载器 (Train dataloader)
train_dataloader = dict(
    batch_size=32,  # 每个 GPU 的 batch size
    num_workers=4,  # 加载数据的进程数，可根据你的机器 CPU 核心数调整
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 对于 CustomDataset, 我们不使用 'split' 字段，而是直接用 'data_prefix' 指向子目录
        data_prefix='train',
        # with_label=True 是默认的，确保加载标签
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True), # 默认采样器，训练时打乱数据
)

# 6. 验证数据加载器 (Validation dataloader)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='val', # 指向验证集目录
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False), # 验证时不打乱数据
)
# 验证评估器，计算 top-1 和 top-5 准确率
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# 7. 测试数据加载器与评估器 (Test dataloader and evaluator)
# 通常情况下，测试集就是验证集，所以直接复用上面的配置
test_dataloader = val_dataloader
test_evaluator = val_evaluator