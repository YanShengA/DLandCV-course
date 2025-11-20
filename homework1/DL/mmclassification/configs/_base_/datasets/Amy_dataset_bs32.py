# configs/_base_/datasets/my_dataset_bs32.py

# --- 1. 数据集和数据预处理设置 ---
# 数据集类型：对于文件夹格式的数据集，使用 'CustomDataset'
dataset_type = 'CustomDataset'
# 数据集根目录
data_root = '/media/HDD0/wzl/mmcls/dataset_1/'

# 数据预处理器配置 ( normalization, one-hot, etc. )
data_preprocessor = dict(
    # 你的数据集有 27 个类别
    num_classes=27,
    # 图像归一化参数，通常直接使用 ImageNet 的预训练统计值
    # RGB 格式的均值和标准差
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # MMCls 默认读取的图像是 BGR 格式，这里转换为 RGB
    to_rgb=True,
)

# --- 2. 数据处理流水线 (Pipelines) ---
# 训练数据流水线，包含数据增强
train_pipeline = [
    # 从文件加载图像
    dict(type='LoadImageFromFile'),
    # 随机裁剪图像到 224x224
    dict(type='RandomResizedCrop', scale=224),
    # 50% 的概率水平翻转
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 将数据打包成 MMCls 需要的格式
    dict(type='PackInputs'),
]

# 测试/验证数据流水线，不包含随机数据增强
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # 将图像短边缩放到 256，长边等比缩放
    dict(type='ResizeEdge', scale=256, edge='short'),
    # 从图像中心裁剪出 224x224 的区域
    dict(type='CenterCrop', crop_size=224),
    # 打包数据
    dict(type='PackInputs'),
]


# --- 3. 数据加载器 (Dataloaders) ---
# 训练数据加载器
train_dataloader = dict(
    # 每个 GPU 的批量大小 (Batch Size)
    batch_size=32,
    # 加载数据的进程数，可以根据你的 CPU 和内存进行调整
    num_workers=4,
    # 数据集具体定义
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 指定训练数据的前缀路径
        data_prefix='train',
        pipeline=train_pipeline),
    # 采样器，保证每个 epoch 都会打乱数据
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 指定验证数据的前缀路径
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False), # 验证时不需要打乱
)

# --- 4. 评估器 (Evaluator) ---
# 验证评估器
val_evaluator = dict(type='Accuracy', topk=(1, 5)) # 计算 Top-1 和 Top-5 准确率

# 默认情况下，测试集和验证集使用相同的配置
test_dataloader = val_dataloader
test_evaluator = val_evaluator