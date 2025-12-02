import argparse
import time
import torch
import warnings

# -----------------------------------------------------------
# 1. 自动适配不同版本的 fuse_conv_bn
# -----------------------------------------------------------
fuse_conv_bn = None
# 尝试路径 1: MMCV (最常见的标准位置, 适用于 mmcv>=2.0.0)
try:
    from mmcv.cnn.utils import fuse_conv_bn
    print("[Info] Imported fuse_conv_bn from mmcv.cnn.utils")
except ImportError:
    pass

# 尝试路径 2: MMEngine Model Utils (部分版本)
if fuse_conv_bn is None:
    try:
        from mmengine.model.utils import fuse_conv_bn
        print("[Info] Imported fuse_conv_bn from mmengine.model.utils")
    except ImportError:
        pass

# 尝试路径 3: MMEngine Model (旧尝试)
if fuse_conv_bn is None:
    try:
        from mmengine.model import fuse_conv_bn
        print("[Info] Imported fuse_conv_bn from mmengine.model")
    except ImportError:
        pass

# 如果都找不到，定义一个空函数防止报错，并打印警告
if fuse_conv_bn is None:
    def fuse_conv_bn(model):
        warnings.warn("Could not find 'fuse_conv_bn'. Testing speed without Conv-BN fusion.")
        return model

# -----------------------------------------------------------
# 2. 导入其他必要库
# -----------------------------------------------------------
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS # 确保使用的是 mmpretrain

def parse_args():
    parser = argparse.ArgumentParser(description='Measure inference speed')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--shape', type=int, nargs='+', default=[224, 224], help='input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='test batch size')
    parser.add_argument('--iterations', type=int, default=200, help='number of iterations to test')
    parser.add_argument('--device', default='cuda:0', help='device used for testing')
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载配置
    print(f"Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)

    # 构建模型
    print("Building model...")
    model = MODELS.build(cfg.model)
    
    # 加载权重
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 移动到 GPU
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # 融合 Conv-BN
    print("Attempting to fuse Conv and BN layers...")
    try:
        # 有些特殊的模型结构可能不支持 fuse，加上 try-except 防止脚本中断
        model = fuse_conv_bn(model)
        print("Fusion complete (or skipped if not supported).")
    except Exception as e:
        print(f"Warning: Fusion failed: {e}. Proceeding without fusion.")

    # 构造输入
    input_shape = (args.batch_size, 3, args.shape[0], args.shape[1])
    input_data = torch.randn(input_shape).to(device)
    
    print(f"Testing with Input Shape: {input_shape}")

    # 预热
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            model(input_data, mode='tensor') 

    # 测速
    print(f"Start testing speed ({args.iterations} iterations)...")
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(args.iterations):
            model(input_data, mode='tensor')
    
    torch.cuda.synchronize()
    end_time = time.time()

    # 计算结果
    total_time = end_time - start_time
    avg_latency = (total_time / args.iterations) * 1000 # ms
    fps = (args.iterations * args.batch_size) / total_time

    print("\n" + "="*40)
    print(f"Model: {args.config}")
    print(f"Total Time: {total_time:.4f} s")
    print(f"Latency:    {avg_latency:.4f} ms/batch")
    print(f"FPS:        {fps:.2f} images/s")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()