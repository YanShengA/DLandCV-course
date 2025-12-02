import argparse
from ultralytics import YOLO

def train_model(model_name, data_path, epochs, batch_size, device_ids):
    print(f"🚀 开始训练 | 模型: {model_name} | Batch: {batch_size} | 设备: {device_ids}")
    
    # 1. 加载模型
    model = YOLO(model_name)

    # 2. 设置保存路径
    project_name = "classification_comparison"
    # 在运行名称中标记 batch 大小，方便后续分析
    run_name = f"exp_{model_name.replace('.pt', '')}_bs{batch_size}"

    # 3. 开始训练
    # 注意: 在多卡模式下，batch 是总批次大小
    results = model.train(
    data=data_path,
    epochs=100,
    
    # --- 关键修改 1: 提高分辨率 ---
    # 植物病害需要看细节，224可能看不清斑点。
    # 尝试 320, 416, 甚至 640 (注意显存占用，imgsz变大，batch要减小)
    imgsz=640,  
    label_smoothing=0.1,
    
    batch=batch_size, # 如果改了大图，记得调小 batch，比如 32 或 16
    device=device_ids,
    project=project_name,
    name=run_name,
    
    # --- 关键修改 2: 禁用/削弱破坏颜色的增强 ---
    hsv_h=0.0,      # ★ 彻底关闭色相变化 (防止黄叶变绿叶)
    hsv_s=0.1,      # 饱和度微调 (允许轻微变化)
    hsv_v=0.1,      # 亮度微调 (允许光照变化)
    
    # --- 关键修改 3: 禁用遮挡类增强 ---
    mixup=0.0,      # ★ 关闭 Mixup
    erasing=0.0,    # ★ 关闭随机擦除 (防止挡住病灶)
    dropout=0.0,    # 分类头Dropout可以保留，或者设为0测试一下
    
    # --- 保留几何增强 (这些是安全的) ---
    fliplr=0.5,     # 水平翻转 (叶片左右翻转没问题)
    flipud=0.5,     # 垂直翻转 (叶片上下翻转也没问题)
    scale=0.5,      # 缩放 (模拟远近拍摄)
    degrees=15.0,   # 旋转 (模拟拍摄角度)
    
    # --- 优化器 ---
    # 小数据集微调，学习率要小
    lr0=0.0001,     
    optimizer='AdamW'
    )
    print(f"✅ 模型 {model_name} 训练完成！")

if __name__ == '__main__':
    # ★★★ 多卡训练必须在 if __name__ == '__main__': 下运行，否则会报错 ★★★
    parser = argparse.ArgumentParser(description='YOLO11 Multi-GPU Classification')
    
    parser.add_argument('--model', type=str, required=True, help='模型权重, e.g., yolo11n-cls.pt')
    parser.add_argument('--data', type=str, default='/media/HDD0/wzl/mmcls/dataset_1', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    
    # 新增 Batch Size 参数
    parser.add_argument('--batch', type=int, default=64, help='总Batch Size (所有显卡之和)')
    
    # 修改 device 参数说明
    parser.add_argument('--device', type=str, default='0,1', help='显卡ID, 多卡用逗号分隔, e.g., "0,1"')

    args = parser.parse_args()

    train_model(args.model, args.data, args.epochs, args.batch, args.device)