import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# ================= 配置区域 =================
# 请务必修改为你的一张真实图片路径
image_path = '/media/HDD0/wzl/mmcls/dataset_1/train/grape leaf black rot/black_rot-grape_leaf.jpeg.jpg'  # <-- 修改这里
# ===========================================

def visualize_augmentations():
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片 {image_path}，请修改脚本中的 image_path！")
        return

    # 使用 PIL 读取图片
    img_pil = Image.open(image_path).convert('RGB')
    
    results = []
    titles = []

    # -------------------------------------------------
    # 1. Original (仅 Resize)
    # -------------------------------------------------
    transform_orig = transforms.Resize((224, 224))
    img_orig = transform_orig(img_pil)
    results.append(np.array(img_orig))
    titles.append("Original\n(Resize Only)")

    # -------------------------------------------------
    # 2. Baseline / Standard (RandomResizedCrop + Flip)
    # -------------------------------------------------
    transform_std = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # 随机裁剪
        transforms.RandomHorizontalFlip(p=1.0),             # 强制翻转
    ])
    img_std = transform_std(img_pil)
    results.append(np.array(img_std))
    titles.append("Standard Baseline\n(Crop & Flip)")

    # -------------------------------------------------
    # 3. AutoAugment (自动增强)
    # -------------------------------------------------
    # torchvision 自带 ImageNet 的 AutoAugment 策略
    transform_auto = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.Resize((224, 224))
    ])
    img_auto = transform_auto(img_pil)
    results.append(np.array(img_auto))
    titles.append("AutoAugment\n(Policy Learning)")

    # -------------------------------------------------
    # 4. Mixup (模拟两图混合)
    # -------------------------------------------------
    # 手动实现 Mixup 的视觉效果
    # 1. 准备第一张图 (Resize 后的原图)
    img_a = np.array(transform_orig(img_pil)).astype(np.float32)
    
    # 2. 准备第二张图 (制造一个干扰图：翻转+变色)
    # 真实训练中是取 Batch 里另一张图，这里为了展示效果，人工造一张
    img_b = np.array(img_std).astype(np.float32) # 用刚才的标准增强图作为图B
    img_b = img_b * 0.8 # 稍微变暗一点
    
    # 3. 混合
    lam = 0.6 # 混合比例
    img_mixup = lam * img_a + (1 - lam) * img_b
    img_mixup = img_mixup.astype(np.uint8)
    
    results.append(img_mixup)
    titles.append(f"Mixup\n(Blend $\lambda$={lam})")

    # -------------------------------------------------
    # 绘图
    # -------------------------------------------------
    plt.figure(figsize=(14, 4.5), dpi=300)
    
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(results[i])
        plt.title(titles[i], fontsize=13, fontweight='bold', pad=10)
        plt.axis('off')
    
    output_filename = 'E1_Augmentation_Showcase.png'
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"✅ 图片已生成: {output_filename}")
    plt.show()

if __name__ == '__main__':
    visualize_augmentations()