import os
import argparse
from PIL import Image
from tqdm import tqdm

def convert_dataset(source_dir, dest_dir, phase):
    print(f"Processing '{phase}' set...")
    source_path = os.path.join(source_dir, phase)
    if not os.path.exists(source_path):
        print(f"Source directory not found: {source_path}")
        return

    # 创建目标文件夹
    images_path = os.path.join(dest_dir, phase, 'images')
    labels_path = os.path.join(dest_dir, phase, 'segmentations')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # 获取所有拼接图像
    ab_images = [f for f in os.listdir(source_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for fname in tqdm(ab_images, desc=f"Converting {phase}"):
        ab_path = os.path.join(source_path, fname)
        ab_image = Image.open(ab_path).convert("RGB")
        
        w, h = ab_image.size
        w2 = w // 2
        
        # A (照片) 在左, B (标签) 在右
        photo_pil = ab_image.crop((0, 0, w2, h))
        label_pil = ab_image.crop((w2, 0, w, h))
        
        # 保存到新目录
        photo_pil.save(os.path.join(images_path, fname))
        label_pil.save(os.path.join(labels_path, fname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to the original Cityscapes dataset (e.g., ../pytorch-CycleGAN-and-pix2pix/datasets/cityscapes)')
    parser.add_argument('--dest', type=str, default='./data/cityscapes', help='Path to the destination directory for the new format')
    args = parser.parse_args()
    
    os.makedirs(args.dest, exist_ok=True)
    
    convert_dataset(args.source, args.dest, 'train')
    convert_dataset(args.source, args.dest, 'val')
    
    print("Dataset conversion finished!")