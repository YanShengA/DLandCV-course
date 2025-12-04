import os
from PIL import Image
from tqdm import tqdm
import shutil

# 配置路径
source_dir = "./datasets/cityscapes"
target_root = "./datasets/cityscapes_unpaired"

def split_folder(phase):
    # 源文件夹 (train 或 val)
    src_path = os.path.join(source_dir, phase)
    if not os.path.exists(src_path):
        return

    # 目标文件夹: CycleGAN unaligned 模式要求必须叫 trainA 和 trainB
    # 我们定义:
    # A = Label (输入域) -> 来自原图右半边
    # B = Photo (输出域) -> 来自原图左半边
    # 这样训练方向就是 AtoB (Label -> Photo)
    
    # 也就是: datasets/cityscapes_unpaired/trainA (放Label)
    #        datasets/cityscapes_unpaired/trainB (放Photo)
    
    dest_A = os.path.join(target_root, f"{phase}A")
    dest_B = os.path.join(target_root, f"{phase}B")
    
    os.makedirs(dest_A, exist_ok=True)
    os.makedirs(dest_B, exist_ok=True)
    
    print(f"Processing {phase} set...")
    files = os.listdir(src_path)
    
    for filename in tqdm(files):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 读取拼接图
            img = Image.open(os.path.join(src_path, filename))
            w, h = img.size
            
            # Cityscapes Pix2Pix 格式: 左图 Photo, 右图 Label
            # 我们要 Label -> Photo
            
            # 切割
            photo = img.crop((0, 0, w//2, h))      # 左边
            label = img.crop((w//2, 0, w, h))      # 右边
            
            # 保存到各自的文件夹
            # 物理隔离：A 文件夹里只有 Label，B 文件夹里只有 Photo
            label.save(os.path.join(dest_A, filename))
            photo.save(os.path.join(dest_B, filename))

if __name__ == "__main__":
    split_folder("train")
    split_folder("val")
    print("数据集切割完成！保存在 ./datasets/cityscapes_unpaired")