import json
import os
import glob

# 路径配置 (使用你之前切分好的 unpaired 数据，因为那里 Label 和 Photo 是分开的)
ROOT_DIR = "/media/HDD0/wzl/mmcls/DLandCV-course/homework2/pytorch-CycleGAN-and-pix2pix-master/datasets/cityscapes_unpaired"
OUTPUT_JSON = "/media/HDD0/wzl/mmcls/DLandCV-course/homework2/crontrolnet/cityscapes_controlnet.jsonl"

def make_jsonl():
    # trainA 是 Label, trainB 是 Photo
    label_dir = os.path.join(ROOT_DIR, "trainA")
    photo_dir = os.path.join(ROOT_DIR, "trainB")
    
    # 获取所有图片
    # 假设文件名是一样的 (例如 1.jpg)
    files = glob.glob(os.path.join(label_dir, "*.jpg")) + glob.glob(os.path.join(label_dir, "*.png"))
    
    print(f"Found {len(files)} images. Generating JSONL...")
    
    with open(OUTPUT_JSON, 'w') as f:
        for label_path in files:
            filename = os.path.basename(label_path)
            photo_path = os.path.join(photo_dir, filename)
            
            if not os.path.exists(photo_path):
                print(f"Warning: Missing photo for {filename}")
                continue
                
            # 写入 JSONL 格式
            # image: 目标图 (Photo)
            # conditioning_image: 控制图 (Label)
            # text: 提示词
            entry = {
                "image": os.path.abspath(photo_path),
                "conditioning_image": os.path.abspath(label_path),
                "text": "high quality realistic city street view, driving scene"
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"Done! Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    make_jsonl()