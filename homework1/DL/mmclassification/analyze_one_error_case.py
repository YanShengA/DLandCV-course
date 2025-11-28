import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

# 检查 mmpretrain
try:
    from mmpretrain.apis import init_model
except ImportError:
    print("❌ 错误: 请安装 mmpretrain (pip install mmpretrain)")
    exit()

# ================= 1. 配置区域 =================

# 1.1 模型路径
config_path = '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/configs/Amytest/D_10.py'
checkpoint_path = '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/work_dirs/D_10/best_accuracy_top1_epoch_29.pth'

# 1.2 图片路径
image_path = '/media/HDD0/wzl/mmcls/dataset_1/val/Potato leaf late blight/B2750109-Late_blight_on_a_potato_plant-SPL.jpg'

# 1.3 目标层 (ResNet-50)
target_layer_name = 'backbone.layer4[-1]'

# 1.4 保存路径
save_dir = "./error_ana"
save_name = "heatmap_original_size.png"

# =========================================================

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        feats = self.model.extract_feat(x)
        if hasattr(self.model.head, 'forward'):
            try:
                logits = self.model.head(feats)
            except:
                if isinstance(feats, tuple): feat = feats[-1]
                else: feat = feats
                logits = self.model.head.fc(self.model.head.pre_logits(feat))
        else:
             logits = self.model.head.fc(feats)
        if isinstance(logits, dict):
            return logits['pred_scores']
        return logits

def get_target_layer(model, layer_str):
    try:
        parts = layer_str.replace('[-1]', '.__last__').split('.')
        current = model
        for part in parts:
            if part == '__last__': current = current[-1]
            elif '[' in part:
                name, idx = part[:-1].split('[')
                current = getattr(current, name)[int(idx)]
            else:
                current = getattr(current, part)
        return [current]
    except Exception as e:
        print(f"❌ 找不到层 {layer_str}: {e}")
        return []

def main():
    # 1. 加载模型
    print(f"🚀 Loading model...")
    try:
        model = init_model(config_path, checkpoint_path, device='cuda:0')
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 读取原图 (保持原始尺寸)
    img_origin = cv2.imread(image_path)
    if img_origin is None:
        print(f"❌ 图片读取失败: {image_path}")
        return
    
    # 获取原图尺寸 (高度, 宽度)
    h_origin, w_origin = img_origin.shape[:2]
    
    # 3. 制作模型输入 (必须缩放到 224x224)
    img_resized = cv2.resize(img_origin, (224, 224))
    rgb_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    rgb_resized_float = np.float32(rgb_resized) / 255
    input_tensor = preprocess_image(rgb_resized_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 4. 推理
    wrapped_model = ModelWrapper(model)
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    with torch.no_grad():
        logits = wrapped_model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    target_category = np.argmax(probs)
    print(f"🔍 预测类别ID: {target_category} (置信度: {probs[target_category]:.2%})")

    # 5. 生成 Grad-CAM Mask
    target_layers = get_target_layer(model, target_layer_name)
    with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_category)]
        # 这里生成的 grayscale_cam 也是 224x224 的
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # === 关键步骤：把 224x224 的 mask 拉伸回原图尺寸 ===
        grayscale_cam_highres = cv2.resize(grayscale_cam, (w_origin, h_origin))
        
        # 准备原图数据用于叠加
        rgb_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
        rgb_origin_float = np.float32(rgb_origin) / 255
        
        # 在原图尺寸上叠加热力图
        visualization_rgb = show_cam_on_image(rgb_origin_float, grayscale_cam_highres, use_rgb=True)

    # 6. 保存
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    full_save_path = os.path.join(save_dir, save_name)
    visualization_bgr = cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(full_save_path, visualization_bgr)
    print(f"✅ 原尺寸热力图已保存: {full_save_path} ({w_origin}x{h_origin})")

if __name__ == '__main__':
    main()