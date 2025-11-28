import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

# 尝试导入新版接口 (兼容 MMCV 2.x)
try:
    from mmpretrain.apis import init_model
except ImportError:
    print("❌ 错误: 你的环境是 MMCV 2.x，请安装 mmpretrain!")
    print("👉 运行: pip install mmpretrain")
    exit()

# ================= 1. 用户配置区域 =================

models_config = [
    {
        'name': 'ResNet-50',
        'config': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/configs/Amytest/B_1.py', 
        'ckpt': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/work_dirs/B_1/best_accuracy_top1_epoch_91.pth',
        'target_layer_name': 'backbone.layer4[-1]' 
    },
    {
        'name': 'ViT-Small',
        'config': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/configs/Amytest/B_2.py',
        'ckpt': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/work_dirs/B_2/best_accuracy_top1_epoch_95.pth',
        'target_layer_name': 'backbone.layers[-1].ln1',
        'is_transformer': True
    },
    {
        'name': 'Swin-Tiny',
        'config': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/configs/Amytest/B_3.py',
        'ckpt': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/work_dirs/B_3/best_accuracy_top1_epoch_90.pth',
        'target_layer_name': 'backbone.stages[-1].blocks[-1].norm1', 
        'is_transformer': True
    },
    {
        'name': 'ConvNeXt-T',
        'config': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/configs/Amytest/B_4.py',
        'ckpt': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/work_dirs/B_4/best_accuracy_top1_epoch_89.pth',
        'target_layer_name': 'backbone.stages[3][-1].norm',
        'is_transformer': True 
    },
    {
        'name': 'MobileNet-V3',
        'config': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/configs/Amytest/B_5.py',
        'ckpt': '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/work_dirs/B_5/best_accuracy_top1_epoch_65.pth',
        'target_layer_name': 'backbone.layer16' 
    }
]

images_info = [
    {'path': '/media/HDD0/wzl/mmcls/dataset_1/train/apple leaf/apple-tree-branch-plant-fruit-leaf-flower-food-green-produce-flora-immature-apple-tree-flowering-plant-wild-apple-tree-apple-gear-land-plant-606389.jpg', 'gt_label': 0},
    {'path': '/media/HDD0/wzl/mmcls/dataset_1/train/apple rust leaf/20130802_111648.jpg',     'gt_label': 1},
    {'path': '/media/HDD0/wzl/mmcls/dataset_1/val/Apple Scab Leaf/1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg',   'gt_label': 2},
    {'path': '/media/HDD0/wzl/mmcls/dataset_1/val/bell pepper leaf/e33155.jpg',      'gt_label': 3},
    {'path': '/media/HDD0/wzl/mmcls/dataset_1/val/bell pepper leaf spot/CMVpepperLeafShock-copy-50QUALITY-1ge8umw.jpg',  'gt_label': 4},
]

# ==============================================================

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
                if isinstance(feats, tuple):
                    feat = feats[-1]
                else:
                    feat = feats
                logits = self.model.head.fc(self.model.head.pre_logits(feat))
        else:
             logits = self.model.head.fc(feats)
        if isinstance(logits, dict):
            return logits['pred_scores']
        return logits

def reshape_transform_vit(tensor):
    if len(tensor.shape) == 3:
        token = tensor[:, 1:, :] 
        seq_len = token.shape[1]
        h = w = int(np.sqrt(seq_len))
        result = token.reshape(tensor.size(0), h, w, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    return tensor

def reshape_transform_swin_convnext(tensor):
    if len(tensor.shape) == 3:
        B, L, C = tensor.shape
        H = W = int(np.sqrt(L))
        result = tensor.reshape(B, H, W, C)
        result = result.permute(0, 3, 1, 2)
        return result
    if len(tensor.shape) == 4:
        if tensor.shape[-1] > tensor.shape[1]: 
             return tensor.permute(0, 3, 1, 2)
    return tensor

def get_target_layer(model, layer_str):
    try:
        parts = layer_str.replace('[-1]', '.__last__').split('.')
        current = model
        for part in parts:
            if part == '__last__':
                current = current[-1]
            elif '[' in part and ']' in part:
                name = part.split('[')[0]
                idx = int(part.split('[')[1].replace(']',''))
                current = getattr(current, name)[idx]
            else:
                current = getattr(current, part)
        return [current]
    except:
        return []

def process_one_image(model, img_info, target_layer_name, is_transformer, model_name=''):
    img_path = img_info['path']
    
    img = cv2.imread(img_path)
    if img is None: 
        return None, None
        
    img_resized = cv2.resize(img, (224, 224))
    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    rgb_img_float = np.float32(rgb_img) / 255
    
    input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    wrapped_model = ModelWrapper(model)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 依然需要推理，为了获取预测类别来生成热力图
    with torch.no_grad():
        logits = wrapped_model(input_tensor)
    pred_score = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(pred_score).item()
    
    target_layers = get_target_layer(model, target_layer_name)
    if not target_layers:
        return rgb_img, rgb_img
    
    reshape_func = None
    if is_transformer:
        if 'swin' in model_name.lower() or 'convnext' in model_name.lower() or 'swin' in target_layer_name.lower():
            reshape_func = reshape_transform_swin_convnext
        else:
            reshape_func = reshape_transform_vit
            
    with GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_func) as cam:
        targets = [ClassifierOutputTarget(pred_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        # visualization 是 RGB 格式
    
    # 不加边框，直接返回 RGB 格式的图像
    return visualization, rgb_img

def main():
    num_imgs = len(images_info)
    num_models = len(models_config)
    
    # 1 列原图 + N 列模型
    total_cols = num_models + 1
    total_rows = num_imgs
    
    fig, axes = plt.subplots(total_rows, total_cols, figsize=(3 * total_cols, 3.5 * total_rows), dpi=150)
    plt.subplots_adjust(wspace=0.05, hspace=0.1) # 边框去掉了，间距可以更小一点
    
    # 设置左上角标题
    axes[0, 0].set_title("Original", fontsize=20, fontweight='bold')
    
    for m_idx, m_conf in enumerate(models_config):
        print(f"🚀 处理模型 [{m_idx+1}/{num_models}]: {m_conf['name']} ...")
        
        # 列标题
        col_idx = m_idx + 1
        axes[0, col_idx].set_title(m_conf['name'], fontsize=20, fontweight='bold')
        
        try:
            model = init_model(m_conf['config'], m_conf['ckpt'], device='cuda:0')
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            continue

        for i_idx, img_conf in enumerate(images_info):
            cam_img, orig_img = process_one_image(
                model, 
                img_conf, 
                m_conf.get('target_layer_name'),
                m_conf.get('is_transformer', False),
                m_conf['name']
            )
            
            # --- 绘制 CAM 图 (Row=i_idx, Col=col_idx) ---
            ax_cam = axes[i_idx, col_idx]
            if cam_img is not None:
                ax_cam.imshow(cam_img)
            else:
                ax_cam.text(0.5, 0.5, "Err", ha='center')
            
            # --- 绘制 原图 (Row=i_idx, Col=0) ---
            # 只有在第一个模型时绘制原图，避免重复
            if m_idx == 0:
                ax_orig = axes[i_idx, 0]
                if orig_img is not None:
                    ax_orig.imshow(orig_img)
                
                # 左侧标签
                ax_orig.set_ylabel(f"IMG {i_idx+1}", fontsize=11, fontweight='bold', labelpad=10)
                
                # 隐藏原图坐标轴
                ax_orig.set_xticks([]); ax_orig.set_yticks([])
                for spine in ax_orig.spines.values(): spine.set_visible(False)

            # 隐藏 CAM 坐标轴
            ax_cam.set_xticks([]); ax_cam.set_yticks([])
            for spine in ax_cam.spines.values(): spine.set_visible(False)
        
        del model
        torch.cuda.empty_cache()

    output_filename = 'B_Attention_Matrix_Clean.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\n✅ 成功！干净的矩阵图已保存为: {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()