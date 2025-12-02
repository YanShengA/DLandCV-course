import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 尝试导入 Config，自动适配新旧版本
try:
    from mmengine import Config  # MMCV 2.x / MMEngine
except ImportError:
    from mmcv import Config      # MMCV 1.x (旧版)

# ================= 配置区域 =================
# 1. 你的配置文件路径
config_file = '/media/HDD0/wzl/mmcls/DLandCV-course/homework1/DL/mmclassification/configs/Amytest/D_10.py' 

# 2. 你的预测结果文件 (.pkl)
result_file = 'results.pkl'

# 3. 输出图片名称
output_file = 'F1_Confusion_Matrix_Final.png'

# 4. PlantDoc 类别名称 (硬编码最安全)
class_names = [
    'Apple leaf', 'Apple rust leaf', 'Apple Scab Leaf', 
    'Bell_pepper leaf', 'Bell_pepper leaf spot', 
    'Blueberry leaf', 
    'Cherry leaf', 
    'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 
    'Grape Black Rot', 'Grape leaf', 'Grape leaf spot', 
    'Peach leaf', 
    'Potato leaf', 'Potato leaf early blight', 'Potato leaf late blight', 
    'Raspberry leaf', 
    'Soyabean leaf', 
    'Squash Powdery mildew leaf', 
    'Strawberry leaf', 'Strawberry Leaf Scorch', 
    'Tomato leaf', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 
    'Tomato Septoria leaf spot', 'Tomato leaf yellow virus'
]
# ===========================================

def get_ann_file_from_config(cfg_path):
    """从配置文件中解析出验证集/测试集的 txt 路径"""
    cfg = Config.fromfile(cfg_path)
    
    # 尝试寻找 test 或 val 的配置
    if hasattr(cfg.data, 'test'):
        data_cfg = cfg.data.test
    elif hasattr(cfg.data, 'val'):
        data_cfg = cfg.data.val
    else:
        raise ValueError("无法在配置中找到 data.test 或 data.val")

    # 拼接路径
    # 通常配置是 data_prefix='data/plantdoc', ann_file='val.txt'
    # 或者直接 ann_file='data/plantdoc/val.txt'
    ann_file = data_cfg.get('ann_file')
    data_prefix = data_cfg.get('data_prefix', '')
    
    # 如果 ann_file 已经是绝对路径或包含路径，直接用
    if os.path.exists(ann_file):
        return ann_file
    
    # 否则尝试拼接
    full_path = os.path.join(data_prefix, ann_file)
    # 处理 mmcls 常见的 data_prefix 配置格式 (可能是 dict)
    if isinstance(data_prefix, dict): 
        # 有时候是 img_path='data/plantdoc'
        prefix_path = data_prefix.get('img_path', '')
        full_path = os.path.join(prefix_path, ann_file)
        
    # 如果拼接后还不对，尝试去 data_root 找
    if not os.path.exists(full_path) and hasattr(cfg, 'data_root'):
         full_path = os.path.join(cfg.data_root, ann_file)

    print(f"🔍 从配置中解析出标签文件: {full_path}")
    return full_path

def load_ground_truth(txt_path):
    """读取 txt 获取真实标签"""
    if not os.path.exists(txt_path):
        print(f"❌ 错误: 找不到文件 {txt_path}")
        # 如果自动解析失败，请手动在这里填入路径并取消注释:
        # txt_path = 'data/plantdoc/val.txt' 
        # return load_ground_truth(txt_path)
        return None
    
    gt_labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            gt_labels.append(int(parts[-1]))
    return np.array(gt_labels)

def plot_cm():
    # 1. 自动获取真实标签
    try:
        ann_path = get_ann_file_from_config(config_file)
        gt_labels = load_ground_truth(ann_path)
        if gt_labels is None: return
    except Exception as e:
        print(f"❌ 解析配置失败: {e}")
        print("建议直接修改脚本中的 ann_path 为你的 val.txt 路径")
        return

    # 2. 加载预测结果
    print(f"正在加载预测结果: {result_file}")
    with open(result_file, 'rb') as f:
        results = pickle.load(f)

    pred_labels = []
    for res in results:
        # 兼容各种格式
        if isinstance(res, dict) and 'pred_score' in res:
            score = res['pred_score']
        elif isinstance(res, np.ndarray):
            score = res
        else:
            score = np.array(res)
        pred_labels.append(np.argmax(score))
    
    pred_labels = np.array(pred_labels)

    # 3. 校验长度
    if len(gt_labels) != len(pred_labels):
        print(f"⚠️ 警告: 标签数({len(gt_labels)}) != 预测数({len(pred_labels)})")
        min_len = min(len(gt_labels), len(pred_labels))
        gt_labels = gt_labels[:min_len]
        pred_labels = pred_labels[:min_len]

    # 4. 绘图
    print("正在绘图...")
    cm = confusion_matrix(gt_labels, pred_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(16, 14), dpi=300)
    sns.set(font_scale=0.85)
    
    # 绘图
    ax = sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='YlGnBu', 
                     xticklabels=class_names, yticklabels=class_names,
                     square=True, linewidths=0.5, linecolor='#d8d8d8',
                     cbar_kws={'shrink': 0.8})

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel('Predicted Label', fontsize=15, fontweight='bold', labelpad=20)
    plt.ylabel('Ground Truth Label', fontsize=15, fontweight='bold', labelpad=20)
    plt.title('Confusion Matrix (PlantDoc)', fontsize=18, fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ 图片已保存: {output_file}")
    plt.show()

if __name__ == '__main__':
    plot_cm()