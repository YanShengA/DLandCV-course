# feature_engineering.py
"""
特征工程模块
功能：图像预处理、特征提取、数据增强和特征可视化
"""

import os
import random
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# 特征提取相关库
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

# ----------------- 路径配置 -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'PlantDoc')
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'TRAIN')
TEST_DATA_PATH = os.path.join(DATA_ROOT, 'TEST')

# Matplotlib 中文显示配置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# === 辅助函数：图像数据增强 ===
def augment_image_data(image_path: str) -> np.ndarray | None:
    """
    对图像进行随机数据增强：随机旋转、随机翻转。
    
    Args:
        image_path: 原始图像路径。
        
    Returns:
        增强后的图像数据 (np.ndarray)，或 None (如果失败)。
    """
    try:
        # 使用 PIL 加载，因为 PIL 对几何变换操作更稳定和灵活
        img = Image.open(image_path)
        
        # 1. 随机水平或垂直翻转
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
        # 2. 随机旋转 (以 90 度为单位)
        rotation_angle = random.choice([0, 90, 180, 270])
        if rotation_angle != 0:
            # expand=True 确保旋转后图像不会被裁剪
            img = img.rotate(rotation_angle, expand=True) 
        
        # 转换为 numpy 数组 (供 skimage 使用)
        img_np = np.array(img)
        return img_np
        
    except Exception:
        return None


# === 3.1.1 图像预处理函数 ===
def preprocess_image_for_hog(image_path: str, size: tuple) -> np.ndarray | None:
    """
    3.1.1 预处理：加载图像、转换为灰度图、统一缩放至指定尺寸。
    """
    try:
        img = imread(image_path)
        
        # 转换为灰度图 
        if img.ndim == 3:
            # 使用 PIL 转换为灰度图，保证转换的可靠性
            pil_img = Image.open(image_path).convert('L') 
            img = np.array(pil_img)
        
        # 统一缩放尺寸 (skimage resize 自动进行 [0, 1] 归一化)
        resized_img = resize(img, size, anti_aliasing=True)
        return resized_img
    except Exception as e:
        return None


# === 3.1.2 特征提取：HOG 特征提取 ===
def extract_hog_features(image_data: np.ndarray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(3, 3)) -> np.ndarray:
    """
    3.1.2 特征提取：从预处理后的图像数据中提取 HOG 特征。
    """
    features = hog(image_data, 
                   orientations=orientations, 
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, 
                   transform_sqrt=True,
                   channel_axis=None) 
    return features


# === 3.1.3 构建特征矩阵 (已集成数据增强/过采样逻辑) ===
def build_feature_matrix(data_path: str, input_size: tuple, is_training: bool, target_sample_count: int = 100) -> tuple[np.ndarray, np.ndarray, list]:
    """
    3.1.3 构建特征矩阵：遍历数据集，提取 HOG 特征，并构建 X 和标签向量 y。
    
    Args:
        is_training: 是否为训练集 (仅训练集进行增强)。
        target_sample_count: 训练集小类别的目标样本数。
    """
    X = []
    y_raw = []
    
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_image_files = {} # {类别名: [文件路径列表]}
    current_class_counts = defaultdict(int)

    # 1. 预先加载所有文件的路径
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                 if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.jpg', '.jpeg'))]
        class_image_files[class_name] = files
        current_class_counts[class_name] = len(files)

    print(f"  > 开始提取 {os.path.basename(data_path)} 集 HOG 特征 (3.1.2)...")
    
    # 2. 遍历所有类别进行特征提取 (包括原始样本和增强样本)
    for class_name in class_names:
        files = class_image_files[class_name]
        
        # 2A. 计算增强次数 (仅针对训练集)
        if is_training and current_class_counts[class_name] < target_sample_count:
            # 需要生成的额外样本数
            needed_augmentations = target_sample_count - current_class_counts[class_name]
            # 基础样本数 (原始样本)
            base_sample_count = len(files) 
            # 每个基础样本平均需要增强多少次
            times_to_augment_per_sample = needed_augmentations // base_sample_count
            # 剩余需要随机抽取的次数
            remaining_augmentations = needed_augmentations % base_sample_count
            
            # 打印过采样信息
            print(f"    - 训练集过采样: {class_name} ({base_sample_count} -> {target_sample_count}). Augment count: {needed_augmentations}")
            
            # 3. 执行过采样：对所有基础样本进行多次增强
            augmented_paths = files * times_to_augment_per_sample + random.sample(files, remaining_augmentations)
            files_to_process = files + augmented_paths
        
        else:
            # 非训练集或样本数已达标，只处理原始样本
            files_to_process = files

        # 4. 提取特征
        for file_path in files_to_process:
            is_augmented_sample = (file_path in files_to_process and file_path not in files)

            if is_augmented_sample:
                # 对增强样本：先增强，再预处理
                augmented_img_data = augment_image_data(file_path) # 返回 np.ndarray (RGB/Gray)
                
                if augmented_img_data is not None:
                    # 3.1.1 预处理：缩放和最终归一化
                    # skimage resize 会将 RGB/Gray 自动处理
                    resized_img = resize(augmented_img_data, input_size, anti_aliasing=True)
                else:
                    continue # 增强失败，跳过
            else:
                # 对原始样本：直接预处理 (3.1.1)
                resized_img = preprocess_image_for_hog(file_path, input_size)

            if resized_img is not None:
                # 3.1.2 提取特征
                features = extract_hog_features(resized_img)
                
                # 3. 构建矩阵
                X.append(features)
                y_raw.append(class_name)
                    
    X_matrix = np.array(X)
    y_vector = np.array(y_raw)
    
    print(f"  > {os.path.basename(data_path)} 集特征矩阵 X.shape: {X_matrix.shape}")
    
    return X_matrix, y_vector, class_names


# === 阶段 3 总体执行函数 ===
def run_feature_engineering(input_size: tuple):
    
    print("\n\n# --- 阶段 3：特征工程 ---")
    
    # 转换为 (height, width) 供 skimage resize 使用
    resize_dims = (input_size[1], input_size[0]) 

    # --- 3.1.3 HOG 特征矩阵构建 ---
    
    # 1. 增强版 X_train (用于主流程)
    X_train_augmented, y_train_raw, class_names = build_feature_matrix(TRAIN_DATA_PATH, resize_dims, is_training=True)
    
    # 2. 非增强版 X_train (用于消融实验 6.2)
    X_train_original, y_train_original_raw, _ = build_feature_matrix(TRAIN_DATA_PATH, resize_dims, is_training=False, target_sample_count=1) # 不进行增强，只提取原始样本

    X_test, y_test_raw, _ = build_feature_matrix(TEST_DATA_PATH, resize_dims, is_training=False) # 最终测试集
    
    # --- 3.2 标签编码 ---
    print("\n# --- 3.2 标签编码 ---")
    le = LabelEncoder()
    le.fit(class_names) 
    
    y_train = le.transform(y_train_raw)               # 增强版的 y
    y_test = le.transform(y_test_raw)                 # 测试集的 y
    y_train_original = le.transform(y_train_original_raw) # 原始版的 y
    
    # 打印最终结果
    print(f"  > 类别编码成功。总类别数: {len(le.classes_)}")
    print(f"  > X_train (增强) 最终形状: {X_train_augmented.shape}")
    print(f"  > X_test 最终形状: {X_test.shape}")
    print(f"  > X_train_original (非增强) 最终形状: {X_train_original.shape}")
    
    # --- 新增：特征工程可视化 ---
    visualize_feature_engineering_results(X_train_augmented, X_train_original, X_test, 
                                        y_train, y_train_original, y_test, class_names, le, resize_dims)
    
    return X_train_augmented, y_train, X_test, y_test, X_train_original, y_train_original

def visualize_preprocessing_steps(data_path: str, input_size: tuple):
    """
    可视化图像预处理全过程
    """
    print("  > 生成图像预处理可视化示例...")
    
    # 随机选择几个样本进行可视化
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    selected_classes = random.sample(class_dirs, min(3, len(class_dirs)))
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i, class_name in enumerate(selected_classes):
        if i >= 3:
            break
            
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not image_files:
            continue
            
        # 随机选择一个图像
        random_file = random.choice(image_files)
        image_path = os.path.join(class_path, random_file)
        
        try:
            # 1. 原始彩色图像
            original_img = Image.open(image_path)
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f'{class_name}\n原始彩色图像\n{original_img.size}', fontsize=9)
            axes[i, 0].axis('off')
            
            # 2. 原始图像的灰度图
            gray_img = original_img.convert('L')
            axes[i, 1].imshow(gray_img, cmap='gray')
            axes[i, 1].set_title('直接灰度化\n(未缩放)', fontsize=9)
            axes[i, 1].axis('off')
            
            # 3. 预处理后的灰度图（缩放后）
            processed_img = preprocess_image_for_hog(image_path, input_size)
            axes[i, 2].imshow(processed_img, cmap='gray')
            axes[i, 2].set_title(f'预处理后\n{input_size}尺寸', fontsize=9)
            axes[i, 2].axis('off')
            
            # 4. 预处理图像的像素值分布
            axes[i, 3].hist(processed_img.flatten(), bins=50, color='blue', alpha=0.7)
            axes[i, 3].set_title('像素值分布\n(归一化后)', fontsize=9)
            axes[i, 3].set_xlabel('像素值')
            axes[i, 3].set_ylabel('频数')
            
            original_img.close()
            
        except Exception as e:
            print(f"    - 预处理可视化失败: {class_name}/{random_file}")
            for j in range(4):
                axes[i, j].text(0.5, 0.5, '加载失败', ha='center', va='center', 
                               transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '11_Preprocessing_Steps.png'))
    plt.close()
    print("✅ 图像预处理步骤图已保存为：11_Preprocessing_Steps.png")


def visualize_preprocessing_comparison(data_path: str, input_size: tuple):
    """
    可视化不同类别的预处理效果对比
    """
    print("  > 生成预处理效果对比...")
    
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    selected_classes = random.sample(class_dirs, min(6, len(class_dirs)))
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    
    for i, class_name in enumerate(selected_classes):
        if i >= 6:
            break
            
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not image_files:
            continue
            
        # 随机选择一个图像
        random_file = random.choice(image_files)
        image_path = os.path.join(class_path, random_file)
        
        try:
            # 原始图像
            original_img = Image.open(image_path)
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'{class_name}\n原始图像', fontsize=10)
            axes[0, i].axis('off')
            
            # 预处理后图像
            processed_img = preprocess_image_for_hog(image_path, input_size)
            axes[1, i].imshow(processed_img, cmap='gray')
            axes[1, i].set_title(f'预处理后\n{processed_img.shape}', fontsize=10)
            axes[1, i].axis('off')
            
            original_img.close()
            
        except Exception as e:
            axes[0, i].text(0.5, 0.5, '加载失败', ha='center', va='center', 
                           transform=axes[0, i].transAxes)
            axes[0, i].axis('off')
            axes[1, i].axis('off')
    
    plt.suptitle('不同类别图像预处理效果对比', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '12_Preprocessing_Comparison.png'))
    plt.close()
    print("✅ 预处理效果对比图已保存为：12_Preprocessing_Comparison.png")

# === 新增：特征工程可视化函数 ===
def visualize_feature_engineering_results(X_train_aug, X_train_orig, X_test, 
                                        y_train, y_train_orig, y_test, class_names, label_encoder, input_size):
    """
    可视化特征工程结果
    
    Args:
        input_size: 图像预处理尺寸 (height, width)
    """
    print("\n# --- 特征工程可视化 ---")
    
    # 1. 特征维度分布图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(X_train_aug.flatten(), bins=50, alpha=0.7, color='blue', label='增强训练集')
    plt.xlabel('特征值')
    plt.ylabel('频数')
    plt.title('增强训练集特征值分布')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(X_train_orig.flatten(), bins=50, alpha=0.7, color='green', label='原始训练集')
    plt.xlabel('特征值')
    plt.ylabel('频数')
    plt.title('原始训练集特征值分布')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(X_test.flatten(), bins=50, alpha=0.7, color='red', label='测试集')
    plt.xlabel('特征值')
    plt.ylabel('频数')
    plt.title('测试集特征值分布')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '7_Feature_Distribution.png'))
    plt.close()
    print("✅ 特征值分布图已保存为：7_Feature_Distribution.png")
    
    # 2. 数据增强效果对比
    plt.figure(figsize=(12, 6))
    
    # 统计各类别样本数量
    unique_train_aug, counts_train_aug = np.unique(y_train, return_counts=True)
    unique_train_orig, counts_train_orig = np.unique(y_train_orig, return_counts=True)
    
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x_pos - width/2, counts_train_orig, width, label='原始训练集', alpha=0.7)
    plt.bar(x_pos + width/2, counts_train_aug, width, label='增强训练集', alpha=0.7)
    
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('数据增强前后样本数量对比')
    plt.xticks(x_pos, [str(i) for i in range(len(class_names))], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '8_Data_Augmentation_Comparison.png'))
    plt.close()
    print("✅ 数据增强对比图已保存为：8_Data_Augmentation_Comparison.png")
    
    # 3. 图像预处理可视化（新增）
    visualize_preprocessing_steps(TRAIN_DATA_PATH, input_size)
    visualize_preprocessing_comparison(TRAIN_DATA_PATH, input_size)

    # 4. HOG特征可视化示例
    visualize_hog_features_example(TRAIN_DATA_PATH, input_size)
    
    # 5. 数据增强样本可视化
    visualize_augmented_samples(TRAIN_DATA_PATH, input_size)


def visualize_hog_features_example(data_path: str, input_size: tuple):
    """
    可视化HOG特征提取过程
    """
    print("  > 生成HOG特征可视化示例...")
    
    # 随机选择几个样本进行可视化
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    selected_classes = random.sample(class_dirs, min(4, len(class_dirs)))
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    for i, class_name in enumerate(selected_classes):
        if i >= 4:
            break
            
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not image_files:
            continue
            
        # 随机选择一个图像
        random_file = random.choice(image_files)
        image_path = os.path.join(class_path, random_file)
        
        try:
            # 原始图像
            original_img = Image.open(image_path)
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f'{class_name}\n原始图像', fontsize=10)
            axes[i, 0].axis('off')
            
            # 预处理后的灰度图
            processed_img = preprocess_image_for_hog(image_path, input_size)
            axes[i, 1].imshow(processed_img, cmap='gray')
            axes[i, 1].set_title('预处理后灰度图', fontsize=10)
            axes[i, 1].axis('off')
            
            # HOG特征可视化
            features, hog_image = hog(processed_img, 
                                    orientations=9, 
                                    pixels_per_cell=(16, 16),
                                    cells_per_block=(3, 3), 
                                    transform_sqrt=True,
                                    channel_axis=None,
                                    visualize=True)
            
            axes[i, 2].imshow(hog_image, cmap='hot')
            axes[i, 2].set_title(f'HOG特征\n特征维度: {len(features)}', fontsize=10)
            axes[i, 2].axis('off')
            
            original_img.close()
            
        except Exception as e:
            print(f"    - 可视化失败: {class_name}/{random_file}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '9_HOG_Feature_Visualization.png'))
    plt.close()
    print("✅ HOG特征可视化图已保存为：9_HOG_Feature_Visualization.png")


def visualize_augmented_samples(data_path: str, input_size: tuple):
    """
    可视化数据增强效果
    """
    print("  > 生成数据增强可视化示例...")
    
    # 随机选择一个类别和图像
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not class_dirs:
        return
        
    class_name = random.choice(class_dirs)
    class_path = os.path.join(data_path, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        return
        
    original_image_path = os.path.join(class_path, random.choice(image_files))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 原始图像
    try:
        original_img = Image.open(original_image_path)
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')
        original_img.close()
    except:
        axes[0, 0].text(0.5, 0.5, '加载失败', ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # 生成并显示7个增强样本
    augmentation_count = 0
    attempts = 0
    max_attempts = 20  # 防止无限循环
    
    while augmentation_count < 7 and attempts < max_attempts:
        attempts += 1
        augmented_img = augment_image_data(original_image_path)
        
        if augmented_img is not None:
            row = (augmentation_count + 1) // 4
            col = (augmentation_count + 1) % 4
            axes[row, col].imshow(augmented_img)
            axes[row, col].set_title(f'增强样本 {augmentation_count + 1}', fontsize=10)
            axes[row, col].axis('off')
            augmentation_count += 1
    
    # 隐藏多余的子图
    for i in range(augmentation_count + 1, 8):
        row = i // 4
        col = i % 4
        if row < 2 and col < 4:
            axes[row, col].axis('off')
    
    plt.suptitle(f'数据增强效果示例 - {class_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '10_Data_Augmentation_Examples.png'))
    plt.close()
    print("✅ 数据增强示例图已保存为：10_Data_Augmentation_Examples.png")


# === 模型训练辅助函数（从main.py复制，用于特征工程后的快速验证）===
def get_model_report(model, X_train, X_test, y_train, y_test, name) -> pd.DataFrame:
    """训练模型并计算多分类指标 (Accuracy, Macro-Recall/Precision/F1)。"""
    
    # 设置模型参数：处理类别不均衡
    if hasattr(model, 'class_weight'):
        # 针对 LR, SVM, RF
        model.set_params(class_weight='balanced', random_state=42)
    elif hasattr(model, 'random_state'):
        # 针对 KNN (如果使用)
        model.set_params(random_state=42)
        
    print(f"  > 训练模型: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 评估指标 (多分类必须指定 average 参数)
    accuracy = accuracy_score(y_test, y_pred)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    df = pd.DataFrame({"Model": [name],
                       "Accuracy": [f"{accuracy:.4f}"],
                       "Macro_Recall": [f"{macro_recall:.4f}"],
                       "Macro_Precision": [f"{macro_precision:.4f}"],
                       "Macro_F1": [f"{macro_f1:.4f}"],
                       "Weighted_F1": [f"{weighted_f1:.4f}"],
                       })

    return df


if __name__ == "__main__":
    # 测试特征工程模块
    input_size = (256, 256)  # 示例输入尺寸
    X_train_aug, y_train, X_test, y_test, X_train_orig, y_train_orig = run_feature_engineering(input_size)
    print(f"\n特征工程完成！")
    print(f"增强训练集: {X_train_aug.shape}")
    print(f"原始训练集: {X_train_orig.shape}")
    print(f"测试集: {X_test.shape}")