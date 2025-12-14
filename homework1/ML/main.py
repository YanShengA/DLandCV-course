import os
import sys
from collections import defaultdict
import csv
import random
# 所有核心库导入
import numpy as np 
from PIL import Image 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties # 用于解决中文显示问题
# === 新增导入 ===
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder 
from skimage.transform import rotate
import numpy as np 
# 导入 PCA 和 SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler # 确保特征在PCA前是标准化的
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 模型评估
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.metrics import ConfusionMatrixDisplay
from skimage.transform import rescale
from matplotlib import colors
# Matplotlib 中文显示配置（可选，解决图表中文乱码）
# 尝试使用 SimHei 字体，如果系统没有，请注释掉或更换为系统支持的中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception:
    # 如果 SimHei 字体找不到，将使用默认字体，图表中文会显示为方块（正如之前所见）
    pass


# ----------------- 路径配置 -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_ROOT = os.path.join(BASE_DIR, 'PlantDoc') 
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'TRAIN')
TEST_DATA_PATH = os.path.join(DATA_ROOT, 'TEST')


# === 辅助函数 (数据统计与可视化) ===
# --- 1.2 统计分类数量 依赖 ---
def get_file_counts(data_path: str) -> dict:
    """遍历指定路径下的所有子文件夹（类别），统计每个类别的图像数量。"""
    class_counts = defaultdict(int)
    if not os.path.exists(data_path):
        return class_counts
        
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            file_count = sum([1 for item in os.listdir(class_path) 
                              if os.path.isfile(os.path.join(class_path, item)) and item.lower().endswith(('.jpg', '.jpeg'))])
            class_counts[class_name] = file_count
    return class_counts


# --- 2.1 & 2.2 检查损坏文件 & 图像尺寸分析 依赖 ---
def analyze_image_properties(data_path: str, set_name: str) -> tuple[dict, int]:
    """遍历图像，检查损坏（2.1）并记录尺寸（2.2）。"""
    image_sizes = defaultdict(int) 
    damaged_files = []             
    total_images_checked = 0

    if not os.path.exists(data_path): return image_sizes, 0

    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                file_path = os.path.join(class_path, image_file)
                if os.path.isfile(file_path) and image_file.lower().endswith(('.jpg', '.jpeg')):
                    total_images_checked += 1
                    try:
                        img = Image.open(file_path)
                        img.load() 
                        image_sizes[img.size] += 1
                        img.close()
                    except Exception as e:
                        damaged_files.append((class_name, image_file, str(e)))
                        
    # 结果展示
    print(f"  > {set_name} 总共检查图像数量: {total_images_checked} 个")
    print(f"  > {set_name} 发现损坏文件数量: {len(damaged_files)} 个")
    
    if damaged_files:
        with open(os.path.join(BASE_DIR, f'{set_name}_damaged_files.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['类别', '文件名', '错误信息'])
            writer.writerows(damaged_files)
    
    print(f"  > {set_name} 图像尺寸分布 (Top 5 尺寸):")
    sorted_sizes = sorted(image_sizes.items(), key=lambda item: item[1], reverse=True)
    for (size, count) in sorted_sizes[:5]:
        width, height = size
        ratio = round(width / height, 2)
        print(f"    尺寸: {width}x{height} (长宽比: {ratio}) -> 数量: {count} 个")

    return image_sizes, len(damaged_files)

# --- 2.3.2 随机图像可视化 依赖 ---
def visualize_random_samples(data_path, num_samples_per_class=1):
    """从每个类别随机选择N张图片进行可视化"""
    
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    num_classes = len(class_dirs)
    
    grid_size = int(np.ceil(np.sqrt(num_classes)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_dirs):
        if i >= len(axes): break
            
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if image_files:
            random_file = random.choice(image_files)
            try:
                img = Image.open(os.path.join(class_path, random_file))
                axes[i].imshow(img)
                # 使用中文标题
                axes[i].set_title(class_name.replace(' ', '\n'), fontsize=10)
                axes[i].axis('off')
                img.close()
            except Exception as e:
                axes[i].set_title(f"{class_name}\n(加载失败)", fontsize=10)
                axes[i].axis('off')

    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '2_Random_Samples.png'))
    plt.close()


# === 主程序执行逻辑 ===
def run_data_loading_and_eda():
    
    # --- 1.1 数据集描述 ---
    print("\n# --- 1.1 数据集描述 ---")
    train_counts = get_file_counts(TRAIN_DATA_PATH)
    test_counts = get_file_counts(TEST_DATA_PATH)
    
    # --- 1.2 统计分类数量  ---
    print("\n# --- 1.2 统计分类数量 ---")
    print(f"训练集类别数: {len(train_counts)}, 总样本: {sum(train_counts.values())}")
    print(f"测试集类别数: {len(test_counts)}, 总样本: {sum(test_counts.values())}")
    
    # --- 1.3 确认训练/测试集划分  ---
    print("\n# --- 1.3 确认训练/测试集划分 ---")
    df_train = pd.DataFrame(train_counts.items(), columns=['类别名', '训练集数量'])
    df_test = pd.DataFrame(test_counts.items(), columns=['类别名', '测试集数量'])
    df_combined = pd.merge(df_train, df_test, on='类别名', how='outer').fillna(0)
    df_combined['训练集数量'] = df_combined['训练集数量'].astype(int); 
    df_combined['总样本数量'] = df_combined['训练集数量'] + df_combined['测试集数量']
    df_combined['训练集占比 (%)'] = round((df_combined['训练集数量'] / df_combined['总样本数量']) * 100, 2)
    df_combined = df_combined.sort_values(by='总样本数量', ascending=False).reset_index(drop=True)

    print("\n== 详细分类统计表 ==")
    print(df_combined)
    df_combined.to_csv(os.path.join(BASE_DIR, 'dataset_distribution_stats.csv'), index=False, encoding='utf-8')
    print("✅ 详细统计已保存至：dataset_distribution_stats.csv")
            
    # --- 2.1 检查损坏文件 ---
    print("\n# --- 2.1 检查损坏文件 ---")
    train_sizes, train_damaged = analyze_image_properties(TRAIN_DATA_PATH, 'TRAIN')
    test_sizes, test_damaged = analyze_image_properties(TEST_DATA_PATH, 'TEST')
    
    if train_damaged or test_damaged:
        print(f"发现损坏文件！请检查 TRAIN_damaged_files.csv 和 TEST_damaged_files.csv")
        
    # --- 2.2 图像尺寸分布分析 ---
    print("\n# --- 2.2 图像尺寸分布分析 ---")
    all_sizes = train_sizes.copy()
    for size, count in test_sizes.items(): 
        all_sizes[size] += count
    sorted_all_sizes = sorted(all_sizes.items(), key=lambda item: item[1], reverse=True)
    
    standard_size = (256, 256)
    if sorted_all_sizes:
        most_common_size = sorted_all_sizes[0][0]
        # 给出标准输入尺寸建议
        standard_size = (512, 512) if most_common_size[0] > 256 or most_common_size[1] > 256 else (256, 256)
        print(f"\n✅ 建议的模型标准输入尺寸为： {standard_size[0]}x{standard_size[1]}")
    else:
        print(f"\n✅ 无法确定最常见尺寸，使用默认输入尺寸： {standard_size[0]}x{standard_size[1]}")

    # --- 2.3 图像长宽比分析 ---
    print("\n# --- 2.3 图像长宽比分析 ---")
    # 提取长宽比
    ratios = {round(w/h, 2) for w, h in all_sizes.keys()}
    print(f"  > 发现的不同长宽比数量: {len(ratios)}")
    print(f"  > 长宽比集合: {sorted(list(ratios))[:5]} ... (仅显示前5个)")

    # --- 2.4 可视化探索分析 ---
    # --- 2.4.1 类别分布可视化 ---
    print("\n# --- 2.4.1 类别分布可视化---")
    plt.figure(figsize=(18, 6)); 
    plt.bar(df_combined['类别名'], df_combined['训练集数量'], color='skyblue')
    plt.xticks(rotation=90, fontsize=10)
    plt.title('训练集各类别样本数量分布 (类别不均衡性)', fontsize=16)
    plt.ylabel('样本数量', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '1_Category_Distribution.png')); 
    plt.close()
    print("✅ 类别分布柱状图已保存为：1_Category_Distribution.png")

    # --- 2.4.2 随机图像可视化 ---
    print("\n# --- 2.4.2 随机图像可视化 ---")
    visualize_random_samples(TRAIN_DATA_PATH, num_samples_per_class=1)
    print("✅ 随机样本可视化图已保存为：2_Random_Samples.png")

    # --- 2.4.3 稀疏类别对比图 (Top N Least Frequent Classes) ---
    print("\n# --- 2.4.3 稀疏类别对比 (Top 10 Least Frequent) ---")

    # 1. 选取训练集样本数量最少的前 10 个类别
    df_sparse = df_combined.sort_values(by='训练集数量', ascending=True).head(10)

    plt.figure(figsize=(12, 6))
    # 绘制训练集和测试集的样本数量对比
    plt.bar(df_sparse['类别名'], df_sparse['训练集数量'], label='训练集样本数', color='darkorange')
    plt.bar(df_sparse['类别名'], df_sparse['测试集数量'], label='测试集样本数', color='darkblue', alpha=0.6)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title('训练集样本数最少的前10个类别及其测试集分布', fontsize=14)
    plt.ylabel('样本数量', fontsize=12)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, '3_Sparse_Class_Comparison.png'))
    plt.close()
    print("✅ 稀疏类别对比图已保存为：3_Sparse_Class_Comparison.png")

    return standard_size


# === 阶段 3：特征工程===
# --- 辅助函数：图像数据增强 ---
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

# --- 3.1.1 图像预处理函数 ---
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

# --- 3.1.2 特征提取：HOG 特征提取 ---
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

# --- 3.1.3 构建特征矩阵 (已集成数据增强/过采样逻辑) ---
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

# --- 阶段 3 总体执行函数 (最终修复版) ---
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
    
    return X_train_augmented, y_train, X_test, y_test, X_train_original, y_train_original


# --- 阶段4 辅助函数：多分类模型报告 ---
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

# --- 阶段 4 总体执行函数 (包含 4.2.1-4.2.3) ---
def run_model_training(X_train, y_train, X_test, y_test):
    
    print("\n\n# --- 阶段 4：模型训练与性能分析 ---")
    
    # --- 4.1 特征降维 (PCA) ---
    print("\n# --- 4.1 特征降维 (PCA) ---")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    target_dim = 512
    pca = PCA(n_components=target_dim, random_state=42)
    
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"  > 降维后维度: {X_train_pca.shape[1]}")
    
    # --- 4.2 模型训练与配置 ---
    models_to_run = [
        ("4.2.1 SVM Classifier (RBF)", SVC(kernel='rbf', C=10, gamma='scale', probability=False, verbose=False)), 
        ("4.2.2 Logistic Regression", LogisticRegression(max_iter=1000, multi_class='multinomial')), 
        ("4.2.3 Random Forest", RandomForestClassifier(n_estimators=100)),
    ]

    report_list = []
    trained_models = {} # 用于存储训练好的模型对象

    for name, model in models_to_run:
        report_df = get_model_report(model, X_train_pca, X_test_pca, y_train, y_test, name)
        report_list.append(report_df)
        trained_models[name] = model # 存储模型对象

    # 拼接表
    model_performances = pd.concat(report_list, axis=0).reset_index(drop=True)
    
    print("\n== 4.2 模型性能对比表 (数值) ==")
    print(model_performances)
    
    # --- 4.3 性能分析与最终分类报告 ---
    print("\n# --- 4.3 性能分析与最终分类报告 ---")

    # 找出 Macro F1 最佳的模型
    df_temp = model_performances.copy()
    df_temp['Macro_F1_val'] = df_temp['Macro_F1'].apply(lambda x: float(x.replace('<', '').strip()))
    best_row = df_temp.loc[df_temp['Macro_F1_val'].idxmax()]
    best_model_name = best_row['Model']
    
    # 获取最佳模型的预测结果并打印详细报告
    final_best_model = trained_models[best_model_name]
    y_pred_final = final_best_model.predict(X_test_pca)
    
    print(f"\n== 最佳模型详细分类报告: {best_model_name} ==")
    class_names = [f'Class {i}' for i in range(27)] 
    print(classification_report(y_test, y_pred_final, digits=3, zero_division=0))

    print("✅ 阶段 4 训练和数值评估完成。结果已计算，转入阶段 5 进行图形化展示。")
    
    # 返回所有必要的数据，包括 X_test_pca, y_test 和性能表
    return model_performances, final_best_model, X_test_pca, y_test

# --- 阶段 5：纯图形化可视化 ---
def run_results_visualization(model_performances: pd.DataFrame, final_best_model: SVC, X_test_pca: np.ndarray, y_test: np.ndarray):
    
    print("\n\n# --- 阶段 5：结果可视化 ---")
    
    # --- 5.1 性能对比图 (Macro F1) ---
    print("\n# --- 5.1 性能对比图 (Macro F1) ---")
    
    # 逻辑：直接绘制模型性能表格中的 Macro F1 数据
    
    # 数据转换：确保 Macro_F1 是数值类型
    df = model_performances.copy()
    df['Macro_F1_val'] = df['Macro_F1'].apply(lambda x: float(x.replace('<', '').strip()))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Macro_F1_val'], color=['skyblue', 'salmon', 'lightgreen'])
    plt.title('ML 模型性能对比: Macro F1-Score (最终)', fontsize=14)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    
    best_macro_f1 = df['Macro_F1_val'].max()
    plt.axhline(y=best_macro_f1, color='r', linestyle='--', label=f'Best F1: {best_macro_f1:.4f}')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '5.1_ML_Model_Comparison_F1.png'))
    plt.close()
    print("✅ 5.1 模型性能对比图已保存为：5.1_ML_Model_Comparison_F1.png")
    
    # --- 5.2 最佳模型的混淆矩阵 (Confusion Matrix) ---
    print("\n# --- 5.2 混淆矩阵 (Confusion Matrix) ---")

    y_pred_final = final_best_model.predict(X_test_pca)

    # 设置图形大小
    plt.figure(figsize=(16, 12))

    disp = ConfusionMatrixDisplay.from_estimator(
        final_best_model,
        X_test_pca,
        y_test,
        cmap=plt.cm.Blues,
        # label的数量太多，只用数字作为标签名，简化图表
        display_labels=[str(i) for i in range(27)], 
        xticks_rotation='vertical'
    )

    # 关键修改：调整颜色条的高度与矩阵一致
    disp.im_.set_clim(0, None)  # 可选：设置颜色范围
    plt.colorbar(disp.im_, ax=disp.ax_, fraction=0.046, pad=0.04)

    disp.ax_.set_title('最佳模型混淆矩阵 (仅显示类别索引)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '5.2_Confusion_Matrix.png'))
    plt.close()

    # 打印最容易混淆的类别（手动提示）
    print("✅ 5.2 最佳模型混淆矩阵图已保存为：5.2_Confusion_Matrix.png")

    # --- 5.3 中间结果可视化 (PCA 反向投影) ---
    print("\n# --- 5.3 中间结果可视化 ---")
    print("\n✅ 中间输出简化：可视化最主要的 2 个主成分 (PCA 2D)")
    
    pca_final = PCA(n_components=2, random_state=42)
    X_test_2d = pca_final.fit_transform(X_test_pca) # 对 512 维特征降到 2 维
    
    # 获取所有标签
    num_classes = 27
    cmap = plt.cm.get_cmap('gist_ncar', num_classes) # 使用更清晰的色图
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        X_test_2d[:, 0], 
        X_test_2d[:, 1], 
        c=y_test, 
        cmap=cmap, 
        s=40
    )
    
    plt.colorbar(scatter, ticks=np.arange(num_classes))
    plt.title('HOG + PCA 特征分布 (降维至 2D)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '5.3_PCA_2D_Distribution.png'))
    plt.close()
    print("✅ 5.3 两个主要主成分分布图已保存为：7_PCA_2D_Distribution.png")

# --- 阶段 6：对比分析与消融实验  ---
def run_svm_c_ablation(X_train_aug, y_train_aug, X_test, y_test, X_train_orig, y_train_orig):
    """
    执行 SVM C 参数的敏感性分析消融实验 (6.2.1 - 6.2.3) 和 数据增强对比 (6.2.1-6.2.2)。
    """
    print("\n\n# --- 阶段 6：SVM 超参数和策略敏感性分析 ---")
    
    # --- 1. 定义消融实验列表 ---
    
    ablation_runs = []
    
    # A. C 参数对比 (基于最佳增强数据 X_train_aug)
    print("\n# --- A. 6.1 SVM C 参数敏感性对比 ---")
    ablation_runs.extend([
        {"C_Value": 1,   "Name": "6.1.1 SVM C=1 (高正则化)",    "X_train": X_train_aug, "y_train": y_train_aug},
        {"C_Value": 10,  "Name": "6.1.2 SVM C=10 (基准)",       "X_train": X_train_aug, "y_train": y_train_aug},
        {"C_Value": 100, "Name": "6.1.3 SVM C=100 (低正则化)",  "X_train": X_train_aug, "y_train": y_train_aug},
    ])
    
    # B. 数据增强策略对比 (使用 Original 和 Augmented)
    print("\n# --- B. 6.2 数据增强有效性对比 (SVM C=10) ---")

    ablation_runs.extend([
        # 6.2.1 无增强：使用原始 X_train_orig (2255 样本)
        {"C_Value": 10, "Name": "6.2.1 无数据增强 (Original)", "X_train": X_train_orig, "y_train": y_train_orig}, 
        # 6.2.2 有增强：使用 X_train_aug (2850 样本)
        {"C_Value": 10, "Name": "6.2.2 有数据增强 (Augmented)", "X_train": X_train_aug, "y_train": y_train_aug},   
    ])

    final_ablation_list = []

    # --- 2 运行消融实验 ---
    
    for run in ablation_runs:
        # 针对每个 X_train 重复 PCA/Scaler，保证数据处理的严格独立性
        scaler = StandardScaler()
        pca = PCA(n_components=512, random_state=42)
        
        # 每次 fit 都会在正确的 X_train 上执行
        X_train_scaled = scaler.fit_transform(run["X_train"]) 
        X_test_scaled = scaler.transform(X_test)
        
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled) # 测试集不变
        
        model = SVC(kernel='rbf', C=run["C_Value"], gamma='scale', 
                    class_weight='balanced', random_state=42, verbose=False)
        
        # 运行并收集报告
        # 注意：此处必须使用 run["y_train"] 来对应当前的训练集 (Original/Augmented)
        report_df = get_model_report(model, X_train_pca, X_test_pca, run["y_train"], y_test, run["Name"])
        
        final_ablation_list.append(report_df)
        
    # --- 3.结果整合与可视化 ---
    
    ablation_performance = pd.concat(final_ablation_list, axis=0).reset_index(drop=True)

    print("\n== 阶段 6 消融实验结果表 (SVM C/数据增强敏感性) ==")
    print(ablation_performance)
    
    # --- 可视化：消融实验对比图 ---
    
    df_plot = ablation_performance.copy()
    df_plot['Accuracy_val'] = df_plot['Accuracy'].apply(lambda x: float(x.strip()))

    # 使用 'Model' 列 (其中包含实验的 Name) 作为 X 轴标签
    plt.figure(figsize=(10, 6))
    
    # 使用 5 种颜色，对应 5 个实验 (3个C值 + 2个增强对比)
    colors_for_plot = ['#A1C9F4', '#FFB482', '#A6DFE8', '#9FE6C0', '#FFE6AA'] 
    
    bars = plt.bar(df_plot['Model'], df_plot['Accuracy_val'], color=colors_for_plot[:len(df_plot)])
    
    plt.title('消融实验: SVM 超参数与数据增强对 Accuracy 的影响', fontsize=14)
    plt.ylabel('Accuracy-Score', fontsize=12)
    # 旋转 X 轴标签以适应长的实验名称
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    for bar in bars:
        yval = bar.get_height()
        # 将文本放在柱子顶端
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '6_Ablation_SVM_C_Parameter.png'))
    plt.close()
    
    print("✅ 阶段 6 消融对比分析图已保存为：6_Ablation_SVM_C_Parameter.png")
    print("\n✅ 阶段 6 SVM C 参数敏感性分析完成。")
    
    # 导出 CSV
    ablation_performance.to_csv(os.path.join(BASE_DIR, '6_SVM_Ablation_Results.csv'), index=False, encoding='utf-8')

    return ablation_performance


if __name__ == "__main__":
    
    print("================== 阶段 1/2：探索性数据分析 (EDA) ==================")
    final_input_size = run_data_loading_and_eda()
    
    print("\n\n================== 阶段 3：数据准备 (特征工程与建模准备) ==================")
    X_train, y_train, X_test, y_test, X_train_original, y_train_original = run_feature_engineering(final_input_size)
    
    print("\n\n================== 阶段 4：模型训练与评估 ==================")
    
    model_performances, final_best_model, X_test_pca, y_test = run_model_training(X_train, y_train, X_test, y_test) 
    
    print("\n\n================== 阶段 5：结果可视化与最终分析 ==================")
    
    run_results_visualization(model_performances, final_best_model, X_test_pca, y_test)

    print("\n\n================== 阶段 6：消融实验 ==================")
    
    svm_ablation_results = run_svm_c_ablation(X_train, y_train, X_test, y_test, X_train_original, y_train_original)
