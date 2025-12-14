# eda_analysis.py
"""
数据探索性分析 (EDA) 模块
功能：对PlantDoc数据集进行全面的探索性分析
包括：类别分布、图像尺寸分析、损坏文件检查、色彩分析等
"""

import os
import sys
from collections import defaultdict
import csv
import random
import numpy as np
from PIL import Image, ImageStat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Matplotlib 中文显示配置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ----------------- 路径配置 -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'PlantDoc')
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'TRAIN')
TEST_DATA_PATH = os.path.join(DATA_ROOT, 'TEST')


# ================= 2.3.1 类别分布可视化 =================

def get_file_counts(data_path: str) -> dict:
    """与原文件完全一致 - 统计每个类别的图像数量"""
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


def visualize_category_distribution(df_combined: pd.DataFrame):
    """与原文件一致 - 可视化类别分布"""
    print("\n# --- 2.3.1 类别分布可视化 ---")
    
    # 训练集类别分布柱状图
    plt.figure(figsize=(18, 6))
    plt.bar(df_combined['类别名'], df_combined['训练集数量'], color='skyblue')
    plt.xticks(rotation=90, fontsize=10)
    plt.title('训练集各类别样本数量分布 (类别不均衡性)', fontsize=16)
    plt.ylabel('样本数量', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '1_Category_Distribution.png'))
    plt.close()
    print("✅ 类别分布柱状图已保存为：1_Category_Distribution.png")
    
    # 稀疏类别对比图
    df_sparse = df_combined.sort_values(by='训练集数量', ascending=True).head(10)
    plt.figure(figsize=(12, 6))
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


def visualize_random_samples(data_path, num_samples_per_class=1):
    """与原文件完全一致 - 随机样本可视化"""
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    num_classes = len(class_dirs)
    
    grid_size = int(np.ceil(np.sqrt(num_classes)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_dirs):
        if i >= len(axes):
            break
            
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if image_files:
            random_file = random.choice(image_files)
            try:
                img = Image.open(os.path.join(class_path, random_file))
                axes[i].imshow(img)
                axes[i].set_title(class_name.replace(' ', '\n'), fontsize=10)
                axes[i].axis('off')
                img.close()
            except Exception as e:
                axes[i].set_title(f"{class_name}\n(加载失败)", fontsize=10)
                axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '2_Random_Samples.png'))
    plt.close()
    print("✅ 随机样本可视化图已保存为：2_Random_Samples.png")


# ================= 2.3.2 图像尺寸与长宽比分析 =================

def analyze_image_properties(data_path: str, set_name: str) -> tuple[dict, int]:
    """与原文件完全一致 - 分析图像尺寸和检查损坏文件"""
    image_sizes = defaultdict(int)
    damaged_files = []
    total_images_checked = 0

    if not os.path.exists(data_path):
        return image_sizes, 0

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


def analyze_image_size_and_ratio(train_sizes: dict, test_sizes: dict):
    """与原文件一致 - 综合分析图像尺寸和长宽比"""
    print("\n# --- 2.3.2 图像尺寸与长宽比分析 ---")
    
    # 合并训练集和测试集的尺寸统计
    all_sizes = train_sizes.copy()
    for size, count in test_sizes.items():
        all_sizes[size] += count
    sorted_all_sizes = sorted(all_sizes.items(), key=lambda item: item[1], reverse=True)
    
    # 建议标准输入尺寸
    standard_size = (256, 256)
    if sorted_all_sizes:
        most_common_size = sorted_all_sizes[0][0]
        standard_size = (512, 512) if most_common_size[0] > 256 or most_common_size[1] > 256 else (256, 256)
        print(f"\n✅ 建议的模型标准输入尺寸为： {standard_size[0]}x{standard_size[1]}")
    else:
        print(f"\n✅ 无法确定最常见尺寸，使用默认输入尺寸： {standard_size[0]}x{standard_size[1]}")
    
    # 长宽比分析
    ratios = {round(w/h, 2) for w, h in all_sizes.keys()}
    print(f"  > 发现的不同长宽比数量: {len(ratios)}")
    print(f"  > 长宽比集合: {sorted(list(ratios))[:5]} ... (仅显示前5个)")
    
    # 新增：尺寸分布可视化
    visualize_size_distribution(all_sizes)
    
    return standard_size


def visualize_size_distribution(all_sizes: dict):
    """新增功能 - 可视化图像尺寸分布"""
    if not all_sizes:
        return
        
    # 提取尺寸数据
    widths = [size[0] for size in all_sizes.keys() for _ in range(all_sizes[size])]
    heights = [size[1] for size in all_sizes.keys() for _ in range(all_sizes[size])]
    ratios = [round(w/h, 2) for w, h in zip(widths, heights)]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 宽度分布
    axes[0, 0].hist(widths, bins=30, color='lightblue', edgecolor='black')
    axes[0, 0].set_title('图像宽度分布')
    axes[0, 0].set_xlabel('宽度 (像素)')
    axes[0, 0].set_ylabel('频数')
    
    # 高度分布
    axes[0, 1].hist(heights, bins=30, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('图像高度分布')
    axes[0, 1].set_xlabel('高度 (像素)')
    axes[0, 1].set_ylabel('频数')
    
    # 长宽比分布
    axes[1, 0].hist(ratios, bins=30, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('图像长宽比分布')
    axes[1, 0].set_xlabel('长宽比')
    axes[1, 0].set_ylabel('频数')
    
    # 散点图：宽度 vs 高度
    axes[1, 1].scatter(widths, heights, alpha=0.6, color='purple')
    axes[1, 1].set_title('图像宽度 vs 高度')
    axes[1, 1].set_xlabel('宽度 (像素)')
    axes[1, 1].set_ylabel('高度 (像素)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '4_Image_Size_Distribution.png'))
    plt.close()
    print("✅ 图像尺寸分布图已保存为：4_Image_Size_Distribution.png")


# ================= 2.3.3 损坏文件检查 =================

def check_damaged_files():
    """与原文件一致 - 检查并报告损坏文件情况"""
    print("\n# --- 2.3.3 损坏文件检查 ---")
    
    train_sizes, train_damaged = analyze_image_properties(TRAIN_DATA_PATH, 'TRAIN')
    test_sizes, test_damaged = analyze_image_properties(TEST_DATA_PATH, 'TEST')
    
    if train_damaged or test_damaged:
        print(f"⚠️ 发现损坏文件！请检查 TRAIN_damaged_files.csv 和 TEST_damaged_files.csv")
    else:
        print("✅ 未发现损坏文件")
        
    return train_sizes, test_sizes, train_damaged, test_damaged


# ================= 新增分析功能 =================

def analyze_color_statistics(data_path: str, sample_per_class: int = 5):
    """
    新增功能 - 分析图像色彩统计特征
    """
    print("\n# --- 色彩统计分析 ---")
    
    color_stats = {
        'mean_r': [], 'mean_g': [], 'mean_b': [],
        'std_r': [], 'std_g': [], 'std_b': [],
        'class_name': []
    }
    
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        sampled_files = random.sample(image_files, min(sample_per_class, len(image_files)))
        
        for image_file in sampled_files:
            try:
                img = Image.open(os.path.join(class_path, image_file))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 计算RGB通道统计
                stat = ImageStat.Stat(img)
                mean_r, mean_g, mean_b = stat.mean
                std_r, std_g, std_b = stat.stddev
                
                color_stats['mean_r'].append(mean_r)
                color_stats['mean_g'].append(mean_g)
                color_stats['mean_b'].append(mean_b)
                color_stats['std_r'].append(std_r)
                color_stats['std_g'].append(std_g)
                color_stats['std_b'].append(std_b)
                color_stats['class_name'].append(class_name)
                
                img.close()
            except Exception as e:
                continue
    
    # 可视化色彩统计
    df_color = pd.DataFrame(color_stats)
    if not df_color.empty:
        visualize_color_statistics(df_color)
    
    return df_color


def visualize_color_statistics(df_color: pd.DataFrame):
    """可视化色彩统计特征"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RGB均值分布
    axes[0, 0].hist([df_color['mean_r'], df_color['mean_g'], df_color['mean_b']], 
                   bins=30, alpha=0.7, label=['Red', 'Green', 'Blue'])
    axes[0, 0].set_title('RGB通道均值分布')
    axes[0, 0].set_xlabel('像素值')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].legend()
    
    # RGB标准差分布
    axes[0, 1].hist([df_color['std_r'], df_color['std_g'], df_color['std_b']], 
                   bins=30, alpha=0.7, label=['Red', 'Green', 'Blue'])
    axes[0, 1].set_title('RGB通道标准差分布')
    axes[0, 1].set_xlabel('标准差')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].legend()
    
    # 散点图：R均值 vs G均值
    axes[1, 0].scatter(df_color['mean_r'], df_color['mean_g'], alpha=0.6)
    axes[1, 0].set_title('R通道均值 vs G通道均值')
    axes[1, 0].set_xlabel('R均值')
    axes[1, 0].set_ylabel('G均值')
    
    # 类别色彩特征热图（采样显示）
    if len(df_color) > 10:
        sample_df = df_color.groupby('class_name').mean().head(10)
        sns.heatmap(sample_df[['mean_r', 'mean_g', 'mean_b']], 
                   annot=True, fmt='.1f', ax=axes[1, 1], cmap='YlOrRd')
        axes[1, 1].set_title('各类别RGB均值热图 (前10类)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, '5_Color_Statistics.png'))
    plt.close()
    print("✅ 色彩统计分析图已保存为：5_Color_Statistics.png")


def analyze_dataset_balance(df_combined: pd.DataFrame):
    """
    新增功能 - 分析数据集平衡性
    """
    print("\n# --- 数据集平衡性分析 ---")
    
    # 计算不平衡系数
    train_counts = df_combined['训练集数量'].values
    imbalance_ratio = train_counts.max() / train_counts.min()
    
    print(f"训练集最大/最小类别样本比: {imbalance_ratio:.2f}")
    print(f"训练集样本标准差: {train_counts.std():.2f}")
    print(f"训练集样本变异系数: {train_counts.std() / train_counts.mean():.3f}")
    
    # 平衡性可视化
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(df_combined)), sorted(train_counts, reverse=True), 'o-')
    plt.title('训练集样本数量排序分布')
    plt.xlabel('类别排序')
    plt.ylabel('样本数量')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, '6_Dataset_Balance.png'))
    plt.close()
    print("✅ 数据集平衡性分析图已保存为：6_Dataset_Balance.png")


# ================= 主执行函数 =================

def run_eda_analysis():
    """
    执行完整的数据探索性分析流程
    """
    print("================== 数据探索性分析 (EDA) ==================")
    
    # 1. 数据集基本统计
    print("\n# --- 数据集基本统计 ---")
    train_counts = get_file_counts(TRAIN_DATA_PATH)
    test_counts = get_file_counts(TEST_DATA_PATH)
    
    print(f"训练集类别数: {len(train_counts)}, 总样本: {sum(train_counts.values())}")
    print(f"测试集类别数: {len(test_counts)}, 总样本: {sum(test_counts.values())}")
    
    # 创建综合统计表
    df_train = pd.DataFrame(train_counts.items(), columns=['类别名', '训练集数量'])
    df_test = pd.DataFrame(test_counts.items(), columns=['类别名', '测试集数量'])
    df_combined = pd.merge(df_train, df_test, on='类别名', how='outer').fillna(0)
    df_combined['训练集数量'] = df_combined['训练集数量'].astype(int)
    df_combined['总样本数量'] = df_combined['训练集数量'] + df_combined['测试集数量']
    df_combined['训练集占比 (%)'] = round((df_combined['训练集数量'] / df_combined['总样本数量']) * 100, 2)
    df_combined = df_combined.sort_values(by='总样本数量', ascending=False).reset_index(drop=True)

    print("\n== 详细分类统计表 ==")
    print(df_combined)
    df_combined.to_csv(os.path.join(BASE_DIR, 'dataset_distribution_stats.csv'), index=False, encoding='utf-8')
    print("✅ 详细统计已保存至：dataset_distribution_stats.csv")
    
    # 2.3.3 损坏文件检查
    train_sizes, test_sizes, train_damaged, test_damaged = check_damaged_files()
    
    # 2.3.2 图像尺寸与长宽比分析
    standard_size = analyze_image_size_and_ratio(train_sizes, test_sizes)
    
    # 2.3.1 类别分布可视化
    visualize_category_distribution(df_combined)
    visualize_random_samples(TRAIN_DATA_PATH)
    
    # 新增分析功能
    analyze_color_statistics(TRAIN_DATA_PATH)
    analyze_dataset_balance(df_combined)
    
    print("\n✅ EDA分析完成！")
    return standard_size, df_combined


if __name__ == "__main__":
    standard_size, df_stats = run_eda_analysis()
    print(f"\n建议的输入尺寸: {standard_size}")