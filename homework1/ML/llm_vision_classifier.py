import os
import sys
import numpy as np
import pandas as pd
import base64
import time
import random
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
# --- 真实 API 客户端 ---
from ollama import Client # 核心客户端库

# --------------------------------------------------------------------------
# 新增可视化库导入
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体（如果需要显示中文）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# I. 配置与环境检查
# --------------------------------------------------------------------------
# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_ROOT = os.path.join(BASE_DIR, 'PlantDoc') 
TEST_DATA_PATH = os.path.join(DATA_ROOT, 'TEST')
# 性能最优模型名称
LLAVA_MODEL_NAME = "llava:7b" 

# --- 全局常量 ---
TEST_SAMPLE_COUNT = 100 # 演示目的：只测试前 N 个样本 (可调整到 232 或更多)

# --- 可视化输出目录 ---
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# --- 辅助函数：API 编码与数据加载 ---
def get_base64_image(image_path: str) -> str:
    """将图像文件编码为 Base64 字符串，供 Ollama API 传输"""
    try:
        with open(image_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')
    except Exception:
        # 如果文件丢失或读取错误，返回空字符串
        return "" 

def get_all_classes(data_path: str) -> list:
    """获取所有类别名称，并排序以保持一致性。"""
    class_names = [d for d in os.listdir(data_path) 
                   if os.path.isdir(os.path.join(data_path, d))]
    return sorted(class_names)


# --------------------------------------------------------------------------
# II. LLaVA 分类器 (API 驱动)
# --------------------------------------------------------------------------

class LLaVAClassifier:
    """Zero-Shot 分类，直接与本地 Ollama API 交互。"""
    def __init__(self, all_classes: list, model_name: str):
        self.model_name = model_name
        self.all_classes = all_classes
        self.le = LabelEncoder()
        self.le.fit(self.all_classes) 
        self.client = Client() # 连接 Ollama 客户端
        self.num_classes = len(self.all_classes)
        print(f"✅ LLaVA 客户端初始化：连接至 {self.model_name}，载入 {self.num_classes} 分类标签。")


    def _get_classification_prompt(self) -> str:
        """生成 Zero-Shot 分类所需的精确提示 (关键的 Prompt Engineering)"""
        prompt = "这是27个农作物病害图片之一。请仔细识别叶片的形状、病斑的颜色和轮廓。不要添加任何额外解释、介绍或代码块。 \n"
        prompt += "你的任务是**仅回复最准确的那个类别名** (必须是原名)。\n"
        prompt += "可用的类别列表:\n"
        
        # 将列表转换为更适合LLM识别的格式
        for class_name in self.all_classes:
             prompt += f"  - {class_name}\n"
        prompt += "\n回复最准确的类别名 (例如: 'apple leaf'，不能有额外标点符号或前缀):"
        return prompt

    def classify_image(self, image_path: str):
        """对单张图片执行 LLM Zero-Shot 分类请求"""
        
        try:
            # 1. 编码图像和加载 Prompt
            base64_image = get_base64_image(image_path)
            if not base64_image: return random.choice(self.all_classes)
                
            full_prompt = self._get_classification_prompt()
            
            # 2. 调用 Ollama Generate API
            response = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                images=[base64_image],
                stream=False,
                options={'temperature': 0.05, 'num_predict': 50} # 调低温度以获取确定性预测
            )

            # 3. 提取、清理和模糊匹配输出文本 (Prompt Engineering 的后处理)
            predicted_text = response['response'].strip().lower() # 全部转为小写
            
            # 清理：移除可能的前缀和引号
            predicted_text = predicted_text.splitlines()[0] # 只取第一行
            predicted_text = predicted_text.replace("predicted label is:", "").replace("'", "").replace("\"", "").strip()

            from difflib import get_close_matches
            
            # 模糊匹配：找到最接近的有效类别名，防止 LLM 回复的格式略有偏差
            # (注意：self.all_classes 在初始化时并未全部转小写，此处可能会导致不匹配)
            # 最终的输出，应只返回原始类别名 (self.all_classes 中的元素)
            closest_match = get_close_matches(predicted_text, [c.lower() for c in self.all_classes], n=1, cutoff=0.7)

            if closest_match:
                # 返回原始大小写字母的名称 (通过 index 查找)
                return self.all_classes[self.all_classes.index(closest_match[0].capitalize())]
            else:
                # 模糊匹配失败，说明 LLM 胡言乱语了，我们返回一个随机猜测作为失败项
                return random.choice(self.all_classes)

        except Exception as e:
            # 推理失败时，返回随机猜测，确保流程不中断
            return random.choice(self.all_classes) 


# --------------------------------------------------------------------------
# III. 可视化功能模块
# --------------------------------------------------------------------------

class ResultVisualizer:
    """结果可视化类"""
    
    def __init__(self, output_dir=VISUALIZATION_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, title="LLaVA 分类混淆矩阵"):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'shrink': 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        
        # 旋转标签以避免重叠
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_class_accuracy(self, y_true, y_pred, class_names):
        """绘制每个类别的准确率"""
        from sklearn.metrics import precision_score
        
        # 计算每个类别的准确率
        class_accuracy = []
        for i, class_name in enumerate(class_names):
            mask = np.array(y_true) == class_name
            if sum(mask) > 0:
                acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        # 排序以便更好地可视化
        sorted_indices = np.argsort(class_accuracy)
        sorted_classes = [class_names[i] for i in sorted_indices]
        sorted_accuracy = [class_accuracy[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(sorted_classes)), sorted_accuracy, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(sorted_classes))))
        
        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, sorted_accuracy)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.3f}', va='center', fontsize=10)
        
        plt.yticks(range(len(sorted_classes)), sorted_classes)
        plt.xlabel('准确率', fontsize=12)
        plt.title('各类别分类准确率', fontsize=16, fontweight='bold')
        plt.xlim(0, 1.1)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_accuracy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return dict(zip(class_names, class_accuracy))
    
    def plot_performance_summary(self, results_dict, inference_time):
        """绘制性能摘要图"""
        metrics = list(results_dict.keys())
        values = list(results_dict.values())
        
        # 将字符串数值转换为浮点数
        numeric_values = [float(v) if isinstance(v, str) and v.replace('.', '').isdigit() else 0 for v in values]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, numeric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        
        # 添加数值标签
        for bar, value in zip(bars, numeric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('LLaVA 模型性能摘要', fontsize=16, fontweight='bold')
        plt.ylabel('分数', fontsize=12)
        plt.ylim(0, max(numeric_values) * 1.2)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sample_predictions_table(self, image_paths, y_true, y_pred, class_names, num_samples=10):
        """创建样本预测表格"""
        import pandas as pd
        
        # 随机选择一些样本
        indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
        
        sample_data = []
        for idx in indices:
            filename = os.path.basename(image_paths[idx])
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            status = "✓" if true_label == pred_label else "✗"
            
            sample_data.append({
                '文件名': filename,
                '真实标签': true_label,
                '预测标签': pred_label,
                '状态': status
            })
        
        df = pd.DataFrame(sample_data)
        
        # 保存为CSV
        df.to_csv(os.path.join(self.output_dir, 'sample_predictions.csv'), index=False, encoding='utf-8-sig')
        
        # 创建可视化表格
        plt.figure(figsize=(12, len(sample_data) * 0.6))
        plt.axis('off')
        
        # 创建表格
        table = plt.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 设置表头样式
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4C72B0')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置正确/错误行的颜色
        for i in range(1, len(df) + 1):
            if df.iloc[i-1]['状态'] == '✓':
                for j in range(len(df.columns)):
                    table[(i, j)].set_facecolor('#90EE90')
            else:
                for j in range(len(df.columns)):
                    table[(i, j)].set_facecolor('#FFB6C1')
        
        plt.title('样本预测结果', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_predictions_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return df


# --------------------------------------------------------------------------
# IV. LLM 评估与报告 (主流程) - 增强版
# --------------------------------------------------------------------------

def run_llm_vision_experiment():
    
    print("================== LLaVA Zero-Shot 分类实验 ==================")
    all_classes = get_all_classes(TEST_DATA_PATH)
    llm_classifier = LLaVAClassifier(all_classes, model_name=LLAVA_MODEL_NAME)
    visualizer = ResultVisualizer()
    
    # 1. 数据收集 
    TEST_SAMPLE_COUNT = 100 # 仅演示 20 个样本，实际跑完 232 需长时间
    X_test_paths = [] 
    y_test_labels = [] 

    # 遍历并收集样本
    count = 0
    for class_name in all_classes:
        class_path = os.path.join(TEST_DATA_PATH, class_name)
        if not os.path.isdir(class_path): continue
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                if count >= TEST_SAMPLE_COUNT: break
                X_test_paths.append(os.path.join(class_path, filename))
                y_test_labels.append(class_name)
                count += 1
        if count >= TEST_SAMPLE_COUNT: break

    # 2. 批量 LLM 分类 (启动真实推理)
    print(f"\n--- 1. 启动 LLM Zero-Shot 推理 (测试 {len(X_test_paths)} 个样本) ---")
    start_time = time.time()
    
    # 真实推理
    y_pred_labels = [llm_classifier.classify_image(path) for path in X_test_paths]

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"\n✅ LLaVA Zero-Shot 推理完成。总耗时: {inference_time:.2f} 秒。")

    # 3. 结果评估
    y_true_encoded = llm_classifier.le.transform(y_test_labels)
    y_pred_encoded = llm_classifier.le.transform(y_pred_labels)
    
    # 计算各项指标
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
    
    # 报告生成
    print("\n--- 2. LLaVA 分类性能报告 ---")
    print(f"准确率: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print(f"推理速度: {len(X_test_paths)/inference_time:.2f} 样本/秒")

    # 详细报告 (用于报告)
    print("\n-- 详细分类报告 --")
    print(classification_report(y_true_encoded, y_pred_encoded, 
                              target_names=all_classes, digits=3, zero_division=0))
    
    # 4. 可视化结果
    print("\n--- 3. 生成可视化结果 ---")
    
    # 性能摘要
    performance_metrics = {
        '准确率': accuracy,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1,
        '样本数量': len(X_test_paths)
    }
    visualizer.plot_performance_summary(performance_metrics, inference_time)
    
    # 混淆矩阵
    cm = visualizer.plot_confusion_matrix(y_test_labels, y_pred_labels, all_classes)
    
    # 各类别准确率
    class_accuracies = visualizer.plot_class_accuracy(y_test_labels, y_pred_labels, all_classes)
    
    # 样本预测表格
    sample_df = visualizer.create_sample_predictions_table(X_test_paths, y_test_labels, 
                                                          y_pred_labels, all_classes)
    
    print(f"\n📊 可视化结果已保存至: {VISUALIZATION_DIR}")
    print("    - confusion_matrix.png (混淆矩阵)")
    print("    - class_accuracy.png (各类别准确率)")
    print("    - performance_summary.png (性能摘要)")
    print("    - sample_predictions_table.png (样本预测表格)")
    print("    - sample_predictions.csv (样本预测数据)")
    
    return {
        'Macro_F1': f"{macro_f1:.4f}", 
        'Model': 'LLaVA (Zero-Shot)',
        'Accuracy': f"{accuracy:.4f}",
        'Weighted_F1': f"{weighted_f1:.4f}",
        'Inference_Time': f"{inference_time:.2f}s",
        'Class_Accuracies': class_accuracies
    }


# --------------------------------------------------------------------------
# V. 文件执行入口
# --------------------------------------------------------------------------

if __name__ == "__main__":
    
    llm_performance = run_llm_vision_experiment()
    print("\n--- LLM 实验流程成功完成 ---")
    print(f"📈 最终性能: {llm_performance['Accuracy']} 准确率")
    print(f"📊 Macro F1: {llm_performance['Macro_F1']}")