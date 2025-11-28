# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mmengine
from mmengine.config import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from mmpretrain.evaluation import ConfusionMatrix
from mmpretrain.registry import DATASETS
from mmpretrain.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval a checkpoint and draw the confusion matrix.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'ckpt_or_result',
        type=str,
        help='The checkpoint file (.pth) or '
        'dumpped predictions pickle file (.pkl).')
    parser.add_argument('--out', help='the file to save the confusion matrix.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the metric result by matplotlib if supports.')
    parser.add_argument(
        '--show-path', type=str, help='Path to save the visualization image.')
    parser.add_argument(
        '--include-values',
        action='store_true',
        help='To draw the values in the figure.')
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='The color map to use. Defaults to "viridis".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    args = parser.parse_args()
    return args


def plot_confusion_matrix(cm, classes, show=True, show_path=None, include_values=True, cmap='viridis'):
    """
    优化的混淆矩阵绘图函数
    特点：保持官方风格，但修复文字重叠和空白过多的问题
    """
    num_classes = len(classes)
    
    # 策略：根据类别数设定一个合理的固定比例，而不是无限放大
    # 15x15 英寸对于30个类别通常是比较紧凑且清晰的
    figsize = (15, 15)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制矩阵
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # --- 修复空白过多问题 (1/2) ---
    # 使用 make_axes_locatable 强制让 colorbar 高度与矩阵图完全一致
    # 这样可以避免右侧出现巨大的空白条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.2)
    plt.colorbar(im, cax=cax)

    # 设置坐标轴刻度
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    # --- 修复字体重叠问题 ---
    # 1. fontsize=10: 保证字体适中
    # 2. rotation=45: 倾斜角度
    # 3. ha="right": 关键！右对齐，让文字尾部对齐刻度，防止左侧文字盖住右侧
    # 4. rotation_mode="anchor": 保证旋转中心正确
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # 将 x 轴标签移到底部（原本可能在顶部导致上方留白太多）
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # 绘制数值
    if include_values:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            val = cm[i, j]
            # 这里如果不希望显示0，可以加 if val > 0:
            if val >= 0: 
                ax.text(j, i, format(val, 'd'),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if val > thresh else "black",
                        fontsize=8) # 数值字体稍微小一点，避免格子爆满

    ax.set_ylabel('True label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, pad=15)

    # --- 修复空白过多问题 (2/2) ---
    # 紧凑布局，自动切除多余白边
    plt.tight_layout()
    
    if show_path is not None:
        # bbox_inches='tight' 再次确保保存时不留白边
        plt.savefig(show_path, dpi=200, bbox_inches='tight')
        print(f'The confusion matrix is saved at {show_path}.')
    
    if show:
        plt.show()


def main():
    args = parse_args()

    # register all modules
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    classes = None
    cm = None

    if args.ckpt_or_result.endswith('.pth'):
        cfg.test_evaluator = dict(type='ConfusionMatrix')
        cfg.load_from = str(args.ckpt_or_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.work_dir = tmpdir
            runner = Runner.from_cfg(cfg)
            classes = runner.test_loop.dataloader.dataset.metainfo.get('classes')
            cm = runner.test()['confusion_matrix/result']
    else:
        predictions = mmengine.load(args.ckpt_or_result)
        evaluator = Evaluator(ConfusionMatrix())
        metrics = evaluator.offline_evaluate(predictions, None)
        cm = metrics['confusion_matrix/result']
        try:
            dataset = DATASETS.build({**cfg.test_dataloader.dataset, 'pipeline': []})
            classes = dataset.metainfo.get('classes')
        except Exception:
            classes = None

    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    if args.out is not None:
        mmengine.dump(cm, args.out)

    if args.show or args.show_path is not None:
        plot_confusion_matrix(
            cm,
            classes=classes,
            show=args.show,
            show_path=args.show_path,
            include_values=args.include_values,
            cmap=args.cmap
        )


if __name__ == '__main__':
    main()