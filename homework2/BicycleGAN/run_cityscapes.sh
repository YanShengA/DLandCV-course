#!/bin/bash
# 这个脚本严格模仿官方 train.sh 的逻辑，并为 cityscapes 数据集进行了配置。
set -ex

# --- 配置区 ---
# 1. 数据集名称 (固定为 cityscapes)
CLASS='cityscapes'

# 2. 模型名称 (根据官方脚本和您的文件名)
MODEL='bicycle_gan'

# 3. GPU ID (通过命令行第一个参数传入，例如: ./run_cityscapes.sh 0)
GPU_ID=${1}
if [ -z "$GPU_ID" ]; then
    echo "错误: 请提供 GPU ID 作为第一个参数。"
    echo "用法: ./run_cityscapes.sh <gpu_id>"
    exit 1
fi

# --- Cityscapes 专属参数设置 (参考官方脚本对类似数据集的配置) ---
DIRECTION='BtoA'          # 任务是 Label -> Photo
LOAD_SIZE=286             # 先放大到 286
CROP_SIZE=256             # 再裁剪到 256
INPUT_NC=3                # Cityscapes 的标签是 3 通道彩色图
NITER=100                 # 初始学习率的 epoch 数 (参考 facades/maps)
NITER_DECAY=100           # 学习率衰减的 epoch 数 (参考 facades/maps)
NZ=8                      # 潜在空间 z 的维度 (来自官方脚本)
SAVE_EPOCH=25             # 每 25 个 epoch 保存一次模型 (参考 facades/maps)
NO_FLIP=''                # Cityscapes 数据集可以进行水平翻转增强，所以留空


# --- 自动生成实验名称 ---
DATE=`date '+%d_%m_%Y_%H%M'`
CHECKPOINTS_DIR=./checkpoints
NAME=${CLASS}_${MODEL}_${DATE}

# --- 最终执行指令 (严格参照官方脚本) ---
# 使用 CUDA_VISIBLE_DEVICES 来隔离 GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./train.py \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --nz ${NZ} \
  --save_epoch_freq ${SAVE_EPOCH} \
  ${NO_FLIP} \
  --use_dropout