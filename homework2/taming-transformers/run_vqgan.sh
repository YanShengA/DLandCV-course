#!/bin/bash
set -ex

GPU_ID=${1}
# ... (和之前一样)

# 训练 VQGAN 模型
python main.py \
    --base configs/cityscapes_vqgan.yaml \
    -t True \
    --gpus $GPU_ID,