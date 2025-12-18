#!/bin/bash
echo "启动 Exp Q (Large Kernel)"
# 核心参数: --use_large_kernel
# 目的: 使用 8x8 大卷积核增大感受野，模仿 ConvNeXt 设计
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expQ_largekernel \
    --model pix2pix \
    --direction BtoA \
    --use_large_kernel \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expQ_largekernel.log 2>&1
