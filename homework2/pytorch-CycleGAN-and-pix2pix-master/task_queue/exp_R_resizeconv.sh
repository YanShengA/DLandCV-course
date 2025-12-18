#!/bin/bash
echo "启动 Exp R (Resize-Conv)"
# 核心参数: --use_resize_conv
# 目的: 使用上采样代替反卷积，消除棋盘格伪影(Checkerboard Artifacts)
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expR_resizeconv \
    --model pix2pix \
    --direction BtoA \
    --use_resize_conv \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expR_resizeconv.log 2>&1
