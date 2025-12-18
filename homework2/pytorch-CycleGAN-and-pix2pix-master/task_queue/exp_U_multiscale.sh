#!/bin/bash
echo "启动 Exp U (Multi-Scale Discriminator)"
# 核心参数: --netD multiscale
# 目的: 使用 3 个尺度的判别器 (原图, 1/2, 1/4) 同时优化全局结构和局部纹理
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expU_multiscale \
    --model pix2pix \
    --direction BtoA \
    --netD multiscale \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expU_multiscale.log 2>&1
