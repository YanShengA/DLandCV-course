#!/bin/bash
echo "启动 Exp N (Attention GAN)"
# 核心参数: --use_attention
# 目的: 在 U-Net 瓶颈层加入自注意力模块 (SAGAN)，增强全局上下文理解能力
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expN_attention \
    --model pix2pix \
    --direction BtoA \
    --use_attention \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expN_attention.log 2>&1
