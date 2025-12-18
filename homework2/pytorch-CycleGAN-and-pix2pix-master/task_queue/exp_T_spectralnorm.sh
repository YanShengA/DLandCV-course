#!/bin/bash
echo "启动 Exp T (Spectral Norm)"
# 核心参数: --use_spectral_norm
# 目的: 使用谱归一化稳定判别器训练
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expT_spectralnorm \
    --model pix2pix \
    --direction BtoA \
    --use_spectral_norm \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expT_spectralnorm.log 2>&1
