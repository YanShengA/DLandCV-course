#!/bin/bash
echo "启动 Exp S (SSIM Loss)"
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expS_ssim \
    --model pix2pix \
    --direction BtoA \
    --use_ssim \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expS_ssim.log 2>&1
