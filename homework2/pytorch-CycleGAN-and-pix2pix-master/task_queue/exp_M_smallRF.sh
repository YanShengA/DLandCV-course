#!/bin/bash
echo "启动 Exp M (Small RF, UNet-128)"
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expM_smallRF \
    --model pix2pix \
    --direction BtoA \
    --netG unet_128 \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expM_smallRF.log 2>&1
