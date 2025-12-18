#!/bin/bash
echo "启动 Exp L (Lightweight Gen, ngf=32)"
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expL_lightweight \
    --model pix2pix \
    --direction BtoA \
    --netG unet_256 \
    --ngf 32 \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expL_lightweight.log 2>&1
