#!/bin/bash
echo "启动 Exp K (No Dropout)"
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expK_nodropout \
    --model pix2pix \
    --direction BtoA \
    --netG unet_256 \
    --no_dropout \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expK_nodropout.log 2>&1
