#!/bin/bash
echo "启动 Exp 2 (Standard Pix2Pix U-Net)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_exp2_standard \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode lsgan \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/exp2_standard.log 2>&1
