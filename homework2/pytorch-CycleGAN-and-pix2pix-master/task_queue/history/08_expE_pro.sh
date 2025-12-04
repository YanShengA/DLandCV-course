#!/bin/bash
echo "启动 Exp E (Pro U-Net)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_expE_pro \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode lsgan \
  --use_vgg --use_fm --use_ssim \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/expE_pro.log 2>&1
