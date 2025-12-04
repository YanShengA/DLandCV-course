#!/bin/bash
echo "启动 Exp F (PixelGAN)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_expF_pixelgan \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode lsgan \
  --netD pixel \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/expF_pixelgan.log 2>&1
