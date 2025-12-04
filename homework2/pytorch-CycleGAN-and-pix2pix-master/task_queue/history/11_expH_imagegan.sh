#!/bin/bash
echo "启动 Exp H (ImageGAN)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_expH_imagegan \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode lsgan \
  --netD n_layers --n_layers_D 5 \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/expH_imagegan.log 2>&1
