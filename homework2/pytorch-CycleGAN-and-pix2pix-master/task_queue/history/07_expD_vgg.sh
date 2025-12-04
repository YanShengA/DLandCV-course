#!/bin/bash
echo "启动 Exp D (VGG Loss U-Net)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_expD_vgg \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode lsgan --use_vgg \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/expD_vgg.log 2>&1
