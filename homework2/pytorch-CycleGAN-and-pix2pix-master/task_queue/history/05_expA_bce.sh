#!/bin/bash
echo "启动 Exp A (Vanilla U-Net)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_expA_bce \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode vanilla \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/expA_bce.log 2>&1
