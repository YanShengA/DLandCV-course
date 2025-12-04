#!/bin/bash
echo "启动 Exp 1 (U-Net Baseline)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_exp1_unet \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --lambda_L1 100 --lambda_GAN 0 \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/exp1_unet.log 2>&1
