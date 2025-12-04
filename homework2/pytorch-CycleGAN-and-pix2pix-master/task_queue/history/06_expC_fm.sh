#!/bin/bash
echo "启动 Exp C (FM Loss U-Net)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_expC_fm \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode lsgan --use_fm \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/expC_fm.log 2>&1
