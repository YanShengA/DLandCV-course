#!/bin/bash
echo "启动 Exp G (Small PatchGAN)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_expG_smallpatch \
  --model pix2pix --direction BtoA --netG unet_256 \
  --batch_size 32 \
  --gan_mode lsgan \
  --netD n_layers --n_layers_D 1 \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/expG_smallpatch.log 2>&1
