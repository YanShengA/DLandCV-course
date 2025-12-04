#!/bin/bash
echo "启动 Exp 3 (CycleGAN Aligned)"
python train.py \
  --dataroot ./datasets/cityscapes --name cityscapes_exp3_cycle \
  --model cycle_gan --direction BtoA --dataset_mode aligned \
  --netG resnet_9blocks \
  --batch_size 4 \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/exp3_cyclegan.log 2>&1
