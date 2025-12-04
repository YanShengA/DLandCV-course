#!/bin/bash
echo "启动 Exp 4 (CycleGAN Unpaired)"
python train.py \
  --dataroot ./datasets/cityscapes_unpaired --name cityscapes_exp4_unpaired \
  --model cycle_gan --direction AtoB --dataset_mode unaligned \
  --netG resnet_9blocks \
  --batch_size 4 \
  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 \
  > logs/exp4_unpaired.log 2>&1
