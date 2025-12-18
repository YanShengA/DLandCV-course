#!/bin/bash
echo "启动 Exp J (Instance Norm, BS=1)"
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expJ_instancenorm \
    --model pix2pix \
    --direction BtoA \
    --norm instance \
    --batch_size 1 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expJ_instancenorm.log 2>&1
