#!/bin/bash
echo "启动 Exp I (ResNet Generator)"
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expI_resnet \
    --model pix2pix \
    --direction BtoA \
    --netG resnet_9blocks \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expI_resnet.log 2>&1
