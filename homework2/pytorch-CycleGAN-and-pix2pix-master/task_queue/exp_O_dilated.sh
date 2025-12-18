#!/bin/bash
echo "启动 Exp O (Dilated Convolution)"
# 核心参数: --use_dilated_conv
# 目的: 在Encoder中使用空洞卷积(Dilation=2)，增大感受野，改善长距离一致性
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expO_dilated \
    --model pix2pix \
    --direction BtoA \
    --use_dilated_conv \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expO_dilated.log 2>&1
